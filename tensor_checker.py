import torch
import sys
import numpy as np
import torch.distributed as dist

target_ops = [torch.nn.Conv2d,
              torch.nn.MaxPool2d,
              torch.nn.ReLU]

rank = -1

parameters = {}
inputs = {}
grad_wrt_outputs = {}
init_params = False

def set_bn_class(bn):
    target_ops.append(bn)

def is_target_op(layer):
    for op in target_ops:
        if isinstance(layer, op):
            return True
    return False

def export_tensor(name, tensor):
    if isinstance(tensor, tuple):
        for i, t in enumerate(tensor):
            export_tensor(name + ("_{}".format(i)), tensor)
        return

    tensor_name = name + "_rank_{}.npy".format(rank)
    print("[tensor_checker] INFO: Rank {}: Exporting {}, shape {}".format(rank, tensor_name, tensor.shape))
    np.save(tensor_name, tensor.cpu().numpy())

def check_nan_inf(tensor, module, tensor_name, tensors_to_export=None):
    if isinstance(tensor, tuple):
        for i, t in enumerate(tensor):
            check_nan_inf(t, module, tensor_name + ("_{}".format(i)))
        return
    if tensor is None:
        return
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    if has_nan or has_inf:
        print("[tensor_checker] ERROR: Rank {}: {} of {} has nan/inf".format(rank, tensor_name, module))
        export_tensor("output_" + tensor_name, tensor)
        if tensors_to_export is not None:
            for name in tensors_to_export:
                export_tensor(name, tensors_to_export[name])
        sys.exit(-1)

# Check backward activation gradient output
# grad_wrt_input = the gradient wrt. the input in the forward pass
# grad_wrt_output = the gradient wrt. the output in the forward pass
def module_backward_hook(module, grad_wrt_input, grad_wrt_output):
    tensors_to_export = {"backwad_grad_wrt_output": grad_wrt_output}
    if isinstance(module, torch.nn.Conv2d):
        tensors_to_export["weight"] = module.weight
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, target_ops[-1]):
        if len(grad_wrt_output) == 1:
            grad_wrt_outputs[module] = grad_wrt_output[0].clone().detach()
        else:
            grad_wrt_outputs[module] = [i.clone().detach() for i in grad_wrt_output]
    check_nan_inf(grad_wrt_input, module, "backward activation grad", tensors_to_export)

# Check forward activation output
def module_forward_hook(module, input, output):
    tensors_to_export = {"forward_input": input}
    if isinstance(module, torch.nn.Conv2d):
        tensors_to_export["weight"] = module.weight
        tensors_to_export["bias"] = module.bias
    check_nan_inf(output, module, "forward activation output", tensors_to_export)

    requires_grad = False
    for n, p in module.named_parameters():
        if p.requires_grad:
            requires_grad = True
            check_nan_inf(p, module, n)

    if requires_grad:
        # Make a copy of input for debugging if nan/inf is found in the gradient
        if len(input) == 1:
            inputs[module] = input[0].clone().detach()
        else:
            inputs[module] = [i.clone().detach() for i in input]

def attach_module_hooks(net):
    for _, l in net._modules.items():
        if not is_target_op(l):
            attach_module_hooks(l)
        else:
            l.register_forward_hook(module_forward_hook)
            l.register_backward_hook(module_backward_hook)
            for n, p in l.named_parameters():
                if p.requires_grad:
                    if not init_params:
                        parameters[p] = (l, n)

def check_grads():
    for i, p in enumerate(parameters):
        module, name = parameters[p]
        check_nan_inf(p.grad, module, "backward {} grad".format(name),
                      tensors_to_export={"backwad_grad_wrt_output": grad_wrt_outputs[module],
                                         "forward_input": inputs[module]})

def reset():
    global inputs
    global grad_wrt_outputs
    # Remove each item to make sure that the tensors are freed
    keys = list(inputs.keys())
    for k in keys:
        del inputs[k]
    keys = list(grad_wrt_outputs.keys())
    for k in keys:
        del grad_wrt_outputs[k]
    inputs = {}
    grad_wrt_outputs = {}
    torch.cuda.empty_cache()

def inject_fault():
    for p in parameters:
        p.grad[0] = float('nan')
        module, name = parameters[p]

def init(net):
    attach_module_hooks(net)
    global init_params
    init_params = True
    global rank
    rank = dist.get_rank()


