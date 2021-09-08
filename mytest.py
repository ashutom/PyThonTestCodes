
from decimal import Decimal
import numpy as np
import math
import os


HIGH_DNORM_VAL=0.0000610
LOW_DNORM_VAL=-0.0000610
PATH_FOR_SEARCHING="/dockerx/ashutosh_train/LogNewRunWithOutputTensor/"
TotalDeNorms=0
Total_NAN=0
Total_INF=0


def print_tensor(tensor, dim):
        thisdim=dim
        currentdim=0
        global TotalDeNorms
        global Total_NAN
        global Total_INF
        for iter in tensor:
                currentdim=currentdim+1
                if iter.ndim > 1 :
                        #print("==> [ ")
                        print_tensor(iter,thisdim+':'+str(currentdim))
                        #print("] <==")
                else:
                        counter=0
                        for ii in iter:
                                #print(ii),
                                Isthidenormal=False
                                if math.isnan(ii) :
                                        #print('This is inf, or nan ==' , ii, 'at location = ', thisdim , counter )
                                        Total_NAN=1+Total_NAN
                                elif math.isinf(ii):
                                        Total_INF=1+Total_INF
                                else:
                                        if(ii>0):
                                                if(ii < HIGH_DNORM_VAL):
                                                        Isthidenormal=True
                                        elif(ii<0):
                                                if(ii > LOW_DNORM_VAL):
                                                        Isthidenormal=True

                                if Isthidenormal==True:
                                        #print('This is DNORM VALUE ==' , ii , 'at location = ' , thisdim ,counter )
                                        TotalDeNorms=1+TotalDeNorms

                                counter=counter+1



def ReadAndParseFiles():
        for file in os.listdir(PATH_FOR_SEARCHING):
                if file.endswith(".npy"):
                        print('Processing : ', file )
                        data=np.load(file)
                        global TotalDeNorms
                        global Total_NAN
                        global Total_INF
                        TotalDeNorms=0
                        Total_NAN=0
                        Total_INF=0
                        print_tensor(data,'0')
                        print( ' Total INF = ', Total_INF ,  'Total  NAN = ', Total_NAN, ' TotalDenorms = ', TotalDeNorms )
                        print(" ")
                        print(" ")


if __name__=="__main__":
        ReadAndParseFiles()




#test data
#data_ten=torch.randn(8, 512, 16, 16)
#mytest.print_tensor(data_ten,'0')

