# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 21:21:20 2022

@author: Admin
"""
import numpy as np
import math

def main(data_file="C:\\Users\\Admin\\Documents\\python\\gurrieri_dataset_npy.npy" ):
    

    PD=np.load(data_file)
    X=PD[:,1]
    Y=PD[:,2]
    k_max=0
    k = math.inf*np.ones(len(PD))
    for i in range(len(PD)):
        for h in list(range(i)) + list(range(i+1,len(PD))):
            if ((X[h]<X[i]) and 
                 (Y[h]-X[h]-Y[i]+X[i]>0) and 
                 (X[h] >2*X[i]-Y[i]) and 
                 (Y[h]<=Y[i]) and 
                 (X[i]-X[h] < k[i])):
                     k[i] = X[i]-X[h]
            elif ((Y[h]>Y[i]) and 
                  (Y[h]-X[h]-Y[i]+X[i]>=0) and 
                  (Y[h]<2*Y[i]-X[i]) and 
                  (X[h]>=X[i]) and 
                  (Y[h]-Y[i]) < k[i]):
                    k[i] = Y[h]-Y[i]
            elif ((Y[h]>Y[i]) and 
                  (X[h]<X[i]) and 
                  (Y[h]<X[h] +2*(Y[i]-X[i])) and
                  (X[i]-X[h] + Y[h]-Y[i]) < k[i]):
                    k[i] = X[i]-X[h] + Y[h]-Y[i]
            elif  (Y[i]-X[i] < k[i]):
                    k[i]=Y[i]-X[i]
        if k[i]>k_max:
            k_max = k[i]
            j = i
    k[j] = math.inf
    print(k)
    print(sorted(k, reverse=True))
    
if __name__=="__main__":
    main()

            
        
            
