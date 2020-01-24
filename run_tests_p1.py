#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:28:45 2020

runs tests with different costs

@author: Hook
"""

import sys
import test_env_fun
import pickle
import numpy as np

def go(kkk):
    
    for iii in range(10*kkk,10*(kkk+1)):
        
        jjj=2
        """ load problem data and settings """
        with open('../problems/problem_data_'+str(iii)+'_'+str(jjj)+'.pkl', 'rb') as f:
            problem = pickle.load(f)    
        x=problem['x']
        z=problem['z']
        y=problem['y']
        nthreads=1
        cy=1
        cz=0.2
        B=50
        P1=[0.5,0.6,0.7,0.8,0.9]
        
        METHODS=['greedyN','greedyEI','ThompsonFixed','ThompsonAdapt']
        j=2
        
        RESULTS={}
        for i in range(5):
            History=test_env_fun.test(x,y,z,cz,cy,B,nthreads,problem,METHODS[j],p1_for_thompson_fixed=P1[i])
            RESULTS[(METHODS[j],P1[i])]=History
    
        with open('../results_vary_p1/histories_data_'+str(iii)+'.pkl', 'wb') as f:
            pickle.dump(RESULTS,f)
            
if __name__=='__main__':
    go(int(sys.argv[1]))
        
    