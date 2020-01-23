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
    
    for iii in range(kkk*5,(kkk+1)*5):
        
        jjj=2
        """ load problem data and settings """
        with open('../problems/problem_data_'+str(iii)+'_'+str(jjj)+'.pkl', 'rb') as f:
            problem = pickle.load(f)    
        x=problem['x']
        z=problem['z']
        y=problem['y']
        nthreads=1
        cy=1
        B=50
        
        CZ=[0.1,0.2,0.3,0.4,0.5]
        METHODS=['greedyN','greedyEI','ThompsonFixed','ThompsonAdapt']
        
        RESULTS={}
        for i in range(5):
            for j in range(4):
                History=test_env_fun.test(x,y,z,CZ[i],cy,B,nthreads,problem,METHODS[j])
                RESULTS[(METHODS[j],CZ[i])]=History
    
        with open('../results_vary_costs/histories_data_'+str(iii)+'.pkl', 'wb') as f:
            pickle.dump(RESULTS,f)
            
if __name__=='__main__':
    go(int(sys.argv[1]))
        
    