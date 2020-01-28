#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:11:40 2020

runs a test using test env class

@author: Hook
"""

import test_env_class
import numpy as np
import two_test_toy_AME
import pickle

iii=0
jjj=2
with open('../data/problems/problem_data_'+str(iii)+'_'+str(jjj)+'.pkl', 'rb') as f:
    problem = pickle.load(f)  

x=problem['x']
z=problem['z']
y=problem['y']
nthreads=5
cy=1
cz=0.2
B=50

ame=two_test_toy_AME.toy_prospector(x,cz,cy,B,problem,'ThompsonEntropy')
sim_env=test_env_class.ParallelScreener(ame,y,z,cz,cy,1)
sim_env.full_screen(ploton=True)