#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:22:32 2020

function version of test enviroment for two_test_toy AME

@author: Hook
"""

import two_test_toy_AME
import numpy as np

def test(x,y,z,cz,cy,B,nthreads,problem,acquisition_function):
    """ lists to store simulation history """
    History=[]
    """ initialize prospector """
    P=two_test_toy_AME.toy_prospector(x,cz,cy,B,problem,acquisition_function)
    """ start by single test of subject 0 """
    P.uu.remove(0)
    P.tt.append(0)
    P.b=P.b-cz-cy
    P.z[0]=z[0]
    P.y[0]=y[0]
    """ now follow controller """
    """ set up initial jobs seperately """
    workers=[(0,0) for i in range(nthreads)]
    finish_time=np.zeros(nthreads)
    for i in range(nthreads):
        ipick=P.pick()
        if ipick in P.uu:
            workers[i]=(ipick,'z')
            P.tz.append(ipick)
            P.b-=cz
            finish_time[i]=np.random.uniform(cz,cz*2)
        else:
            workers[i]=(ipick,'y')
            P.ty.append(ipick)
            P.b-=cy
            finish_time[i]=np.random.uniform(cy,cy*2)
    """ main loop """
    while P.b>=cy:
        i=np.argmin(finish_time)
        t=finish_time[i]
        idone=workers[i][0]
        if workers[i][1]=='z':
            P.tz.remove(idone)
            P.z[idone]=z[idone]
            P.uu.remove(idone)
            P.tu.append(idone)
            History.append((idone,'z'))
        else:
            P.ty.remove(idone)
            P.y[idone]=y[idone]
            P.tu.remove(idone)
            P.tt.append(idone)
            History.append((idone,'y'))
        ipick=P.pick()
        if ipick in P.uu:
            workers[i]=(ipick,'z')
            P.tz.append(ipick)
            P.b-=cz
            finish_time[i]=t+np.random.uniform(cz,cz*2)
        else:
            workers[i]=(ipick,'y')
            P.ty.append(ipick)
            P.b-=cy
            finish_time[i]=t+np.random.uniform(cy,cy*2)
    """ let final jobs finish """    
    for i in range(nthreads):
        i=np.argmin(finish_time)
        t=finish_time[i]
        idone=workers[i][0]
        if workers[i][1]=='z':
            P.tz.remove(idone)
            P.z[idone]=z[idone]
            P.uu.remove(idone)
            P.tu.append(idone)
            History.append((idone,'z'))
        else:
            P.ty.remove(idone)
            P.y[idone]=y[idone]
            P.tu.remove(idone)
            P.tt.append(idone)
            History.append((idone,'y'))
        workers[i]=None
        finish_time[i]=np.inf
    return History