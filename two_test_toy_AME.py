#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:07:45 2020

two_test_AME for 2d synth problems

@author: Hook
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import norm

class toy_prospector:
    
    """ set up arrays for data and lists of status as well as GP hyperparameters
    and some useful matrices """
    def __init__(self,x,cz,cy,B,hypers,acquisition_function):
        
        self.x=x
        self.n=len(x)
        self.z=np.zeros(self.n)*np.nan
        self.y=np.zeros(self.n)*np.nan
        
        self.tt=[]
        self.tu=[]
        self.uu=list(range(self.n))
        
        self.tz=[]
        self.ty=[]
        
        self.N=10
        self.tau=0
        self.cz=cz
        self.cy=cy
        self.p1=(2*self.cz+self.cy)/(3*self.cz+self.cy)
        self.b=B
        self.upfreq=5
        self.upcount=0
        self.acquisition_function=acquisition_function
        
        self.az=hypers['az']
        self.bz=hypers['bz']
        self.lz=hypers['lz']
        self.ay=hypers['ay']
        self.by=hypers['by']
        self.lyx=hypers['lyx']
        self.lyz=hypers['lyz']
        self.Dx=euclidean_distances(self.x.reshape(-1,1),self.x.reshape(-1,1),squared=True)
        self.SIG_z=self.az**2*np.exp(-self.Dx/(2*self.lz**2))+self.bz**2*np.identity(self.n)

    """ draw samples from posterior """  
    def sample(self,nz,ny):  
        zsamples=np.zeros((self.n,nz))
        zsamples[self.tu+self.tt,:]=np.repeat(self.z[self.tu+self.tt].reshape(-1,1),nz,1)
        ysamples=np.zeros((self.n,nz*ny))
        ysamples[self.tt,:]=np.repeat(self.y[self.tt].reshape(-1,1),nz*ny,1)
        SIG_z_pos=self.SIG_z[np.ix_(self.uu,self.uu)]-np.matmul(self.SIG_z[np.ix_(self.uu,self.tu+self.tt)],np.linalg.solve(self.SIG_z[np.ix_(self.tu+self.tt,self.tu+self.tt)],self.SIG_z[np.ix_(self.tu+self.tt,self.uu)]))
        mu_z_pos=np.matmul(self.SIG_z[np.ix_(self.uu,self.tu+self.tt)],np.linalg.solve(self.SIG_z[np.ix_(self.tu+self.tt,self.tu+self.tt)],self.z[self.tu+self.tt]))
        zsamples[self.uu,:]=np.random.multivariate_normal(mu_z_pos,SIG_z_pos,nz).T
        for i in range(nz):
            Dz=euclidean_distances(zsamples[:,i].reshape(-1,1),zsamples[:,i].reshape(-1,1),squared=True)
            SIG_y=self.ay**2*np.exp(-self.Dx*self.lyx**2/2-Dz*self.lyz**2/2)+self.by**2*np.identity(self.n)
            SIG_y_pos=SIG_y[np.ix_(self.uu+self.tu,self.uu+self.tu)]-np.matmul(SIG_y[np.ix_(self.uu+self.tu,self.tt)],np.linalg.solve(SIG_y[np.ix_(self.tt,self.tt)],SIG_y[np.ix_(self.tt,self.uu+self.tu)]))
            mu_y_pos=np.matmul(SIG_y[np.ix_(self.uu+self.tu,self.tt)],np.linalg.solve(SIG_y[np.ix_(self.tt,self.tt)],self.y[self.tt]))
            ysamples[self.uu+self.tu,i*ny:(i+1)*ny]=np.random.multivariate_normal(mu_y_pos,SIG_y_pos,ny).T
        return zsamples,ysamples
    
    """ estimate threshold for approx reward function """  
    def estiamte_tau(self):
        nz=5
        ny=5
        zsamples,ysamples=self.sample(nz,ny)
        self.tau=np.median([np.sort(ysamples[:,i])[-self.N] for i in range(ny*nz)])

    def greedyN(self):
        self.upcount+=1
        if self.upcount==self.upfreq:
            self.estiamte_tau()
            self.upcount=0
        """ greedy acquisition function """
        nz=10
        zq=np.linspace(-2,2,nz)
        pz=norm.pdf(zq)
        pz=pz/np.sum(pz)
        P=np.zeros((self.n,nz))
        P[self.tt,:]=np.repeat((self.y[self.tt]>self.tau).reshape(-1,1),nz,1)
        zquad=np.zeros(self.n)
        zquad[self.tu+self.tt]=self.z[self.tu+self.tt]
        SIG_z_pos=self.SIG_z[np.ix_(self.uu,self.uu)]-np.matmul(self.SIG_z[np.ix_(self.uu,self.tu+self.tt)],np.linalg.solve(self.SIG_z[np.ix_(self.tu+self.tt,self.tu+self.tt)],self.SIG_z[np.ix_(self.tu+self.tt,self.uu)]))
        sig_z_pos=np.diag(SIG_z_pos)**0.5
        mu_z_pos=np.matmul(self.SIG_z[np.ix_(self.uu,self.tu+self.tt)],np.linalg.solve(self.SIG_z[np.ix_(self.tu+self.tt,self.tu+self.tt)],self.z[self.tu+self.tt]))   
        for i in range(nz):
            zquad[self.uu]=mu_z_pos+zq[i]*sig_z_pos 
            Dz=euclidean_distances(zquad.reshape(-1,1),zquad.reshape(-1,1),squared=True)
            SIG_y=self.ay**2*np.exp(-self.Dx*self.lyx**2/2-Dz*self.lyz**2/2)+self.by**2*np.identity(self.n)
            SIG_y_pos=SIG_y[np.ix_(self.uu+self.tu,self.uu+self.tu)]-np.matmul(SIG_y[np.ix_(self.uu+self.tu,self.tt)],np.linalg.solve(SIG_y[np.ix_(self.tt,self.tt)],SIG_y[np.ix_(self.tt,self.uu+self.tu)]))
            sig_y_pos=np.diag(SIG_y_pos)**0.5
            mu_y_pos=np.matmul(SIG_y[np.ix_(self.uu+self.tu,self.tt)],np.linalg.solve(SIG_y[np.ix_(self.tt,self.tt)],self.y[self.tt])) 
            P[self.uu+self.tu,i]=1-norm.cdf(np.divide(self.tau-mu_y_pos,sig_y_pos))
        alpha=np.matmul(P,pz)
        """ greedy controller """
        uua=[i for i in self.uu if i not in self.tz]
        tua=[i for i in self.tu if i not in self.ty]
        iuu=uua[np.argmax(alpha[uua])]
        if len(tua)>0:
            itu=tua[np.argmax(alpha[tua])]
            r1=np.dot(pz,np.maximum(P[itu],P[iuu]))/(self.cz+self.cy)    
            r2=alpha[itu]/self.cy
            if r2>r1:
                return itu
        return iuu

    def greedyEI(self):
        """ greedy EI acquisition function """
        ymax=np.max(self.y[self.tt])
        nz=10
        zq=np.linspace(-2,2,nz)
        pz=norm.pdf(zq)
        pz=pz/np.sum(pz)
        EI=np.zeros((self.n,nz))
        EI[self.tt,:]=0
        zquad=np.zeros(self.n)
        zquad[self.tu+self.tt]=self.z[self.tu+self.tt]
        SIG_z_pos=self.SIG_z[np.ix_(self.uu,self.uu)]-np.matmul(self.SIG_z[np.ix_(self.uu,self.tu+self.tt)],np.linalg.solve(self.SIG_z[np.ix_(self.tu+self.tt,self.tu+self.tt)],self.SIG_z[np.ix_(self.tu+self.tt,self.uu)]))
        sig_z_pos=np.diag(SIG_z_pos)**0.5
        mu_z_pos=np.matmul(self.SIG_z[np.ix_(self.uu,self.tu+self.tt)],np.linalg.solve(self.SIG_z[np.ix_(self.tu+self.tt,self.tu+self.tt)],self.z[self.tu+self.tt]))   
        for i in range(nz):
            zquad[self.uu]=mu_z_pos+zq[i]*sig_z_pos 
            Dz=euclidean_distances(zquad.reshape(-1,1),zquad.reshape(-1,1),squared=True)
            SIG_y=self.ay**2*np.exp(-self.Dx*self.lyx**2/2-Dz*self.lyz**2/2)+self.by**2*np.identity(self.n)
            SIG_y_pos=SIG_y[np.ix_(self.uu+self.tu,self.uu+self.tu)]-np.matmul(SIG_y[np.ix_(self.uu+self.tu,self.tt)],np.linalg.solve(SIG_y[np.ix_(self.tt,self.tt)],SIG_y[np.ix_(self.tt,self.uu+self.tu)]))
            sig_y_pos=np.diag(SIG_y_pos)**0.5
            mu_y_pos=np.matmul(SIG_y[np.ix_(self.uu+self.tu,self.tt)],np.linalg.solve(SIG_y[np.ix_(self.tt,self.tt)],self.y[self.tt])) 
            EI[self.uu+self.tu,i]=(mu_y_pos-ymax)*norm.cdf(np.divide(mu_y_pos-ymax,sig_y_pos))+sig_y_pos*norm.pdf(np.divide(mu_y_pos-ymax,sig_y_pos))
        alpha=np.matmul(EI,pz)
        """ greedy controller """
        uua=[i for i in self.uu if i not in self.tz]
        tua=[i for i in self.tu if i not in self.ty]
        iuu=uua[np.argmax(alpha[uua])]
        if len(tua)>0:
            itu=tua[np.argmax(alpha[tua])]
            r1=np.dot(pz,np.maximum(EI[itu],EI[iuu]))/(self.cz+self.cy)    
            r2=alpha[itu]/self.cy
            if r2>r1:
                return itu
        return iuu

    def Thompson(self):
        zsample,ysample=self.sample(1,1)
        alpha=ysample.reshape(-1)
        uua=[i for i in self.uu if i not in self.tz]
        tua=[i for i in self.tu if i not in self.ty]
        iuu=uua[np.argmax(alpha[uua])]
        if len(tua)>0 and np.random.rand()>self.p1:
            itu=tua[np.argmax(alpha[tua])]
            return itu
        else:
            return iuu

    def reverse_cumsum(self,a):
        return np.sum(a)-np.cumsum(a)+a
    
    """ adapt action 1 probability """
    def update_p1(self):
        uua=[i for i in self.uu if i not in self.tz]
        tua=[i for i in self.tu if i not in self.ty]
        if len(tua)>0:
            """ draw a load of samples """
            nz=5
            ny=5
            zsamples,ysamples=self.sample(nz,ny)
            """ estimate threshold for approx reward function """  
            self.tau=np.median([np.sort(ysamples[:,i])[-self.N] for i in range(ny*nz)])
            """ get a load of thompson samlpes from I_UU """
            TH_UU=[uua[np.argmax(ysamples[uua,i])] for i in range(ny*nz)]
            """ quadrature for integrating over z """
            mz=10
            zq=np.linspace(-2,2,mz)
            pz=norm.pdf(zq)
            pz=pz/np.sum(pz)
            """ z distribution for thompson points """
            SIG_z_pos_th_uu=np.diag(self.SIG_z[np.ix_(TH_UU,TH_UU)]-np.matmul(self.SIG_z[np.ix_(TH_UU,self.tu+self.tt)],np.linalg.solve(self.SIG_z[np.ix_(self.tu+self.tt,self.tu+self.tt)],self.SIG_z[np.ix_(self.tu+self.tt,TH_UU)])))**0.5
            mu_z_pos_th_uu=np.matmul(self.SIG_z[np.ix_(TH_UU,self.tu+self.tt)],np.linalg.solve(self.SIG_z[np.ix_(self.tu+self.tt,self.tu+self.tt)],self.z[self.tu+self.tt]))   
            """ x and z values and probabilities """
            x_th_uu=np.concatenate([self.x[TH_UU] for i in range(mz)])
            z_th_uu=np.concatenate([mu_z_pos_th_uu+SIG_z_pos_th_uu*zq[i] for i in range(mz)])
            p_th_uu=np.repeat(pz,nz*ny)/(ny*nz)
            """ prior on I_tt points """
            Dz_tt=euclidean_distances(self.z[self.tt].reshape(-1,1),self.z[self.tt].reshape(-1,1),squared=True)
            SIG_tt=self.ay**2*np.exp(-self.Dx[np.ix_(self.tt,self.tt)]*self.lyx**2/2-Dz_tt*self.lyz**2/2)+self.by**2*np.identity(len(self.tt))
            """ expected rewards for thompson samples """
            Dx_th_uu=euclidean_distances(x_th_uu.reshape(-1,1),self.x[self.tt].reshape(-1,1),squared=True)
            Dz_th_uu=euclidean_distances(z_th_uu.reshape(-1,1),self.z[self.tt].reshape(-1,1),squared=True)
            SIG_y_th_uu=self.ay**2*np.exp(-Dx_th_uu*self.lyx**2/2-Dz_th_uu*self.lyz**2/2)
            mu_y_th_uu=np.matmul(SIG_y_th_uu,np.linalg.solve(SIG_tt,self.y[self.tt]))
            sig_y_th_uu=(self.ay**2+self.by**2-np.sum(np.multiply(SIG_y_th_uu.T,np.linalg.solve(SIG_tt,SIG_y_th_uu.T)),0))**0.5
            r_th_uu=1-norm.cdf(np.divide(self.tau-mu_y_th_uu,sig_y_th_uu))
            """ expected rewards for points in I_TU """
            Dx_tu=euclidean_distances(self.x[tua].reshape(-1,1),self.x[self.tt].reshape(-1,1),squared=True)
            Dz_tu=euclidean_distances(self.z[tua].reshape(-1,1),self.z[self.tt].reshape(-1,1),squared=True)
            SIG_y_tu=self.ay**2*np.exp(-Dx_tu*self.lyx**2/2-Dz_tu*self.lyz**2/2)
            mu_y_tu=np.matmul(SIG_y_tu,np.linalg.solve(SIG_tt,self.y[self.tt]))
            sig_y_tu=(self.ay**2+self.by**2-np.sum(np.multiply(SIG_y_tu.T,np.linalg.solve(SIG_tt,SIG_y_tu.T)),0))**0.5
            r_tu=1-norm.cdf(np.divide(self.tau-mu_y_tu,sig_y_tu))   
            """ now put rewards and probabilities into big sorted structure """
            S=np.zeros((mz*ny*nz+len(tua),5))
            S[:mz*ny*nz,0]=r_th_uu
            S[:mz*ny*nz,2]=r_th_uu
            S[:mz*ny*nz,4]=p_th_uu
            S[mz*ny*nz:,0]=r_tu
            S[mz*ny*nz:,1]=r_tu
            S[mz*ny*nz:,3]=1    
            """ sort by reward """
            ix=np.argsort(S[:,0])
            S=S[ix,:]
            """ get optimal costs and rewards and choosing proability """
            S[:,1]=self.reverse_cumsum(S[:,1])
            S[:,2]=self.reverse_cumsum(S[:,2])
            S[:,3]=self.reverse_cumsum(S[:,3])
            S[:,4]=self.reverse_cumsum(S[:,4])
            M1=np.divide(self.b-self.cy*S[:,3],self.cy*S[:,4]+self.cz)
            R=S[:,1]+M1*S[:,2]
            m1=M1[np.argmax(R)]
            m2=(self.b-self.cz*m1)/self.cy
            self.p1=m1/(m1+m2)
        else:
            self.p1=(2*self.cz+self.cy)/(3*self.cz+self.cy)
        
    def ThompsonAdapt(self):
        self.upcount+=1
        if self.upcount==self.upfreq:
            self.update_p1()
            self.upcount=0  
        return self.Thompson()
    
    def pick(self):
        if self.acquisition_function=='greedyN':
            return self.greedyN() 
        if self.acquisition_function=='greedyEI':
            return self.greedyEI()
        if self.acquisition_function=='ThompsonFixed':
            return self.Thompson()
        if self.acquisition_function=='ThompsonAdapt':
            return self.ThompsonAdapt()
        print('enter a valid acquisition function!')
        
    def plot(self,x,y,z):
        top=np.argsort(y)[-10:]
        plt.figure(figsize=(15,5))
        plt.subplot(121)        
        plt.plot(x[top],z[top],'.',color='red')
        plt.plot(x[self.tu],z[self.tu],'s',color='black')
        plt.plot(x[self.tt],z[self.tt],'d',color='red')
        plt.plot(x[self.tz],z[self.tz],'x',markersize=7.5,markeredgewidth=3,color='black')
        plt.plot(x[self.ty],z[self.ty],'x',markersize=7.5,markeredgewidth=3,color='red')
        plt.scatter(x,z,10,y)
        plt.xlabel('x')
        plt.ylabel('z')
        plt.subplot(122)
        plt.scatter(x[self.tt],z[self.tt],50,y[self.tt])
        plt.plot(x[self.tu],z[self.tu],'s',color='black')
        plt.plot(x[self.uu],np.ones(len(self.uu))*np.min(z),'x')
        plt.plot(x[self.tz],z[self.tz],'x',markersize=7.5,markeredgewidth=3,color='black')
        plt.plot(x[self.ty],z[self.ty],'x',markersize=7.5,markeredgewidth=3,color='red')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.show()
        
