#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 13:39:12 2018

@author: arent
"""

import numpy as np




class BAI:
    
    def __init__(self, dists, delta, epsilon, labda, beta, sigma, store=False):
        self.dists = dists
        self.delta = delta
        self.epsilon = epsilon
        self.labda = labda
        self.beta = beta
        self.sigma = sigma
        self.store = store
        
        self.n = len(dists)
        self.T, self.mut, self.t = self.initialize()
        
        if store:
            self.results = {'T':[self.T.copy()],
                            'mut':[self.mut.copy()]}


    def initialize(self):
        T = np.ones(self.n)
        mut = np.zeros_like(T)
        t = self.n
        for i in range(self.n):
            mut[i] = self.dists[i].rvs()
        
        return T, mut, t
    
    
    def sample(self, i):
        samp = self.dists[i].rvs()
        self.T[i] += 1
    
        self.mut[i] += (samp - self.mut[i])/self.T[i]

    
    def run(self):
        print('Starting BAI run:')

        it = 0
        while True:
            if self.check_stop():
                break
            
            i = self.choose_arm()
            self.sample(i)
            
            if self.store:
                self.results['T'].append(self.T.copy())
                self.results['mut'].append(self.mut.copy())
            
            print(it, i, self.T, self.mut)
            it += 1
        
        if self.store:
            self.results['T'] = np.vstack(self.results['T'])
            self.results['mut'] = np.vstack(self.results['mut'])
        
        
        result = np.argmax(self.T)
        print('Finished BAI, best arm is %i'%result)
        return result, it
    
        
    def check_stop(self):
        stop = False
        for i in range(self.n):
            arr = [elem for j,elem in enumerate(self.T) if i!=j]
            stop = (self.T[i] >= 1+self.labda*np.sum(arr))
            
            if stop:
                break
        
        return stop
    
    
    def choose_arm(self):
        bound = (1+self.beta)*(1+np.sqrt(self.epsilon)) \
                    *np.sqrt(2*self.sigma**2*(1+self.epsilon)*
                             np.log(np.log((1+self.epsilon)*self.T+2)
                             /self.delta)/self.T)
        UCB = self.mut + bound
        return np.argmax(UCB)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        