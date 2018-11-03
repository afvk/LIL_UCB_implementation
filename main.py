#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 13:38:55 2018

@author: arent
"""


from BAI import BAI

import numpy as np
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('thesis')



dists = [norm(0,2), norm(1,0.5), norm(1.5,1)]
delta = 0.5
epsilon = 0.0
beta = 0.5
sigma = 0.1
labda = 1+10/len(dists)

bai = BAI(dists, delta, epsilon, labda, beta, sigma, store=True)

result, N_it = bai.run()

results = bai.results

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
x = np.linspace(-2, 5, 1000)
pdfs = [dist.pdf(x) for dist in dists]
max_peak = max([max(pdf) for pdf in pdfs])



plt.figure(figsize=(6.5,2.5))
for i,dist in enumerate(dists):
    print(i)
    plt.fill_betweenx(x, pdfs[i]/max_peak*0.25*N_it, color=colors[i], alpha=0.2)
    plt.plot(results['mut'][:,i], color=colors[i])

plt.xlabel(r'$t$')
plt.ylabel(r'$\hat{\mu}$')
plt.tight_layout()
plt.savefig('figures/run.eps')
plt.show()











