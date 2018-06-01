import numpy as np
# import qit
import matplotlib
import matplotlib.pyplot as plt
from random import randint
# import pycosat
import random
import scipy.linalg as la
import pandas as pd
import scipy.sparse as sparse
import scipy.sparse.linalg as lasp
from numba import jit
# for parallel computations on CPU
import pymp
import qit
import scipy

from sat import *
from qaoa import qaoa,get_params
from numpy import array


N=20
beta=0.20
with pymp.Parallel(16) as pp:
    for m in pp.range(1, 20*N):
        clauses=[]
        overlap=[]
        for i in range(500):
            cnf=cnf_random_generator(N,m)
            overlap.append(thermal_overlap(cnf,N,beta))
            clauses.append(m)
        bits=[N]*500
        bet=[beta]*500
        df=pd.DataFrame(data={'bits':np.array(bits).ravel(), 
                            'beta':np.array(bet).ravel(), 
                            'clauses': np.array(clauses).ravel(), 
                            'overlap': np.array(overlap).ravel()})
        df=df[['bits','beta','clauses','overlap']]
        with open('3_SAT_cooling_20.csv', 'a') as f:
            df.to_csv(f, header=False)
