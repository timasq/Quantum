import numpy as np
from random import randint
import random
import pandas as pd
import tensorflow as tf
import sys

import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as lasp
import scipy


def cnf_random_generator(n_variables,m_clauses):
    """
    Returns a random 3-SAT problem in CNF form 
    (see https://en.wikipedia.org/wiki/Boolean_satisfiability_problem#3-satisfiability)

    Args:
    ----------
        n_variables (int): The number of boolean variables of 3-SAT problem.
        m_clauses (int): The number of clauses of CNF form of 3-SAT problem.

    Returns:
    ----------
        numpy array: random 3-SAT problem in CNF form
    """    
    cnf=[]
    for i in range(m_clauses):
        a=np.array(sorted(random.sample(range(1,n_variables+1),3)))
        b=np.array([(-1)**randint(0,1),(-1)**randint(0,1),(-1)**randint(0,1)])
        cnf.append(a*b)
    
    return cnf

# Define projectors as diagonal matrix - array

def clause_to_hamiltonian(clause,n_variables):
    """
    Returns a Hamiltonian of clause of 3-SAT problem in CNF form 
    Args:
    ----------
        clause (1D numpy array or list): The clause.
        n_variables (int): The number of variables of the 3-SAT problem.

    Returns:
    ----------
        scipy sparse array: Hamiltonian of clause of 3-SAT problem
    """
    P_0=np.array([1,0], dtype=np.float32)
    P_1=np.array([0,1], dtype=np.float32)
    I=np.array([1,1], dtype=np.float32)

    if np.sign(clause[0])==1:
        H=P_0
    else:
        H=P_1
    for i in range(1,np.abs(clause[0])):
        H=sparse.kron(I,H)
    
    if np.sign(clause[1])==1:
        H_1=P_0
    else:
        H_1=P_1        
    for i in range(np.abs(clause[0])+1,np.abs(clause[1])):
        H=sparse.kron(H,I)
    H=sparse.kron(H,H_1)
    
    if np.sign(clause[2])==1:
        H_2=P_0
    else:
        H_2=P_1        
    for i in range(np.abs(clause[1])+1,np.abs(clause[2])):
        H=sparse.kron(H,I)
    H=sparse.kron(H,H_2)
    
    for i in range(np.abs(clause[2])+1, n_variables+1):
        H=sparse.kron(H,I)
    
    return H

def cnf_to_hamiltonian(cnf, n_variables):
    """
    Returns a Hamiltonian of 3-SAT problem in CNF form 
    
    Args:
    ----------
        cnf (numpy array or list): The 3-SAT problem.
        n_variables (int): The number of variables of the 3-SAT problem.

    
    Parameters:
    ----------
        scipy sparse array: Hamiltonian of 3-SAT problem
    """

    H=clause_to_hamiltonian(cnf[0],n_variables)
    for i in range(1,len(cnf)):
        H+=clause_to_hamiltonian(cnf[i],n_variables)
    return H
# Hamiltonian=tf.while_loop(n>m, tf.add(), name='Hamiltonian')


    # input data
H = tf.placeholder(tf.float32, shape=(None,None), name='Hamiltonian')
beta = tf.placeholder(tf.float32, name='beta')
N=tf.placeholder(tf.float32, name='N')
    # calculations
# H = tf.py_func(cnf_to_hamiltonian, [cnf, N], tf.float32, name='Hamiltonian')
min_H=tf.reduce_min(H, name='Hamiltonian_min')
index_min = tf.where(tf.equal(H, min_H), name='Indicies_min')
trace=tf.reduce_sum(tf.exp(-beta*H), name='trace')
degeneracy=tf.to_float(tf.slice(tf.shape(index_min),[0],[1]), name='degeneracy_gs')
overlap=degeneracy*tf.exp(-beta*min_H)/trace


import time
start_time = time.time()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    N_bits=20
    beta_value=2
    for m in range(1, 10*N_bits):
        clauses=[]
        over=[]
        for i in range(500):
            cnf_random=cnf_random_generator(N_bits,m)
            H_cnf=cnf_to_hamiltonian(cnf_random,N_bits).todense()
            over.append(sess.run(overlap, feed_dict={N: N_bits, beta: beta_value, H: H_cnf}))
            clauses.append(m)

        bits=[N_bits]*500
        bet=[beta_value]*500
        df=pd.DataFrame(data={'bits':np.array(bits).ravel(), 
                            'beta':np.array(bet).ravel(), 
                            'clauses': np.array(clauses).ravel(), 
                            'overlap': np.array(over).ravel()})
        df=df[['bits','beta','clauses','overlap']]
        with open('3_SAT_cooling_20.csv', 'a') as f:
            df.to_csv(f, header=False)