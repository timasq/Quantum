import numpy as np
from random import randint
import random
import pandas as pd
import tensorflow as tf
import sys

device_name = "/gpu:0"


with tf.device(device_name):
    # function to generate randomly 3-SAT with n_variables and m_clauses
    def cnf_random_generator(n_variables,m_clauses):
        cnf=[]
        for i in range(m_clauses):
            a=np.array(sorted(random.sample(range(1,n_variables+1),3)))
            b=np.array([(-1)**randint(0,1),(-1)**randint(0,1),(-1)**randint(0,1)])
            cnf.append(a*b)
        return cnf

    # Define projectors as diagonal matrix - array

    P_0=tf.convert_to_tensor(np.array([[1,0]]), dtype=tf.float32)
    P_1=tf.convert_to_tensor(np.array([[0,1]]), dtype=tf.float32)
    I=tf.convert_to_tensor(np.array([[1,1]]), dtype=tf.float32)


    def clause_to_hamiltonian(clause,n_variables):
       
        if clause[0]>0:
            H=P_0
        else:
            H=P_1
        for i in range(1,np.abs(clause[0])):
            H=tf.contrib.kfac.utils.kronecker_product(I,H)
        
        if clause[1]>0:
            H_1=P_0
        else:
            H_1=P_1        
        for i in range(np.abs(clause[0])+1,np.abs(clause[1])):
            H=tf.contrib.kfac.utils.kronecker_product(H,I)
        H=tf.contrib.kfac.utils.kronecker_product(H,H_1)
        
        if clause[2]>0:
            H_2=P_0
        else:
            H_2=P_1        
        for i in range(np.abs(clause[1])+1,np.abs(clause[2])):
            H=tf.contrib.kfac.utils.kronecker_product(H,I)
        H=tf.contrib.kfac.utils.kronecker_product(H,H_2)
        
        for i in range(np.abs(clause[2])+1, n_variables+1):
            H=tf.contrib.kfac.utils.kronecker_product(H,I)
        
        return H



    def cnf_to_hamiltonian(cnf, n_variables):
        H=clause_to_hamiltonian(cnf[0],n_variables)
        for i in range(1,len(cnf)):
            H=tf.add(H,clause_to_hamiltonian(cnf[i],n_variables))
        return H


    def thermal_overlap(cnf, n_variables,beta):
        H=cnf_to_hamiltonian(cnf,n_variables)     
        min_H=tf.reduce_min(H)
        index_min = tf.where(tf.equal(H, min_H))
        trace=tf.reduce_sum(tf.exp(-beta*H))
        degeneracy=tf.to_float(tf.slice(tf.shape(index_min),[0],[1]))
        overlap=degeneracy*tf.exp(-beta*min_H)/trace

        return overlap

# # simulation of evolution
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # you need to initialize all variables
    N=20
    for beta in [1, 2, 3, 4, 5]:
        for m in range(1, 16*N):
            clauses=[]
            overlap=[]
            for i in range(500):
                cnf=cnf_random_generator(N,m)
                overlap.append(sess.run(thermal_overlap(cnf,N,beta)))
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