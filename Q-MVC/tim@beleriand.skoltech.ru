import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as lasp
import scipy
import tensorflow as tf
import pandas as pd



# Define Operators
P_1=np.array([0,1])
I=np.array([1,1])
P_0=np.array([1,0])
# X=np.array([0,1],[1,0]])



# interaction terms
def OnEdgeTerms(operator ,edge, N):
    if edge[0]<edge[1]:
        m,n=edge[0], edge[1]
    else:
        n,m=edge[0], edge[1]
        
    h=operator   
    for i in range(m):
        h=sparse.kron(I,h)
    for i in range(m+1, n):
        h=sparse.kron(h,I)
    h=sparse.kron(h,operator)
    for i in range(n+1, N):
        h=sparse.kron(h,I)
    
    return h



#on-site terms
def OnSiteTerms(operator, n, N):
    h=operator
    for i in range(n):
        h=sparse.kron(I,h)
    for i in range(n+1, N):
        h=sparse.kron(h,I)
    
    return 0.5*h



# function transforms Graph to Hamiltonian
def FromGraphToHamiltonian(graph):
    N=graph.number_of_nodes()
    
    # on-site terms
    h=OnSiteTerms(P_1, 0, N)
    for n in range(1, N):
        h+=OnSiteTerms(P_1, n, N)
    
    # interaction terms
    for edge in graph.edges:
        h+=OnEdgeTerms(P_0, edge, N)
    
    return h


# Computational graph
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


with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    N_bits=15
    beta_value=1
    for m in range(1, 10*N_bits):
        xc=[]
        over=[]
        # average degree
		k=2
		p=k/(N_bits-1)
        for i in range(500):
            graph=nx.gnp_random_graph(N_bits, p=p)
            H_mvc=FromGraphToHamiltonian(graph).todense()
            min_energy, ov = sess.run([index_min,overlap], feed_dict={N: N_bits, beta: beta_value, H: H_mvc})
            xc.append(bin(min_energy[0]).count("1"))
            over.append(ov)

        bits=[N_bits]*500
        bet=[beta_value]*500
        df=pd.DataFrame(data={'bits':np.array(bits).ravel(), 
                            'beta':np.array(bet).ravel(), 
                            'n_0': np.array(xc).ravel(), 
                            'overlap': np.array(over).ravel()})
        df=df[['bits','beta','n_0','overlap']]
        with open('MVC_cooling.csv', 'a') as f:
            df.to_csv(f, header=False)