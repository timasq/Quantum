import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as lasp
import scipy


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



# calculating sum of ground states of MIS Hamiltonian
def thermal_overlap(graph,beta):
    # from cnf to Hamiltonian represented array, since it diagonal
    H=FromGraphToHamiltonian(graph).toarray()[0]
    #H=sparse.csr_matrix(H).toarray()[0,:]
    # find minimal eigenvalue and the superposition of ground states
    min_H=H.min()
    min_energy=np.where(H == min_H)[0]
    # mixinf of all grround states
    n_variables=graph.number_of_nodes()
    # ground_state=sparse.lil_matrix((1, 2**n_variables), dtype=int)
    # for i in min_energy:
    #     ground_state[0,i]=1

    # mvc="{0:b}".format(min_energy[0])
    # # print(mvc)
    n0=bin(min_energy[0]).count("1")
    # print(n0)
    trace=0
    for i in range(len(H)):
        trace+=np.exp(-beta*H[i])
    
    overlap=len(min_energy)*np.exp(-beta*min_H)/trace

    return n0, overlap



# def thermal_overlap(graph, beta):
# #     generate superposition of ground states
#     g_state=ground_state(graph)[0]
# #     generate initial state
# #     i_state=initial_state
# #     generate cooling operator
#     H=FromGraphToHamiltonian(graph)
#     U=np.exp(-beta*H)
#     trace=U.sum()
    
#     return ((g_state.dot(U))/trace)[0]