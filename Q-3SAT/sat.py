import numpy as np
from random import randint
import random
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
P_0=np.array([1,0])
P_1=np.array([0,1])
I=np.array([1,1])


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
    return sparse.diags(H.toarray()[0])


def ground_state(cnf, n_variables):
    """
    Returns superposition of ground states of 3-SAT Hamiltonian 

    Args:
    ----------
        cnf (numpy array or list): The 3-SAT problem.
        n_variables (int): The number of variables of the 3-SAT problem.

    Returns:
    ----------
        numpy array: superposition of ground states of 3-SAT Hamiltonian 
    """

#   from cnf to Hamiltonian represented array, since it diagonal
    H=cnf_to_hamiltonian(cnf,n_variables).diagonal()
#     find minimal eigenvalue and the superposition of ground states
    min_energy=np.where(H == H.min())[0]
#   mixinf of all ground states
    ground_state=np.zeros(2**n_variables)
    for i in min_energy:
        ground_state[i]=1
    return ground_state


def ground_state_projector(cnf, n_variables):
    """
    Returns projector on a ground space of 3-SAT Hamiltonian 

    Args:
    ----------
        cnf (numpy array or list): The 3-SAT problem.
        n_variables (int): The number of variables of the 3-SAT problem.

    Returns:
    ----------
        scipy sparse diagonal matrix: projector on a ground space of 3-SAT Hamiltonian 
    """

#   from cnf to Hamiltonian represented array, since it diagonal
    H=cnf_to_hamiltonian(cnf,n_variables).diagonal()
#     find minimal eigenvalue and the superposition of ground states
    min_energy=np.where(H == H.min())[0]
#   mixinf of all ground states
    ground_state=np.zeros(2**n_variables)
    for i in min_energy:
        ground_state[i]=1
    return sparse.diags(ground_state)


# evolution function: returns overlap between 
# initial_state=initial_state(10)
def thermal_overlap(cnf,n_variables, beta):
    """
    Returns overlap between thermal state of 3-SAT Hamiltonian

    Args:
    ----------
        cnf (numpy array or list): The 3-SAT problem.
        n_variables (int): The number of variables of the 3-SAT problem.
        beta (float): The inverse temperature of the thermal state.

    Returns:
    ----------
        number: overlap between thermal state of 3-SAT Hamiltonian
    """
#     generate superposition of ground states
    g_state=ground_state(cnf,n_variables)
#     generate initial state
#     i_state=initial_state
#     generate cooling operator
    H=cnf_to_hamiltonian(cnf,n_variables).toarray()[0,:]
    U=np.exp(-beta*H)
    
    return np.matmul(g_state, U.T/np.sum(U))  