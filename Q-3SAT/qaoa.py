import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as lasp
import scipy



def plus_state(n_qubits):
    """Returns state $|+>^{\otimes n_qubits}$
    Args:
    ----------
        n_qubits (int): number of qubits

    Returns:
    ----------
        numpy array: state $|+>^{\otimes n_qubits}$ 

    """
    return np.array([1]*(2**n_qubits))/np.sqrt(2**n_qubits)

def B_operator(n_qubits):
    """Returns driver Hamiltonian
    Args:
    ----------
        n_qubits (int): number of qubits

    Returns:
    ----------
        scipy sparse array: \sum X_i 
    """
    X=np.array([[0,1],[1,0]])
    B=np.array([[0,1],[1,0]])
    for i in range(1,n_qubits):
        B=sparse.kron(X,B)
    return B

def qaoa_step(state, H, n_qubits, params):
    """Returns a result of one QAOA step
    $e^{-1j*params[1]*B}e^{1j*params[0]*H}|state>$

    Args:
    ----------
        state (array): state  
        H (array): Hamiltonian of interest
        n_qubits (int): number of qubits
        params: parameters of step

    Returns:
    ----------
        scipy sparse array: state after application of $e^{-1j*params[1]*B}e^{1j*params[0]*H}|state>$
    """

    state=lasp.expm_multiply(1j*params[0]*H, state)
    return lasp.expm_multiply(-1j*params[1]*B_operator(n_qubits),state)  

def cost_function(H, n_qubits, p, params):
    """Returns cost function of QAOA and QAOA state
    Args:
    ----------
        H (array): Hamiltonian of interest
        n_qubits (int): number of qubits
        p (int):    number of QAOA steps
        params: parameters of QAOA ($\alpha_1, \beta_1,\ldots, \alpha_n, \beta_n$)

    Returns:
    ----------
        number (float): value of cost function
        numpy array: state

    """
    ini_state=plus_state(n_qubits)
    for i in range(p):
        ini_state=qaoa_step(ini_state,H,n_qubits,params=[params[2*i],params[2*i+1]])
    return ((sparse.spmatrix.getH(ini_state)).dot(H.dot(ini_state))).real, ini_state


def get_params(H, n_qubits, p):
    """Returns optimal; parameters for QAOA
    
    Args:
    ----------
        H (array): Hamiltonian of interest
        n_qubits (int): number of qubits
        p (int):    number of QAOA steps

    Returns:
    ----------
        array: optimal parameters for QAOA
    """
#     function to optimimize
    def fun(x):
#         we minimize f to find max for F 
        return cost_function(H, n_qubits, p, params=x)[0]
# starting point
    params_0=[0.25*np.pi for i in range(2*p)]
    params_min=[0 for i in range(2*p)]
    params_max=[2*np.pi if i%2==0 else np.pi for i in range(2*p)]
    # the bounds required by L-BFGS-B
    bounds = [(low, high) for low, high in zip(params_min, params_max)]
# use method L-BFGS-B because the problem is smooth and bounded
    result = scipy.optimize.minimize(fun, params_0, method="L-BFGS-B",bounds=bounds)
    return [result.x[i] for i in range(2*p)]


def qaoa(H, n_qubits, p):
    """Returns the value of cost function of QAOA for p steps
        and QAOA step 
        See https://arxiv.org/abs/1411.4028 
        Note that we minimize cost function (not maximize like in the paper)

    Args:
    ----------
        H (array): Hamiltonian of interest
        n_qubits (int): number of qubits
        p (int):    number of QAOA steps

    Returns:
    ----------
        number (float): the value of cost function of QAOA for p steps
        numpy array: QAOA state
    """
    params=get_params(H,n_qubits,p)
    return cost_function(H, n_qubits, p, params)
