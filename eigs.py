import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import DiagonalGate, QFTGate, StatePreparation, UnitaryGate
from qiskit.quantum_info import Statevector
from scipy.linalg import expm, eigvals
from lchs import spectral_diff_adv, spectral_diff_adv_op

n_qubits = 4

D = 1
N = 2**n_qubits
L = 2 * np.pi               
t = 1  
x = np.linspace(0, L, N, endpoint=False)
dx = L / N

# Parameters
c_L = 0.0
c_H = 0.5
#c = c_H*np.ones_like(x)
c = -4*(c_H -c_L)/L**2 *x**2 + 4*(c_H -c_L)/L *x 
#cx = np.zeros_like(x)
cx = -8*(c_H - c_L)/L**2 *x + 4*(c_H -c_L)/L 

a = -0.1
a_x = a*np.ones_like(x)+0.0*np.sin(2*np.pi*x/L)



U, A = spectral_diff_adv_op(n_qubits, t, cx, c, D, L, a_x, 0.1)

print(np.real(eigvals(A)))

