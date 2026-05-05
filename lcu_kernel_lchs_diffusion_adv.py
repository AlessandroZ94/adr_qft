import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import DiagonalGate, QFTGate, StatePreparation, UnitaryGate
from qiskit.quantum_info import Statevector
from scipy.linalg import expm
from lchs import lchs , spectral_diff_adv

n_qubits = 4

D = 0.1
N = 2**n_qubits
L = 2 * np.pi               
t = 0.4   
x = np.linspace(0, L, N, endpoint=False)
dx = L / N

# Parameters
c_L = 0.0
c_H = 0.2
c = -4*(c_H -c_L)/L**2 *x**2 + 4*(c_H -c_L)/L *x 
cx = -8*(c_H - c_L)/L**2 *x + 4*(c_H -c_L)/L 


# Initial State: Gaussian centered in the middle
sigma = L/20
state = np.exp(-0.5*((x-dx*N/4)/sigma)**2)
state = state / np.linalg.norm(state)

ts = [0.5, 1.0, 1.5, 2.0]

plt.rcParams.update({
    "font.size": 16,
    "figure.figsize": (6, 5), # SMALLER figure = LARGER relative text
    "axes.labelsize": 16,       # Now 14 will look huge and readable
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})
plt.figure()
plt.plot(x, state, 'ko', markersize=4)
plt.plot(x, state, 'k-', label='$t=0$',)
errs = []

for t in ts:
    lchs_state, success_prob = lchs(n_qubits, t, cx, c, D, L, init_state=state, r_steps=100,   useFixedJ=True, fixed_J=128)
    exact = spectral_diff_adv(n_qubits, t, cx, c, D, L, np.zeros_like(c), init_state=state)
    errs.append(np.linalg.norm(np.real(lchs_state) - np.real(exact)))

    line, = plt.plot(x, np.real(lchs_state), 'o', markersize=4)
    color = line.get_color()
    plt.plot(x, exact, '-', color=color, label=f'$t={t:.1f}$', )
plt.xlabel('$x$')
plt.ylabel('$|\phi \\rangle$')
plt.legend(loc='best', fontsize=14)
plt.savefig('./figures/diff_adv.png', bbox_inches='tight')
plt.show()

plt.figure() 
plt.semilogy(ts, errs, 'o-')
plt.xlabel('$t$ ')
plt.ylabel('$|||\\phi\\rangle- |\\phi_{h}\\rangle ||$')
plt.savefig('./figures/diff_adv_error.png', bbox_inches='tight')
plt.show()