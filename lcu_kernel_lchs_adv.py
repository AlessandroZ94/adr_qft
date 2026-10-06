import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import DiagonalGate, QFTGate, StatePreparation, UnitaryGate
from qiskit.quantum_info import Statevector
from scipy.linalg import expm
from lchs import get_qft_mat, lchs

n_qubits = 5
N = 2**n_qubits
L = 2 * np.pi               
t = 5     
x = np.linspace(0, L, N, endpoint=False)
dx = L / N

# Parameters
c_L = 0.0
c_H = 0.2
c = -4*(c_H -c_L)/L**2 *x**2 + 4*(c_H -c_L)/L *x 
cx = -8*(c_H - c_L)/L**2 *x + 4*(c_H -c_L)/L 



sigma = L/20
state = np.exp(-0.5*((x-dx*N/4)/sigma)**2)
state = state / np.linalg.norm(state)





ts = [1, 2, 3]



plt.rcParams.update({
    "font.size": 16,
    "figure.figsize": (6, 5), # SMALLER figure = LARGER relative text
    "axes.labelsize": 16,       # Now 14 will look huge and readable
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})
plt.figure()
plt.plot(x, state, 'ko',  markersize=4)
plt.plot(x, state, 'k-', label='$t=0$')

errs = []
for t in ts:

    final, _ = lchs(n_qubits, t, cx, c, 0, L, state, fixed_J=True, J=128, normalize=True)


    # Exact Solution 
    QFT, QFT_inv = get_qft_mat(n_qubits)
    j_indices = np.arange(N)
    k_j = 2*np.pi /L * np.where(j_indices < N/2, j_indices, j_indices - N)
    P1 = np.diag(1j*k_j)
    P2 = np.diag(-k_j**2)
    D1 = QFT_inv @ P1 @ QFT
    D2 = QFT_inv @ P2 @ QFT

    A = - (-0.5 * (D1 @ np.diag(c) + np.diag(c) @ D1) - 0.5 * np.diag(cx))

    exact_final = expm(-A * t) @ state
    exact_final = exact_final / np.linalg.norm(exact_final)

    # Error Calculation
    err = np.linalg.norm(np.real(exact_final) - np.real(final))
    errs.append(err)
    print(f"Error: {err:.4e}")

    line, = plt.plot(x, np.real(final), 'o',  markersize=4)
    color = line.get_color()
    plt.plot(x, np.real(exact_final), '-',  label=f'$t={t:.1f}$',color=color, )
plt.xlabel('$x$')
plt.ylabel('$|\phi \\rangle$')
plt.legend(loc='upper center', ncol=3, fontsize=14)
plt.legend()
plt.savefig('./figures/par_advection.png', bbox_inches='tight')
plt.show()


plt.rcParams.update({
    "font.size": 16,
    "figure.figsize": (6, 5), # SMALLER figure = LARGER relative text
    "axes.labelsize": 16,       # Now 14 will look huge and readable
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})
plt.figure() 
plt.semilogy(ts, errs, 'o-')
plt.xlabel('$t$ ')
plt.ylabel('$|||\\phi\\rangle- |\\phi_{h}\\rangle ||$')
plt.savefig('./figures/par_advection_error.png', bbox_inches='tight')
plt.show()
