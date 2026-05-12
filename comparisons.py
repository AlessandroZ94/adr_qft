import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import DiagonalGate, QFTGate, StatePreparation, UnitaryGate
from qiskit.quantum_info import Statevector
from scipy.linalg import expm
from lchs import spectral_diff_adv, fe_fd_diff_adv, be_fd_diff_adv, fe_spectral_diff_adv

ns = [3, 4, 5, 6, 7]
errs = []
plt.rcParams.update({
    "font.size": 16,
    "figure.figsize": (6, 5), # SMALLER figure = LARGER relative text
    "axes.labelsize": 16,       # Now 14 will look huge and readable
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})
plt.figure()
for n_qubits in ns:
    print(f'Running for n={n_qubits} qubits...')
    D = 0.1
    N = 2**n_qubits
    L = 2 * np.pi               
    t = 0.5   
    x = np.linspace(0, L, N, endpoint=False)
    dx = L / N
    a = 0.01

    # Parameters
    c_L = 0.0
    c_H = 0.2
    c = c_H*np.ones_like(x)
    cx = np.zeros_like(x)

    # Initial State: Gaussian centered in the middle

    state = np.zeros_like(x)
    x_0 = dx * N/2
    state[N//2] = 1.0
    state = state / dx
    #state = state / np.linalg.norm(state)



    spectral = spectral_diff_adv(n_qubits, t, cx, c, D, L, a*np.ones_like(c), init_state=state, normalize=False)
    #fd = fe_fd_diff_adv(n_qubits, t, cx, c, D, L, np.zeros_like(c), init_state=state)
    exact = + 1/(L)*np.exp(-a*t) + np.zeros_like(x)
    for k in range(1,1000000):
        exact = exact +np.exp(-D*(2*np.pi*k/L)**2*t-a*t)*2/L*np.cos(2*k*np.pi/L*(x-(x_0+c_H*t)))
    #exact = np.exp(-(x-(x_0+c_H*t))**2/(4*D*t)-a*t)
    #exact = exact / np.linalg.norm(exact)
    
    if N == 8 or N == 32 or N==128:
        line, = plt.plot(x, np.real(spectral), 'o--', markersize=4, label=f'$N={N:.0f}$')

    #color = line.get_color()
    #plt.plot(x, fd, 'x', color=color, markersize=4)
    #plt.plot(x, exact, '-', color=color, label=f'$N={N:.0f}$', )
    errs.append(np.linalg.norm(np.real(spectral) - exact,np.inf))

plt.plot(x, exact, 'k-', label="$\\psi$")
plt.xlabel('$x$')
plt.ylabel('$\phi$')
plt.legend(loc='best', fontsize=14)
plt.title(f'$t={t:.1f}$', fontsize=14)
plt.savefig('./figures/exact.png', bbox_inches='tight')
plt.show()


plt.figure()
plt.semilogy(2**np.array(ns), errs, 'o-')
plt.xlabel('$N$ ')
plt.ylabel('$||\\psi- \\phi ||_{\\infty}$')
plt.title(f'$t={t:.1f}$', fontsize=14)
plt.savefig('./figures/exact_error.png', bbox_inches='tight')
plt.show()


