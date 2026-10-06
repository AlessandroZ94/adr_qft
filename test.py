import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from lchs import lchs

def spectral_diff_adv(n, t, cx, c, D, L, a, init_state):
    """Computes the exact pseudo-spectral solution phi(t) = exp(-At) * phi(0)."""
    N = 2**n
    j_indices = np.arange(N)
    k_indices = np.where(j_indices < N/2, j_indices, j_indices - N)
    k_vals = (2 * np.pi / L) * k_indices
    
    P = np.diag(1j * k_vals)
    P2 = np.diag(-k_vals**2)
    
    # QFT matrix
    omega = np.exp(2j * np.pi / N)
    j_mesh, k_mesh = np.meshgrid(np.arange(N), np.arange(N))
    QFT = np.power(omega, j_mesh * k_mesh) / np.sqrt(N)
    QFT_dag = QFT.conj().T
    
    D_mat = QFT_dag @ P @ QFT
    D2_mat = QFT_dag @ P2 @ QFT
    
    C_mat = np.diag(c)
    C_prime = np.diag(cx)
    R_mat = np.diag(a)
    
    # A = -D * D^2 + 1/2 (C*D + D*C) + R + C'/2
    A_mat = -D * D2_mat + 0.5 * (C_mat @ D_mat + D_mat @ C_mat) + R_mat + 0.5 * C_prime
    
    return expm(-A_mat * t) @ init_state

# ==== SIMULATION EXECUTION ====
n_qubits = 5

D = 0.1
N = 2**n_qubits
L = 2 * np.pi               
x = np.linspace(0, L, N, endpoint=False)
dx = L / N

c_L = 0.0
c_H = 0.2
c = -4*(c_H - c_L)/L**2 * x**2 + 4*(c_H - c_L)/L * x 
cx = -8*(c_H - c_L)/L**2 * x + 4*(c_H - c_L)/L 

# Initial State: Gaussian centered in the middle
sigma = L/10
state = np.exp(-0.5*((x - dx*N/4)/sigma)**2)
state = state / np.linalg.norm(state)

ts = [0.5, 1.0, 1.5, 2.0]

plt.rcParams.update({
    "font.size": 16,
    "figure.figsize": (6, 5),
    "axes.labelsize": 16,     
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

plt.figure()
plt.plot(x, state, 'ko', markersize=4)
plt.plot(x, state, 'k-', label='$t=0$',)
errs = []

for t in ts:
    # Note: r_steps is removed as LCHS integrates the full time t natively.
    lchs_state, success_prob = lchs(
        n_qubits,
        t,
        cx,
        c,
        D,
        L,
        np.zeros_like(c),
        init_state=state,
        fixed_J=True,
    )
    exact = spectral_diff_adv(n_qubits, t, cx, c, D, L, np.zeros_like(c), init_state=state)
    errs.append(np.linalg.norm(np.real(lchs_state) - np.real(exact),np.inf))
    lchs_state = np.real(lchs_state)
    exact = np.real(exact)
    line, = plt.plot(x, lchs_state/np.linalg.norm(lchs_state), 'o', markersize=4)
    color = line.get_color()
    plt.plot(x, exact/np.linalg.norm(exact), '-', color=color, label=f'$t={t:.1f}$')
    
plt.xlabel('$x$')
plt.ylabel('$|\phi \\rangle$')
plt.legend(loc='best', fontsize=14)
plt.savefig('./figures/diff_adv.png', bbox_inches='tight')
plt.show()


plt.figure() 
plt.semilogy(ts, errs, 'o-')
plt.xlabel('$t$')
plt.ylabel('$|||\\phi\\rangle- |\\phi_{h}\\rangle ||$')
plt.savefig('./figures/diff_adv_error.png', bbox_inches='tight')
plt.show()