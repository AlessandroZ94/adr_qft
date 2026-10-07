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
    
    D_mat = QFT @ P @ QFT_dag
    D2_mat = QFT @ P2 @ QFT_dag
    
    C_mat = np.diag(c)
    C_prime = np.diag(cx)
    R_mat = np.diag(a)
    
    # A = -D * D^2 + 1/2 (C*D + D*C) + R + C'/2
    A_mat = -D * D2_mat + 0.5 * (C_mat @ D_mat + D_mat @ C_mat) + R_mat + 0.5 * C_prime
    
    return expm(-A_mat * t) @ init_state

# ==== SIMULATION EXECUTION ====
n_qubits = 5

D = 0.001
a_am = 0.1
a_mid = 0.2
c_max = 0.4
N = 2**n_qubits
L = 2 * np.pi               
x = np.linspace(0, L, N, endpoint=False)
x_periodic = np.concatenate((x, [L]))
dx = L / N

c_L = 0.0
c_H = c_max
c = -4*(c_H - c_L)/L**2 * x**2 + 4*(c_H - c_L)/L * x 
cx = -8*(c_H - c_L)/L**2 * x + 4*(c_H - c_L)/L 

plt.plot(x, c, 'k-', label='$c(x)$')
plt.plot(x, cx, 'k--', label='$c\'(x)$')
plt.show()
a = (a_am * np.sin(x)+a_mid)
# Initial State: Gaussian
sigma = L/20
mu = L/2
state = np.exp(-0.5*((x-mu)/sigma)**2)
state = state / np.linalg.norm(state)

ts = [0.5, 1.0, 1.5, 2.0]

plt.rcParams.update({""
    "text.usetex": True,
    "font.size": 16,
    "figure.figsize": (5, 4), # SMALLER figure = LARGER relative text
    "axes.labelsize": 16,       # Now 14 will look huge and readable
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

plt.figure()
plt.plot(x_periodic, np.concatenate((state, [state[0]])), 'ko', markersize=4)
plt.plot(x_periodic, np.concatenate((state, [state[0]])), 'k-', label='$t=0$',)
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
        a=a,
        init_state=state,
        fixed_J=True,
    )
    lchs_state = np.concatenate((lchs_state, [lchs_state[0]]))  # Enforce periodicity
    exact = spectral_diff_adv(n_qubits, t, cx, c, D, L, a, init_state=state)
    exact = np.concatenate((exact, [exact[0]]))  # Enforce periodicity
    errs.append(np.linalg.norm(np.real(lchs_state) - np.real(exact)))
    lchs_state = np.real(lchs_state)
    exact = np.real(exact)
    line, = plt.plot(x_periodic, lchs_state/np.linalg.norm(lchs_state), 'o', markersize=4)
    color = line.get_color()
    plt.plot(x_periodic, exact/np.linalg.norm(exact), '-', color=color, label=f'$t={t:.1f}$')
    
plt.xlabel('$x$')
plt.ylabel('$|\phi \\rangle$')
plt.legend(loc='best', fontsize=14)
parameter_suffix = f'D{D:g}_a_am{a_am:g}_c_max{c_max:g}'
plt.savefig(f'./figures/diff_adv_{parameter_suffix}.pdf', bbox_inches='tight')
plt.show()


plt.figure() 
plt.semilogy(ts, errs, 'o-')
plt.xlabel('$t$')
plt.ylabel('$|||\\phi\\rangle- |\\phi_{h}\\rangle ||$')
plt.savefig(f'./figures/diff_adv_error_{parameter_suffix}.pdf', bbox_inches='tight')
plt.show()