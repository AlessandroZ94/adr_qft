import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import DiagonalGate, StatePreparation, UnitaryGate
from qiskit.quantum_info import Statevector
from scipy.linalg import expm, block_diag

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

def lchs(n, t, cx, c, D, L, init_state, useFixedJ=True, fixed_J=128, normalize=True):
    N = 2**n
    
    # --- 1. CONSTRUCT EXACT OPERATORS ---
    j_indices = np.arange(N)
    k_indices = np.where(j_indices < N/2, j_indices, j_indices - N)
    k_vals_spectral = (2 * np.pi / L) * k_indices
    
    P = np.diag(1j * k_vals_spectral)
    P2 = np.diag(-k_vals_spectral**2)

    omega = np.exp(2j * np.pi / N)
    j_mesh, k_mesh = np.meshgrid(np.arange(N), np.arange(N))
    QFT = np.power(omega, j_mesh * k_mesh) / np.sqrt(N)
    QFT_dag = QFT.conj().T
    
    D_mat = QFT_dag @ P @ QFT
    D2_mat = QFT_dag @ P2 @ QFT
    
    C_mat = np.diag(c)
    C_prime = np.diag(cx)
    R_mat = np.zeros_like(C_mat) # a(x) = 0
    
    # Shift to ensure L is positive semi-definite (Eq 18 in text)
    lambda_val = np.min(np.diag(R_mat + 0.5 * C_prime).real)
    if lambda_val < 0:
        shift = lambda_val
        print(f"Applying shift lambda = {shift:.4f} to ensure positive semidefiniteness.")
    else:
        shift = 0.0
        
    # L = (A + A^dag)/2 = -D*D^2 + R + C'/2 - lambda*I
    L_mat = -D * D2_mat + R_mat + 0.5 * C_prime - shift * np.eye(N)
    
    # H = (A - A^dag)/(2i) = -i/2 (C*D + D*C)
    H_mat = -0.5j * (C_mat @ D_mat + D_mat @ C_mat)
    
    # --- 2. LCHS PARAMETERS & INTEGRAL WEIGHTS ---
    c_lchs = 1
    eps_lchs = 1e-6
    eps_quad = 1e-6
    
    norm_L = D * (np.pi * N / L)**2 + np.max(np.abs(cx / 2))
    norm = t * norm_L
    h = np.pi / (norm/2 + np.log(64*np.exp(3*c_lchs/2)/(15*eps_quad)))
    gamma = 1/c_lchs * np.sqrt(c_lchs+np.log((1+1/(2*np.pi))/eps_lchs))
    
    if useFixedJ:
        J_int = fixed_J
    else:
        R_val = 2*c_lchs*gamma**2
        J = R_val/h
        J_int = int(2**(np.floor(np.log2(J))))
        
    num_terms = 2 * J_int
    n_ancilla = int(np.log2(num_terms))
    
    j_vals = np.arange(-J_int, J_int) 
    omega_vals = h * j_vals
    
    # f_hat(omega) including the correct constants (1/sqrt(2pi) * sqrt(2/pi) = 1/pi)
    weights = (h / np.pi) * np.exp(c_lchs*(1-1j*omega_vals)) * np.exp(-(omega_vals**2+1)/(4*gamma**2)) / (1 + omega_vals**2)

    magnitudes = np.abs(weights)
    phases = np.angle(weights) # Evaluates exactly to -omega * c_lchs

    coeffs = np.sqrt(magnitudes)
    coeffs = coeffs / np.linalg.norm(coeffs)         

    # --- 3. CONSTRUCT BLOCK-DIAGONAL SELECT UNITARY ---
    # We do not use Trotter steps because V(w, t) = exp(-i(wL + H)t) 
    # can be block-encoded exactly for the statevector simulation.
    V_blocks = []
    for w in omega_vals:
        exponent = -1j * (w * L_mat + H_mat) * t
        V_blocks.append(expm(exponent))
        
    SEL_matrix = block_diag(*V_blocks)

    # ==== 4. CIRCUIT CONSTRUCTION ====
    reg_s = QuantumRegister(n, 'system')
    reg_a = QuantumRegister(n_ancilla, 'lchs_ancilla') 
    
    # Qiskit Endianness: by defining the circuit with reg_s first, then reg_a,
    # and appending gates to list(reg_s) + list(reg_a), we ensure that reg_a acts 
    # as the Most Significant Bits. This maps perfectly to the block-diagonal structure.
    circuit = QuantumCircuit(reg_s, reg_a)

    # Initialize System State
    circuit.append(StatePreparation(init_state), reg_s)

    # LCU Preparation
    prep_gate = StatePreparation(coeffs)
    circuit.append(prep_gate, reg_a)
    
    # PREP_bar phase injection (Eq 31 in text implies PREP_bar has e^{-ihjC} phases)
    circuit.append(DiagonalGate(np.exp(1j * phases)), reg_a)

    # SELECT Application 
    circuit.append(UnitaryGate(SEL_matrix, label="SEL"), list(reg_s) + list(reg_a))

    # UNPREP (Inverse of PREP without phase, collapsing the LCU)
    circuit.append(prep_gate.inverse(), reg_a)

    # ==== 5. SIMULATION & POST-SELECTION ====
    final_state = Statevector(circuit)
    full_data = np.array(final_state)

    # Because reg_a consists of the MSBs, post-selecting on ancilla |0> 
    # corresponds precisely to the first N elements of the full statevector.
    system_state = full_data[:N]
    
    # Compensate for the shift applied to L (phi(t) = exp(-lambda t) chi(t))
    system_state = system_state * np.exp(-shift * t)

    success_prob = np.linalg.norm(system_state)**2
    print(f"Success Probability at t={t}: {success_prob:.4e}")
    
    if normalize:
        final = system_state / np.linalg.norm(system_state)
    else:
        final = system_state
    
    return final, success_prob

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

t = 1.0
Js = [8,16,32,64,128,256]

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

for J in Js:
    # Note: r_steps is removed as LCHS integrates the full time t natively.
    lchs_state, success_prob = lchs(n_qubits, t, cx, c, D, L, init_state=state, useFixedJ=True, fixed_J=J)
    exact = spectral_diff_adv(n_qubits, t, cx, c, D, L, np.zeros_like(c), init_state=state)
    errs.append(np.linalg.norm(np.real(lchs_state) - np.real(exact)))
    lchs_state = np.real(lchs_state)
    exact = np.real(exact)
    line, = plt.plot(x, lchs_state/np.linalg.norm(lchs_state), 'o', markersize=4)
    color = line.get_color()
    plt.plot(x, exact/np.linalg.norm(exact), '-', color=color, label=f'$J={J:.1f}$')
    
plt.xlabel('$x$')
plt.ylabel('$|\phi \\rangle$')
plt.legend(loc='best', fontsize=14)
plt.savefig('./figures/diff_adv.png', bbox_inches='tight')
plt.show()

plt.figure() 
plt.semilogy(Js, errs, 'o-')
plt.xlabel('$J$')
plt.ylabel('$|||\\phi\\rangle- |\\phi_{h}\\rangle ||$')
plt.savefig('./figures/diff_adv_error.png', bbox_inches='tight')
plt.show()