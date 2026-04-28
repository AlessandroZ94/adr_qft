import numpy as np
from numpy.random import beta
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation, DiagonalGate, UnitaryGate
from qiskit.quantum_info import Statevector
from scipy.linalg import expm
from scipy import sparse
from scipy.sparse.linalg import spsolve

# subroutine for QFT matrix (and its inverse) on n qubits
def get_qft_mat(n):
    num_states = 1<<n # this is 2**n
    omega = np.exp(-2j*np.pi/num_states)

    qft = np.zeros((num_states,num_states),dtype=complex)

    for i in range(num_states):
        for j in range(num_states):
            qft[i][j] = omega**(i*j)/np.sqrt(num_states)

    qft_inv = qft.transpose().conjugate()

    return qft, qft_inv

def lchs(n, t, cx, c, D, L, init_state, r_steps=10, useFixedJ=False, fixed_J=64):
    N = 2**n
    final_time = t
    eps_lchs = 1e-3
    eps_quad = 1e-3
    c_lchs = 1

    H_herm_raw = cx / 2
    min_eigenvalue = np.min(H_herm_raw)
    
    # 1. Apply the shift ONLY if there are negative eigenvalues
    if min_eigenvalue < 0:
        lambda_shift = np.abs(min_eigenvalue)
        print(f"Applying shift of {lambda_shift:.4f} to ensure positive semidefiniteness.")
    else:
        lambda_shift = 0
        
    j_indices = np.arange(N)
    k_indices = np.where(j_indices <= N/2, j_indices, j_indices - N)
    P_1 = 1j * (2 * np.pi / L) * (k_indices)
    P_2 = -1 * (2 * np.pi / L)**2 * (k_indices**2) 

    qft, qft_inv = get_qft_mat(n)
    Delta = qft_inv @ np.diag(P_1) @ qft
    
    # 2. SEPARATE HERMITIAN TERMS INTO 1D EIGENVALUE ARRAYS
    H_herm_phys = H_herm_raw + lambda_shift  # Diagonal in physical space
    H_herm_mom = D * P_2                     # Diagonal in momentum space 

    norm_L = D * (np.pi * N / L)**2 + np.max(np.abs(cx / 2))
    norm = t * norm_L
    h = np.pi / (norm/2 + np.log(64*np.exp(3*c_lchs/2)/(15*eps_quad)))
    gamma = 1/c_lchs * np.sqrt(c_lchs+np.log((1+1/(2*np.pi))/eps_lchs))
    
    if useFixedJ:
        J_int = fixed_J
    else:
        R = 2*c_lchs*gamma**2
        J = R/h
        J_int = int(2**(np.floor(np.log2(J))))
        
    num_terms = 2*J_int
    n_ancilla = int(np.log2(num_terms))

    # Creates exactly num_terms points (-J_int to J_int-1)
    j_vals = np.arange(-J_int, J_int) 
    k_vals = h * j_vals
    weights = (h / np.sqrt(2*np.pi)) * np.exp(c_lchs*(1-1j*k_vals)) * np.exp(-(k_vals**2+1)/(4*gamma**2)) / (1 + k_vals**2)

    magnitudes = np.abs(weights)
    phases = np.angle(weights)

    coeffs = np.sqrt(magnitudes)
    coeffs = coeffs / np.linalg.norm(coeffs)         

    dt = final_time / r_steps
    sqrt_dt = np.sqrt(dt)

    # 3. SETUP TWO CONTROLLED-DIAGONAL GATES
    select_diag_phys = np.ones(2**(n_ancilla + n), dtype=complex)
    select_diag_mom = np.ones(2**(n_ancilla + n), dtype=complex)

    for i in range(num_terms):
        # Physical space advection/reaction 
        U_phys = np.exp(-1j * dt * H_herm_phys * k_vals[i])
        
        # Momentum space diffusion
        U_mom = np.exp(1j * dt * H_herm_mom * k_vals[i]) 
        
        for sys_idx in range(N):
            idx = (sys_idx << n_ancilla) + i
            select_diag_phys[idx] = U_phys[sys_idx]
            select_diag_mom[idx] = U_mom[sys_idx]

    gate_phys = DiagonalGate(select_diag_phys)
    gate_mom = DiagonalGate(select_diag_mom)

    # Isolate the LCU phase into a single ancilla gate outside the loop
    ancilla_phase_gate = DiagonalGate(np.exp(1j * phases))

    # --- 4. SETUP TROTTER STEP FOR SKEW-HERMITIAN PART (U_step) ---
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    A = -np.diag(c) / 2
    B = Delta  
    A_tilde = np.kron(A, Y)
    B_tilde = np.kron(B, X)

    # Compute a SINGLE Trotter step for the skew-Hermitian commutator
    step_1 = expm(-1j * sqrt_dt * A_tilde)
    step_2 = expm(sqrt_dt * B_tilde)
    step_3 = expm(1j * sqrt_dt * A_tilde)
    step_4 = expm(-sqrt_dt * B_tilde)
    U_step = step_4 @ step_3 @ step_2 @ step_1 

    # ==== 5. CIRCUIT CONSTRUCTION ====
    reg_a = QuantumRegister(n_ancilla, 'lchs_ancilla') 
    reg_ca = QuantumRegister(1, 'commutator_ancilla') 
    reg_s = QuantumRegister(n, 'system')
    circuit = QuantumCircuit(reg_a, reg_ca, reg_s)

    # Initialize System State
    stateprep = StatePreparation(init_state)
    circuit.append(stateprep, reg_s)

    # --- A. PREP LCU SUPERPOSITION ---
    prep_gate = StatePreparation(coeffs)
    circuit.append(prep_gate, reg_a)
    
    # Apply the LCU complex phases exactly ONCE before the loop
    circuit.append(ancilla_phase_gate, reg_a)

    qft_gate = UnitaryGate(qft, label="QFT")
    iqft_gate = UnitaryGate(qft_inv, label="IQFT")

# --- B. THE TROTTER LOOP ---
    for _ in range(r_steps):
        # 1. Apply w-independent Advection Commutator Step (First mathematically)
        circuit.append(UnitaryGate(U_step), [reg_ca[0]] + list(reg_s))

    
        # 2. Apply w-dependent Advection/Reaction (Second)
        circuit.append(gate_phys, list(reg_a) + list(reg_s))

        # 3. Apply w-dependent Diffusion (Third)
        circuit.append(qft_gate, reg_s)
        circuit.append(gate_mom, list(reg_a) + list(reg_s))
        circuit.append(iqft_gate, reg_s)

    # --- C. UNPREP LCU SUPERPOSITION ---
    circuit.append(prep_gate.inverse(), reg_a)

    # ==== 6. SIMULATION & POST-SELECTION ====
    final_state = Statevector(circuit)
    full_data = np.array(final_state)

    # Qiskit ordering: reg_s (MSB) | reg_ca | reg_a (LSB)
    system_state = full_data[0::2**(n_ancilla + 1)]

    success_prob = np.linalg.norm(system_state)**2
    print(f"Success Probability: {success_prob:.4e}")
    

    final = (system_state) / np.linalg.norm(system_state)
    
    return final, success_prob


def spectral_diff_adv(n, t, cx, c, D, L, a, init_state):
    N= 2**n
    j_indices = np.arange(N)
    k_j = 2*np.pi /L * np.where(j_indices < N/2, j_indices, j_indices - N)
    P1 = np.diag(1j*k_j)
    P2 = np.diag(-k_j**2)
    QFT, QFT_inv = get_qft_mat(n)
    D1 = QFT_inv @ P1 @ QFT
    D2 = QFT_inv @ P2 @ QFT

    A = - (D*D2-0.5*(np.diag(c) @ D1 + D1 @ np.diag(c)) - 0.5*np.diag(cx)-np.diag(a))
    
    final_state = expm(-A*t) @ init_state
    final_state = final_state / np.linalg.norm(final_state)
    return final_state

def fd_diff_adv(n, t, cx, c, D, L, a, init_state):
    N= 2**n
    dx = L / N
    dt = t / 100000
    N_steps = int(t / dt)

    gamma = dt /  dx
    beta = D * dt / dx**2
    diag_main = (1.0 - 2*beta) * np.ones(N)

    c_l = np.roll(c, 1)
    diag_lower = beta + gamma * c_l

    c_u = np.roll(c, -1)
    diag_upper = beta - gamma * c_u

    A = sparse.diags([diag_main, diag_lower, diag_upper, diag_lower[0], diag_upper[N-1]], [0, -1, 1, -N+1, N-1])

    state = init_state
    for i in range(N_steps):
        state = A @ state

    final_state = state / np.linalg.norm(state)
    return final_state

def be_fd_diff_adv(n, t, cx, c, D, L, a, init_state):
    N= 2**n
    dx = L / N
    dt = t / 100000
    N_steps = int(t / dt)

    gamma = dt /  dx
    beta = D * dt / dx**2
    diag_main = (1.0 + 2*beta) * np.ones(N)

    c_l = np.roll(c, 1)
    diag_lower = -beta - gamma * c_l

    c_u = np.roll(c, -1)
    diag_upper = -beta + gamma * c_u

    A = sparse.diags([diag_main, diag_lower, diag_upper, diag_lower[0], diag_upper[N-1]], [0, -1, 1, -N+1, N-1])
    A = A.tocsr()
    state = init_state
    for i in range(N_steps):
        state = spsolve(A, state)

    final_state = state / np.linalg.norm(state)
    return final_state