import numpy as np
from numpy.random import beta
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation, DiagonalGate, UnitaryGate
from qiskit.quantum_info import Statevector
from scipy.linalg import block_diag, expm
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

def lchs(
    n,
    t,
    cx,
    c,
    D,
    L,
    a=None,
    init_state=None,
    eps=1e-3,
    normalize=True,
    fixed_J=False,
    J=128,
    *,
    r_steps=None,
    useFixedJ=None,
):
    """Apply LCHS with exact SELECT blocks; legacy Trotter arguments remain accepted."""
    N = 2**n
    if init_state is None:
        if a is None:
            raise ValueError("init_state is required.")
        init_state = a
        a = None
    if a is None:
        a = np.zeros_like(c)

    if useFixedJ is not None:
        fixed_j_requested = useFixedJ
        if useFixedJ and isinstance(fixed_J, (int, np.integer)) and not isinstance(fixed_J, (bool, np.bool_)):
            J = fixed_J
    else:
        fixed_j_requested = fixed_J
        if isinstance(fixed_J, (int, np.integer)) and not isinstance(fixed_J, (bool, np.bool_)):
            J = fixed_J

    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a finite positive number.")

    c_lchs = 1

    j_indices = np.arange(N)
    k_indices = np.where(j_indices < N / 2, j_indices, j_indices - N)
    k_vals_spectral = (2 * np.pi / L) * k_indices

    P = np.diag(1j * k_vals_spectral)
    P2 = np.diag(-k_vals_spectral**2)

    omega = np.exp(2j * np.pi / N)
    j_mesh, k_mesh = np.meshgrid(np.arange(N), np.arange(N))
    qft = np.power(omega, j_mesh * k_mesh) / np.sqrt(N)
    qft_inv = qft.conj().T

    D_mat = qft @ P @ qft_inv
    D2_mat = qft @ P2 @ qft_inv

    C_mat = np.diag(c)
    C_prime = np.diag(cx)
    R_mat = np.diag(a)

    lambda_val = np.min(np.diag(R_mat + 0.5 * C_prime).real)
    if lambda_val < 0:
        shift = lambda_val
        print(f"Applying shift lambda = {shift:.4f} to ensure positive semidefiniteness.")
    else:
        shift = 0.0

    L_mat = -D * D2_mat + R_mat + 0.5 * C_prime - shift * np.eye(N)
    H_mat = -0.5j * (C_mat @ D_mat + D_mat @ C_mat)

    norm_L = D * (np.pi * N / L) ** 2 + np.max(np.abs(cx / 2))
    norm = t * norm_L
    const = (3 + 3 / (2 * np.pi)) / eps
    gamma = np.sqrt(1 + np.log(const))
    radius = 2 * (1 + np.log(const))
    k = int(
        np.ceil(
            np.log2(
                2
                / np.pi
                * (1 + np.log(const))
                * (norm / 2 + 1.5 + np.log(64 / (5 * eps)))
                + 1
            )
        )
    )
    h = radius / (2**k - 1)
    J_int = radius / h
    J_eff = 2 ** np.ceil(np.log2(J_int))

    if fixed_j_requested:
        J_eff = J
    J_eff = int(J_eff)
    if J_eff <= 0 or J_eff & (J_eff - 1):
        raise ValueError("The fixed LCHS term count J must be a positive power of two.")

    n_ancilla = int(np.log2(2 * J_eff))
    j_vals = np.arange(-J_eff, J_eff)
    omega_vals = h * j_vals

    weights = (
        (h / np.pi)
        * np.exp(c_lchs * (1 - 1j * omega_vals))
        * np.exp(-(omega_vals**2 + 1) / (4 * gamma**2))
        / (1 + omega_vals**2)
    )
    magnitudes = np.abs(weights)
    phases = np.angle(weights)

    coeffs = np.sqrt(magnitudes)
    coeffs = coeffs / np.linalg.norm(coeffs)

    V_blocks = []
    for w in omega_vals:
        exponent = -1j * (w * L_mat + H_mat) * t
        V_blocks.append(expm(exponent))

    SEL_matrix = block_diag(*V_blocks)

    reg_s = QuantumRegister(n, "system")
    reg_a = QuantumRegister(n_ancilla, "lchs_ancilla")
    circuit = QuantumCircuit(reg_s, reg_a)
    circuit.append(StatePreparation(init_state), reg_s)

    prep_gate = StatePreparation(coeffs)
    circuit.append(prep_gate, reg_a)
    circuit.append(DiagonalGate(np.exp(1j * phases)), reg_a)
    circuit.append(UnitaryGate(SEL_matrix, label="SEL"), list(reg_s) + list(reg_a))
    circuit.append(prep_gate.inverse(), reg_a)

    final_state = Statevector(circuit)
    full_data = np.array(final_state)
    system_state = full_data[:N]
    system_state = system_state * np.exp(-shift * t)

    success_prob = np.linalg.norm(system_state) ** 2
    print(f"Success Probability at t={t}: {success_prob:.4e}")

    if normalize:
        final = system_state / np.linalg.norm(system_state)
    else:
        final = system_state

    return final, success_prob

def spectral_diff_adv_op(n, t, cx, c, D, L, a, shift=0):
    N= 2**n
    j_indices = np.arange(N)
    k_j = 2*np.pi /L * np.where(j_indices < N/2, j_indices, j_indices - N)
    P1 = np.diag(1j*k_j)
    P2 = np.diag(-k_j**2)
    QFT, QFT_inv = get_qft_mat(n)
    D1 = QFT_inv @ P1 @ QFT
    D2 = QFT_inv @ P2 @ QFT

    A = - (D*D2-0.5*(np.diag(c) @ D1 + D1 @ np.diag(c)) - 0.5*np.diag(cx)-np.diag(a)) + shift * np.eye(N)
    U = expm(-A*t)
    return U, A

def spectral_diff_adv(n, t, cx, c, D, L, a, init_state, normalize=True):
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
    if normalize:
        final_state = final_state / np.linalg.norm(final_state)
    return final_state

def fe_spectral_diff_adv(n, t, cx, c, D, L, a, init_state, normalize=True):
    N= 2**n
    j_indices = np.arange(N)
    k_j = 2*np.pi /L * np.where(j_indices < N/2, j_indices, j_indices - N)
    P1 = np.diag(1j*k_j)
    P2 = np.diag(-k_j**2)
    QFT, QFT_inv = get_qft_mat(n)
    D1 = QFT_inv @ P1 @ QFT
    D2 = QFT_inv @ P2 @ QFT

    A = - (D*D2-0.5*(np.diag(c) @ D1 + D1 @ np.diag(c)) - 0.5*np.diag(cx)-np.diag(a))

    N_steps = 10000
    dt = t / N_steps
    M = np.eye(N) - A * (dt)

    state = init_state
    for _ in range(N_steps):
        state = M @ state
    final_state = state
    if normalize:
        final_state = final_state / np.linalg.norm(final_state)
    return final_state

def fe_fd_diff_adv(n, t, cx, c, D, L, a, init_state, normalize=True):
    N = 2**n
    dx = L / N
    dt = t / 100000
    N_steps = int(t / dt)

    gamma = dt / dx
    beta = D * dt / dx**2
    
    diag_main = (1.0 - 2*beta) * np.ones(N)

    diag_lower = beta + (gamma * c / 2.0)
    diag_upper = beta - (gamma * c / 2.0)

    A = sparse.diags([diag_main, diag_lower, diag_upper, diag_lower, diag_upper], 
                     [0, -1, 1, -N+1, N-1])

    state = init_state
    for _ in range(N_steps):
        state = A @ state
    final_state = state
    if normalize: 
        final_state = state / np.linalg.norm(state)
    return final_state

def be_fd_diff_adv(n, t, cx, c, D, L, a, init_state, normalize=True):
    N = 2**n
    dx = L / N
    dt = t / 100000
    N_steps = int(t / dt)

    gamma = dt / dx
    beta = D * dt / dx**2
    
    diag_main = (1.0 + 2*beta) * np.ones(N)

    # Note the flipped signs for the implicit LHS matrix
    diag_lower = -beta - (gamma * c / 2.0)
    diag_upper = -beta + (gamma * c / 2.0)

    A = sparse.diags([diag_main, diag_lower, diag_upper, diag_lower, diag_upper], 
                     [0, -1, 1, -N+1, N-1])
    A = A.tocsr()
    
    state = init_state
    for _ in range(N_steps):
        state = spsolve(A, state)
    final_state = state
    if normalize: 
        final_state = state / np.linalg.norm(state)
    return final_state