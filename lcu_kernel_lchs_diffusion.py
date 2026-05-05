
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import DiagonalGate, QFTGate, StatePreparation
from qiskit.quantum_info import Statevector
from scipy.linalg import expm
from lchs import get_qft_mat, lchs   


def compute_diffusion_final_state(t, D, n_qubits, L, state, usefixedJ=False, fixedJ=32):
    # Parameters
    N = 2**n_qubits             
    final_time = t
    eps_lchs = 1e-3
    eps_quad = 1e-3
    c = 1

    # Hamiltonian
    j_indices = np.arange(N)
    k_indices = np.where(j_indices <= N/2, j_indices, j_indices - N)
    Hamiltonian_diag = D * (2 * np.pi / L)**2 * (k_indices**2)

    norm = t * np.linalg.norm(Hamiltonian_diag, 2) # Use infinity norm for spectral bound?
    h = np.pi / (norm/2 + np.log(64*np.exp(3*c/2)/(15*eps_quad)))
    gamma = 1/c * np.sqrt(c+np.log((1+1/(2*np.pi))/eps_lchs))
    # LCHS Parameters 
    if usefixedJ:
        num_terms = fixedJ
        J = fixedJ/2
        R = J * h
    else:
        R = 2*c*gamma**2
        J = R/h
        print(f"J :{J}")

        J_int = int(2**(np.floor(np.log2(J)))) # closest power of 2 <= J
        print(f"J_int :{J_int}")

        R = J_int * h

        print(f"R :{R}, h :{h}, gamma :{gamma}")
        J = J_int
        # LCU terms
        num_terms = 2*J_int

    n_ancilla = np.log2(num_terms).astype(int)

    # Compute Complex Weights
    j_vals = np.arange(-J, J) 
    k = h * j_vals

    weights = (h / np.sqrt(2*np.pi)) * np.exp(c*(1-1j*k)) * np.exp(-(k**2+1)/(4*gamma**2)) / (1 + k**2)

    # 2. SEPARATE Magnitude and Phase 
    magnitudes = np.abs(weights)
    phases = np.angle(weights)

    # 3. Prepare Amplitudes 
    coeffs = np.sqrt(magnitudes)
    coeffs = coeffs / np.linalg.norm(coeffs)         

    # Build Circuit
    reg_a = QuantumRegister(n_ancilla, 'ancilla') 
    reg_s = QuantumRegister(n_qubits, 'system')
    lc_qc = QuantumCircuit(reg_a, reg_s)

    # PREP (Encodes magnitudes only)
    prep_gate = StatePreparation(coeffs)
    lc_qc.append(prep_gate, reg_a)


    # --- FAST SELECT OPTIMIZATION ---
    # Construct the massive diagonal for the SELECT operator directly.
    select_diagonal = np.ones(2**(n_ancilla + n_qubits), dtype=complex)
    
    for i in range(num_terms):
        U_diag = np.exp(1j * phases[i]) * np.exp(-1j * final_time * Hamiltonian_diag * k[i])
        
        for sys_idx in range(N):
            # Handle Qiskit's little-endian ordering: 
            # System register is MSB, Ancilla register is LSB
            idx = (sys_idx << n_ancilla) + i
            select_diagonal[idx] = U_diag[sys_idx]

    # Apply as a single O(1) diagonal gate
    lc_qc.append(DiagonalGate(select_diagonal), list(reg_a) + list(reg_s))
    # --------------------------------


    # # SELECT (Encodes Unitary AND Phase)
    # for i in range(num_terms):
    #     # U(t; k) = exp(-i * k * H * t)
    #     # We multiply by the weight phase here: e^{i arg(w)} * U
    #     U_diag = np.exp(1j * phases[i]) * np.exp(-1j * final_time * Hamiltonian_diag * k[i])
        
    #     # Qiskit endianness: format(i,...) puts LSB on right. 
    #     # DiagonalGate control matches this convention if applied correctly.
    #     ctrl_str = format(i, f'0{n_ancilla}b') 
    #     gate = DiagonalGate(U_diag).control(n_ancilla, ctrl_state=ctrl_str)
    #     lc_qc.append(gate, list(reg_a) + list(reg_s))

    # PREP Dagger (Uncompute)
    lc_qc.append(prep_gate.inverse(), reg_a)

    # Full Simulation Circuit
    circuit = QuantumCircuit(n_ancilla + n_qubits)

    # Initialize System (Gaussian)

    stateprep = StatePreparation(state)
    circuit.append(stateprep, range(n_ancilla, n_ancilla + n_qubits))

    # QFT -> LCU (Momentum Space) -> IQFT
    circuit.append(QFTGate(n_qubits), range(n_ancilla, n_ancilla + n_qubits))
    circuit.compose(lc_qc, inplace=True)
    circuit.append(QFTGate(n_qubits).inverse(), range(n_ancilla, n_ancilla + n_qubits))

    # Simulation
    final_state = Statevector(circuit)
    full_data = np.array(final_state)

    # Post-selection
    system_state = full_data[0::2**n_ancilla]

    # Normalize & Compare
    success_prob = np.linalg.norm(system_state)**2
    final = system_state / np.linalg.norm(system_state)
    return final


D = 0.2
n_qubits = 5
N = 2**n_qubits
L = 2*np.pi 
x = np.linspace(0, L, N, endpoint=False)
dx = L / N
ts = [0.5,1, 1.5,2.0]

# Initial State: Gaussian centered in the middle
sigma = L/20
state = np.exp(-0.5*((x-dx*N/2)/sigma)**2)
state = state / np.linalg.norm(state)



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

    final = compute_diffusion_final_state(t=t, D=D, n_qubits=n_qubits, L=L, state=state, usefixedJ=True, fixedJ=512)


    # Exact Solution 
    QFT, QFT_inv = get_qft_mat(n_qubits)
    j_indices = np.arange(N)
    k_j = 2*np.pi /L * np.where(j_indices < N/2, j_indices, j_indices - N)
    P1 = np.diag(1j*k_j)
    P2 = np.diag(-k_j**2)
    D1 = QFT_inv @ P1 @ QFT
    D2 = QFT_inv @ P2 @ QFT

    A = - (D*D2)

    exact_final = expm(-A * t) @ state
    exact_final = exact_final / np.linalg.norm(exact_final)

    # Error Calculation
    err = np.linalg.norm(np.real(exact_final) - np.real(final))
    errs.append(err)
    print(f"Error: {err:.4e}")

    line, = plt.plot(x, np.real(final), 'o',  markersize=4)
    color = line.get_color()
    plt.plot(x, np.real(exact_final), '-',  label=f'$t={t:.1f}$', color=color)
plt.xlabel('$x$')
plt.ylabel('$|\phi \\rangle$')
plt.legend(loc='best', fontsize=14)
plt.savefig('./figures/diffusion.png', bbox_inches='tight')
plt.show()

plt.figure() 
plt.semilogy(ts, errs, 'o-')
plt.xlabel('$t$ ')
plt.ylabel('$|||\\phi\\rangle- |\\phi_{h}\\rangle ||$')
plt.savefig('./figures/diffusion_error.png', bbox_inches='tight')
plt.show()