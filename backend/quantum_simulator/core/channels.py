"""
Quantum Noise Channels - Kraus operator implementations.
Models various decoherence and error processes.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from .gates import I, X, Y, Z


@dataclass
class NoiseChannel:
    """Represents a quantum noise channel with Kraus operators."""
    name: str
    kraus_ops: List[np.ndarray]
    params: Dict[str, float]

    def apply(self, rho: np.ndarray) -> np.ndarray:
        """Apply channel to single-qubit density matrix."""
        result = np.zeros_like(rho)
        for K in self.kraus_ops:
            result += K @ rho @ K.conj().T
        return result


def validate_kraus_ops(kraus_ops: List[np.ndarray], tol: float = 1e-10) -> bool:
    """
    Validate that Kraus operators satisfy completeness: Σᵢ Kᵢ†Kᵢ = I.

    Args:
        kraus_ops: List of Kraus operators
        tol: Tolerance for numerical comparison

    Returns:
        True if valid, False otherwise
    """
    dim = kraus_ops[0].shape[0]
    total = np.zeros((dim, dim), dtype=complex)

    for K in kraus_ops:
        total += K.conj().T @ K

    return np.allclose(total, np.eye(dim), atol=tol)


# =============================================================================
# Single-Qubit Noise Channels
# =============================================================================

def depolarizing_channel(p: float) -> List[np.ndarray]:
    """
    Depolarizing channel: ρ → (1-p)ρ + p/3(XρX + YρY + ZρZ).

    With probability p, applies a random Pauli error.
    Maximally mixed output at p = 3/4.

    Args:
        p: Depolarizing probability (0 to 1)

    Returns:
        List of 4 Kraus operators [K0, K1, K2, K3]
    """
    if not 0 <= p <= 1:
        raise ValueError("Probability p must be in [0, 1]")

    K0 = np.sqrt(1 - p) * I
    K1 = np.sqrt(p / 3) * X
    K2 = np.sqrt(p / 3) * Y
    K3 = np.sqrt(p / 3) * Z

    return [K0, K1, K2, K3]


def amplitude_damping(gamma: float) -> List[np.ndarray]:
    """
    Amplitude damping channel (energy relaxation, T1 decay).
    Models spontaneous emission from |1⟩ to |0⟩.

    ρ → K0 ρ K0† + K1 ρ K1†
    K0 = [[1, 0], [0, √(1-γ)]]
    K1 = [[0, √γ], [0, 0]]

    Args:
        gamma: Damping parameter (probability of decay)

    Returns:
        List of 2 Kraus operators
    """
    if not 0 <= gamma <= 1:
        raise ValueError("Gamma must be in [0, 1]")

    K0 = np.array([
        [1, 0],
        [0, np.sqrt(1 - gamma)]
    ], dtype=complex)

    K1 = np.array([
        [0, np.sqrt(gamma)],
        [0, 0]
    ], dtype=complex)

    return [K0, K1]


def generalized_amplitude_damping(p: float, gamma: float) -> List[np.ndarray]:
    """
    Generalized amplitude damping (finite temperature).
    Includes both decay and excitation processes.

    Args:
        p: Population of excited state in thermal equilibrium
        gamma: Damping rate

    Returns:
        List of 4 Kraus operators
    """
    if not 0 <= p <= 1 or not 0 <= gamma <= 1:
        raise ValueError("Parameters must be in [0, 1]")

    K0 = np.sqrt(p) * np.array([
        [1, 0],
        [0, np.sqrt(1 - gamma)]
    ], dtype=complex)

    K1 = np.sqrt(p) * np.array([
        [0, np.sqrt(gamma)],
        [0, 0]
    ], dtype=complex)

    K2 = np.sqrt(1 - p) * np.array([
        [np.sqrt(1 - gamma), 0],
        [0, 1]
    ], dtype=complex)

    K3 = np.sqrt(1 - p) * np.array([
        [0, 0],
        [np.sqrt(gamma), 0]
    ], dtype=complex)

    return [K0, K1, K2, K3]


def phase_damping(gamma: float) -> List[np.ndarray]:
    """
    Phase damping channel (T2 dephasing without energy loss).
    Destroys off-diagonal elements without population change.

    Args:
        gamma: Dephasing parameter

    Returns:
        List of 2 Kraus operators
    """
    if not 0 <= gamma <= 1:
        raise ValueError("Gamma must be in [0, 1]")

    K0 = np.array([
        [1, 0],
        [0, np.sqrt(1 - gamma)]
    ], dtype=complex)

    K1 = np.array([
        [0, 0],
        [0, np.sqrt(gamma)]
    ], dtype=complex)

    return [K0, K1]


def phase_flip(p: float) -> List[np.ndarray]:
    """
    Phase flip channel: ρ → (1-p)ρ + pZρZ.

    Args:
        p: Probability of phase flip

    Returns:
        List of 2 Kraus operators
    """
    if not 0 <= p <= 1:
        raise ValueError("Probability must be in [0, 1]")

    K0 = np.sqrt(1 - p) * I
    K1 = np.sqrt(p) * Z

    return [K0, K1]


def bit_flip(p: float) -> List[np.ndarray]:
    """
    Bit flip channel: ρ → (1-p)ρ + pXρX.

    Args:
        p: Probability of bit flip

    Returns:
        List of 2 Kraus operators
    """
    if not 0 <= p <= 1:
        raise ValueError("Probability must be in [0, 1]")

    K0 = np.sqrt(1 - p) * I
    K1 = np.sqrt(p) * X

    return [K0, K1]


def bit_phase_flip(p: float) -> List[np.ndarray]:
    """
    Bit-phase flip channel: ρ → (1-p)ρ + pYρY.

    Args:
        p: Probability of bit-phase flip

    Returns:
        List of 2 Kraus operators
    """
    if not 0 <= p <= 1:
        raise ValueError("Probability must be in [0, 1]")

    K0 = np.sqrt(1 - p) * I
    K1 = np.sqrt(p) * Y

    return [K0, K1]


def reset_channel() -> List[np.ndarray]:
    """
    Reset channel: resets qubit to |0⟩.

    Returns:
        List of 2 Kraus operators
    """
    K0 = np.array([
        [1, 0],
        [0, 0]
    ], dtype=complex)

    K1 = np.array([
        [0, 1],
        [0, 0]
    ], dtype=complex)

    return [K0, K1]


def pauli_channel(px: float, py: float, pz: float) -> List[np.ndarray]:
    """
    General Pauli channel: ρ → (1-px-py-pz)ρ + pxXρX + pyYρY + pzZρZ.

    Args:
        px: X error probability
        py: Y error probability
        pz: Z error probability

    Returns:
        List of 4 Kraus operators
    """
    p_total = px + py + pz
    if p_total > 1 or any(p < 0 for p in [px, py, pz]):
        raise ValueError("Invalid probabilities")

    K0 = np.sqrt(1 - p_total) * I
    K1 = np.sqrt(px) * X
    K2 = np.sqrt(py) * Y
    K3 = np.sqrt(pz) * Z

    return [K0, K1, K2, K3]


# =============================================================================
# Two-Qubit Noise Channels
# =============================================================================

def two_qubit_depolarizing(p: float) -> List[np.ndarray]:
    """
    Two-qubit depolarizing channel.
    Applies random two-qubit Pauli error with probability p.

    Args:
        p: Error probability

    Returns:
        List of 16 Kraus operators
    """
    paulis = [I, X, Y, Z]
    kraus_ops = []

    for P1 in paulis:
        for P2 in paulis:
            if np.allclose(P1, I) and np.allclose(P2, I):
                K = np.sqrt(1 - p) * np.kron(P1, P2)
            else:
                K = np.sqrt(p / 15) * np.kron(P1, P2)
            kraus_ops.append(K)

    return kraus_ops


def correlated_dephasing(gamma: float) -> List[np.ndarray]:
    """
    Correlated dephasing on two qubits.
    Both qubits experience the same phase error.

    Args:
        gamma: Dephasing strength

    Returns:
        List of 2 Kraus operators
    """
    ZZ = np.kron(Z, Z)
    II = np.kron(I, I)

    K0 = np.sqrt((1 + np.sqrt(1 - gamma)) / 2) * II
    K1 = np.sqrt((1 - np.sqrt(1 - gamma)) / 2) * ZZ

    return [K0, K1]


# =============================================================================
# Noise Model Class
# =============================================================================

class NoiseModel:
    """
    Configurable noise model for circuit simulation.
    Allows specifying different noise channels for different gates and qubits.
    """

    def __init__(self):
        self._gate_noise: Dict[str, List[np.ndarray]] = {}
        self._qubit_noise: Dict[int, List[np.ndarray]] = {}
        self._readout_error: Dict[int, Tuple[float, float]] = {}
        self._global_noise: Optional[List[np.ndarray]] = None

    def add_gate_noise(self, gate_name: str, kraus_ops: List[np.ndarray]) -> 'NoiseModel':
        """Add noise after specific gate type."""
        self._gate_noise[gate_name] = kraus_ops
        return self

    def add_qubit_noise(self, qubit: int, kraus_ops: List[np.ndarray]) -> 'NoiseModel':
        """Add noise on specific qubit (applied after each gate on that qubit)."""
        self._qubit_noise[qubit] = kraus_ops
        return self

    def add_readout_error(self, qubit: int, p0_to_1: float, p1_to_0: float) -> 'NoiseModel':
        """
        Add readout error (measurement error).

        Args:
            qubit: Qubit index
            p0_to_1: Probability of reading 1 when state is 0
            p1_to_0: Probability of reading 0 when state is 1
        """
        self._readout_error[qubit] = (p0_to_1, p1_to_0)
        return self

    def add_global_noise(self, kraus_ops: List[np.ndarray]) -> 'NoiseModel':
        """Add noise applied after every gate."""
        self._global_noise = kraus_ops
        return self

    def add_depolarizing(self, p: float, gate_names: Optional[List[str]] = None) -> 'NoiseModel':
        """Convenience method to add depolarizing noise to gates."""
        kraus = depolarizing_channel(p)
        if gate_names is None:
            self._global_noise = kraus
        else:
            for name in gate_names:
                self._gate_noise[name] = kraus
        return self

    def add_amplitude_damping(self, gamma: float, qubits: Optional[List[int]] = None) -> 'NoiseModel':
        """Convenience method to add amplitude damping."""
        kraus = amplitude_damping(gamma)
        if qubits is None:
            self._global_noise = kraus
        else:
            for q in qubits:
                self._qubit_noise[q] = kraus
        return self

    def add_phase_damping(self, gamma: float, qubits: Optional[List[int]] = None) -> 'NoiseModel':
        """Convenience method to add phase damping."""
        kraus = phase_damping(gamma)
        if qubits is None:
            self._global_noise = kraus
        else:
            for q in qubits:
                self._qubit_noise[q] = kraus
        return self

    def get_gate_noise(self, gate_name: str) -> Optional[List[np.ndarray]]:
        """Get noise for a specific gate."""
        return self._gate_noise.get(gate_name)

    def get_qubit_noise(self, qubit: int) -> Optional[List[np.ndarray]]:
        """Get noise for a specific qubit."""
        return self._qubit_noise.get(qubit)

    def get_readout_error(self, qubit: int) -> Optional[Tuple[float, float]]:
        """Get readout error for a qubit."""
        return self._readout_error.get(qubit)

    @property
    def global_noise(self) -> Optional[List[np.ndarray]]:
        """Get global noise channel."""
        return self._global_noise

    def apply_readout_error(self, outcome: int, qubits: List[int]) -> int:
        """
        Apply readout error to measurement outcome.

        Args:
            outcome: Original measurement outcome
            qubits: Qubits that were measured

        Returns:
            Possibly flipped outcome
        """
        result = outcome
        for i, q in enumerate(qubits):
            if q in self._readout_error:
                p0_to_1, p1_to_0 = self._readout_error[q]
                bit = (outcome >> i) & 1

                if bit == 0 and np.random.random() < p0_to_1:
                    result ^= (1 << i)  # Flip bit
                elif bit == 1 and np.random.random() < p1_to_0:
                    result ^= (1 << i)  # Flip bit

        return result

    @classmethod
    def from_t1_t2(cls, t1: float, t2: float, gate_time: float) -> 'NoiseModel':
        """
        Create noise model from T1 and T2 times.

        Args:
            t1: T1 relaxation time
            t2: T2 dephasing time
            gate_time: Duration of a single gate

        Returns:
            NoiseModel with appropriate amplitude and phase damping
        """
        # Calculate damping parameters
        gamma_1 = 1 - np.exp(-gate_time / t1)
        gamma_2 = 1 - np.exp(-gate_time / t2)

        # Phase damping rate (T2 includes T1 contribution)
        if t2 < 2 * t1:
            gamma_phi = gamma_2 - gamma_1 / 2
        else:
            gamma_phi = 0

        model = cls()

        # Combine amplitude and phase damping
        amp_kraus = amplitude_damping(gamma_1)
        phase_kraus = phase_damping(gamma_phi)

        # Compose the channels
        composed = []
        for K_amp in amp_kraus:
            for K_phase in phase_kraus:
                composed.append(K_phase @ K_amp)

        model._global_noise = composed
        return model

    def to_dict(self) -> dict:
        """Serialize noise model."""
        return {
            'gate_noise': {k: [op.tolist() for op in v] for k, v in self._gate_noise.items()},
            'readout_error': self._readout_error,
            'has_global_noise': self._global_noise is not None,
        }


# =============================================================================
# Helper Functions
# =============================================================================

def compose_channels(channel1: List[np.ndarray], channel2: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compose two channels: E2(E1(ρ)).

    Args:
        channel1: First channel (applied first)
        channel2: Second channel (applied second)

    Returns:
        Composed channel Kraus operators
    """
    composed = []
    for K2 in channel2:
        for K1 in channel1:
            composed.append(K2 @ K1)
    return composed


def channel_fidelity(kraus_ops: List[np.ndarray]) -> float:
    """
    Calculate average gate fidelity of a channel.
    Measures how close channel is to identity.

    Args:
        kraus_ops: Kraus operators

    Returns:
        Fidelity value in [0, 1]
    """
    d = kraus_ops[0].shape[0]

    # Average fidelity = (d * F_e + 1) / (d + 1)
    # where F_e = (1/d²) Σᵢ |Tr(Kᵢ)|²

    entanglement_fidelity = 0
    for K in kraus_ops:
        entanglement_fidelity += np.abs(np.trace(K)) ** 2
    entanglement_fidelity /= d ** 2

    return (d * entanglement_fidelity + 1) / (d + 1)


# Export all
__all__ = [
    'NoiseChannel',
    'validate_kraus_ops',
    # Single-qubit channels
    'depolarizing_channel',
    'amplitude_damping',
    'generalized_amplitude_damping',
    'phase_damping',
    'phase_flip',
    'bit_flip',
    'bit_phase_flip',
    'reset_channel',
    'pauli_channel',
    # Two-qubit channels
    'two_qubit_depolarizing',
    'correlated_dephasing',
    # Noise model
    'NoiseModel',
    # Utilities
    'compose_channels',
    'channel_fidelity',
]
