"""Tests for quantum gates."""

import numpy as np
import pytest
from quantum_simulator.core import gates


class TestSingleQubitGates:
    """Test single-qubit gate properties."""

    def test_identity_is_identity(self):
        """I gate should be identity matrix."""
        assert np.allclose(gates.I, np.eye(2))

    def test_pauli_matrices_hermitian(self):
        """Pauli matrices should be Hermitian."""
        for gate in [gates.X, gates.Y, gates.Z]:
            assert np.allclose(gate, gate.conj().T)

    def test_pauli_matrices_unitary(self):
        """Pauli matrices should be unitary."""
        for gate in [gates.X, gates.Y, gates.Z]:
            assert gates.is_unitary(gate)

    def test_pauli_x_flips_state(self):
        """X gate should flip |0⟩ to |1⟩."""
        zero = np.array([1, 0])
        one = np.array([0, 1])
        assert np.allclose(gates.X @ zero, one)
        assert np.allclose(gates.X @ one, zero)

    def test_hadamard_creates_superposition(self):
        """H|0⟩ should give equal superposition."""
        zero = np.array([1, 0])
        result = gates.H @ zero
        expected = np.array([1, 1]) / np.sqrt(2)
        assert np.allclose(result, expected)

    def test_hadamard_unitary(self):
        """Hadamard should be unitary."""
        assert gates.is_unitary(gates.H)

    def test_hadamard_self_inverse(self):
        """H² = I."""
        assert np.allclose(gates.H @ gates.H, gates.I)

    def test_s_gate_is_sqrt_z(self):
        """S² = Z."""
        assert np.allclose(gates.S @ gates.S, gates.Z)

    def test_t_gate_is_sqrt_s(self):
        """T² = S."""
        assert np.allclose(gates.T @ gates.T, gates.S)

    def test_pauli_algebra(self):
        """XY = iZ, YZ = iX, ZX = iY."""
        assert np.allclose(gates.X @ gates.Y, 1j * gates.Z)
        assert np.allclose(gates.Y @ gates.Z, 1j * gates.X)
        assert np.allclose(gates.Z @ gates.X, 1j * gates.Y)


class TestRotationGates:
    """Test rotation gate properties."""

    def test_rx_at_pi(self):
        """Rx(π) should equal -iX."""
        rx_pi = gates.Rx(np.pi)
        assert np.allclose(rx_pi, -1j * gates.X)

    def test_ry_at_pi(self):
        """Ry(π) should equal -iY."""
        ry_pi = gates.Ry(np.pi)
        assert np.allclose(ry_pi, -1j * gates.Y)

    def test_rz_at_pi(self):
        """Rz(π) should equal -iZ."""
        rz_pi = gates.Rz(np.pi)
        assert np.allclose(rz_pi, -1j * gates.Z)

    def test_rotation_at_zero_is_identity(self):
        """R(0) = I for all rotations."""
        assert np.allclose(gates.Rx(0), gates.I)
        assert np.allclose(gates.Ry(0), gates.I)
        assert np.allclose(gates.Rz(0), gates.I)

    def test_rotation_unitary(self):
        """All rotation gates should be unitary."""
        for theta in [0, np.pi/4, np.pi/2, np.pi, 2*np.pi]:
            assert gates.is_unitary(gates.Rx(theta))
            assert gates.is_unitary(gates.Ry(theta))
            assert gates.is_unitary(gates.Rz(theta))

    def test_u3_covers_bloch_sphere(self):
        """U3 should be able to create any single-qubit state."""
        # |+⟩ state
        u_plus = gates.U3(np.pi/2, 0, 0)
        zero = np.array([1, 0])
        result = u_plus @ zero
        expected = np.array([1, 1]) / np.sqrt(2)
        assert np.allclose(np.abs(result), np.abs(expected))


class TestTwoQubitGates:
    """Test two-qubit gate properties."""

    def test_cnot_unitary(self):
        """CNOT should be unitary."""
        assert gates.is_unitary(gates.CNOT)

    def test_cnot_action(self):
        """CNOT should flip target when control is |1⟩."""
        # |00⟩ → |00⟩
        assert np.allclose(gates.CNOT @ np.array([1,0,0,0]), [1,0,0,0])
        # |01⟩ → |01⟩
        assert np.allclose(gates.CNOT @ np.array([0,1,0,0]), [0,1,0,0])
        # |10⟩ → |11⟩
        assert np.allclose(gates.CNOT @ np.array([0,0,1,0]), [0,0,0,1])
        # |11⟩ → |10⟩
        assert np.allclose(gates.CNOT @ np.array([0,0,0,1]), [0,0,1,0])

    def test_cz_symmetric(self):
        """CZ should be symmetric in qubits."""
        assert gates.is_unitary(gates.CZ)
        # CZ is diagonal, so symmetric
        assert np.allclose(gates.CZ, gates.CZ.T)

    def test_swap_action(self):
        """SWAP should exchange qubits."""
        # |01⟩ → |10⟩
        assert np.allclose(gates.SWAP @ np.array([0,1,0,0]), [0,0,1,0])
        # |10⟩ → |01⟩
        assert np.allclose(gates.SWAP @ np.array([0,0,1,0]), [0,1,0,0])

    def test_swap_self_inverse(self):
        """SWAP² = I."""
        assert np.allclose(gates.SWAP @ gates.SWAP, np.eye(4))


class TestThreeQubitGates:
    """Test three-qubit gate properties."""

    def test_toffoli_unitary(self):
        """Toffoli should be unitary."""
        assert gates.is_unitary(gates.TOFFOLI)

    def test_toffoli_action(self):
        """Toffoli should flip target only when both controls are |1⟩."""
        # |110⟩ → |111⟩
        input_state = np.zeros(8)
        input_state[6] = 1  # |110⟩
        result = gates.TOFFOLI @ input_state
        expected = np.zeros(8)
        expected[7] = 1  # |111⟩
        assert np.allclose(result, expected)

        # |100⟩ → |100⟩ (unchanged)
        input_state = np.zeros(8)
        input_state[4] = 1
        result = gates.TOFFOLI @ input_state
        assert np.allclose(result, input_state)

    def test_fredkin_unitary(self):
        """Fredkin should be unitary."""
        assert gates.is_unitary(gates.FREDKIN)


class TestHelperFunctions:
    """Test gate helper functions."""

    def test_tensor_gate_single_qubit(self):
        """tensor_gate should correctly expand single-qubit gate."""
        # X on qubit 0 of 2-qubit system: I ⊗ X
        x_q0 = gates.tensor_gate(gates.X, 0, 2)
        expected = np.kron(gates.I, gates.X)
        assert np.allclose(x_q0, expected)

        # X on qubit 1 of 2-qubit system: X ⊗ I
        x_q1 = gates.tensor_gate(gates.X, 1, 2)
        expected = np.kron(gates.X, gates.I)
        assert np.allclose(x_q1, expected)

    def test_controlled_creates_controlled_gate(self):
        """controlled() should create proper controlled gate."""
        # Controlled-X (CNOT)
        cx = gates.controlled(gates.X, 1)
        assert np.allclose(cx, gates.CNOT)

    def test_multi_qubit_gate_embedding(self):
        """multi_qubit_gate should correctly embed multi-qubit gates."""
        # CNOT on qubits [0,1] of 3-qubit system
        cnot_01 = gates.multi_qubit_gate(gates.CNOT, [0, 1], 3)
        assert cnot_01.shape == (8, 8)
        assert gates.is_unitary(cnot_01)

    def test_get_gate_basic(self):
        """get_gate should return correct gates."""
        assert np.allclose(gates.get_gate('H'), gates.H)
        assert np.allclose(gates.get_gate('X'), gates.X)
        assert np.allclose(gates.get_gate('CNOT'), gates.CNOT)

    def test_get_gate_parameterized(self):
        """get_gate should handle parameterized gates."""
        rx_pi = gates.get_gate('Rx', [np.pi])
        assert np.allclose(rx_pi, gates.Rx(np.pi))

    def test_gate_fidelity(self):
        """gate_fidelity should return 1 for identical gates."""
        assert np.isclose(gates.gate_fidelity(gates.H, gates.H), 1.0)
        assert gates.gate_fidelity(gates.X, gates.Z) < 1.0
