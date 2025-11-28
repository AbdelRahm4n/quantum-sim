"""Tests for StateVector class."""

import numpy as np
import pytest
from quantum_simulator.core.state_vector import StateVector, MeasurementResult
from quantum_simulator.core import gates


class TestStateVectorInit:
    """Test StateVector initialization."""

    def test_init_ground_state(self):
        """Default init should give |0...0⟩."""
        sv = StateVector(2)
        assert sv.n_qubits == 2
        assert sv.dim == 4
        assert np.allclose(sv.amplitudes, [1, 0, 0, 0])

    def test_init_custom_state(self):
        """Should accept custom initial amplitudes."""
        amps = np.array([1, 0, 0, 1]) / np.sqrt(2)
        sv = StateVector(2, amps)
        assert np.allclose(sv.amplitudes, amps)

    def test_from_label(self):
        """from_label should create computational basis state."""
        sv = StateVector.from_label('01')
        assert np.allclose(sv.amplitudes, [0, 1, 0, 0])

        sv = StateVector.from_label('|10⟩')
        assert np.allclose(sv.amplitudes, [0, 0, 1, 0])

    def test_from_amplitudes(self):
        """from_amplitudes should create normalized state."""
        sv = StateVector.from_amplitudes([1, 1])
        expected = np.array([1, 1]) / np.sqrt(2)
        assert np.allclose(sv.amplitudes, expected)


class TestSpecialStates:
    """Test creation of special quantum states."""

    def test_bell_state_phi_plus(self):
        """Bell state |Φ+⟩ should have correct amplitudes."""
        bell = StateVector.bell_state('phi+')
        expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
        assert np.allclose(bell.amplitudes, expected)

    def test_bell_state_psi_minus(self):
        """Bell state |Ψ-⟩ should have correct amplitudes."""
        bell = StateVector.bell_state('psi-')
        expected = np.array([0, 1, -1, 0]) / np.sqrt(2)
        assert np.allclose(bell.amplitudes, expected)

    def test_ghz_state(self):
        """GHZ state should have correct form."""
        ghz = StateVector.ghz_state(3)
        assert ghz.n_qubits == 3
        # Non-zero at |000⟩ and |111⟩
        assert np.isclose(np.abs(ghz.amplitudes[0]), 1/np.sqrt(2))
        assert np.isclose(np.abs(ghz.amplitudes[7]), 1/np.sqrt(2))
        assert np.isclose(np.sum(np.abs(ghz.amplitudes[1:7])), 0)

    def test_w_state(self):
        """W state should have one excitation per term."""
        w = StateVector.w_state(3)
        # Non-zero at |001⟩, |010⟩, |100⟩
        assert np.isclose(np.abs(w.amplitudes[1]), 1/np.sqrt(3))
        assert np.isclose(np.abs(w.amplitudes[2]), 1/np.sqrt(3))
        assert np.isclose(np.abs(w.amplitudes[4]), 1/np.sqrt(3))


class TestGateApplication:
    """Test gate application."""

    def test_apply_x_gate(self):
        """X gate should flip state."""
        sv = StateVector(1)  # |0⟩
        sv_new = sv.apply_gate(gates.X, [0])
        assert np.allclose(sv_new.amplitudes, [0, 1])  # |1⟩

    def test_apply_h_gate(self):
        """H gate should create superposition."""
        sv = StateVector(1)  # |0⟩
        sv_new = sv.apply_gate(gates.H, [0])
        expected = np.array([1, 1]) / np.sqrt(2)
        assert np.allclose(sv_new.amplitudes, expected)

    def test_apply_cnot(self):
        """CNOT should entangle qubits."""
        # Start with |00⟩, apply H to q0, then CNOT
        sv = StateVector(2)
        sv = sv.apply_gate(gates.H, [0])
        sv = sv.apply_gate(gates.CNOT, [0, 1])
        # Should be Bell state (|00⟩ + |11⟩)/√2
        expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
        assert np.allclose(sv.amplitudes, expected)

    def test_apply_gate_inplace(self):
        """apply_gate_inplace should modify state."""
        sv = StateVector(1)
        sv.apply_gate_inplace(gates.X, [0])
        assert np.allclose(sv.amplitudes, [0, 1])


class TestMeasurement:
    """Test measurement operations."""

    def test_measure_deterministic(self):
        """Measuring |0⟩ should always give 0."""
        sv = StateVector(1)  # |0⟩
        result = sv.measure([0])
        assert result.outcome == 0
        assert result.probability == 1.0

    def test_measure_superposition_statistics(self):
        """Measuring superposition should give ~50/50 outcomes."""
        np.random.seed(42)
        sv = StateVector(1)
        sv = sv.apply_gate(gates.H, [0])  # |+⟩

        outcomes = []
        for _ in range(1000):
            sv_copy = sv.copy()
            result = sv_copy.measure([0])
            outcomes.append(result.outcome)

        # Should be roughly 50/50
        zero_count = outcomes.count(0)
        assert 400 < zero_count < 600

    def test_measure_collapse(self):
        """State should collapse after measurement."""
        sv = StateVector(1)
        sv = sv.apply_gate(gates.H, [0])  # |+⟩
        result = sv.measure([0])

        # Post-measurement state should be deterministic
        if result.outcome == 0:
            assert np.allclose(result.post_state.amplitudes, [1, 0])
        else:
            assert np.allclose(result.post_state.amplitudes, [0, 1])

    def test_sample(self):
        """sample should return counts dictionary."""
        sv = StateVector.bell_state('phi+')
        counts = sv.sample(shots=1000)

        # Should only see '00' and '11'
        assert set(counts.keys()).issubset({'00', '11'})
        assert sum(counts.values()) == 1000


class TestExpectation:
    """Test expectation value calculations."""

    def test_expectation_z_on_zero(self):
        """⟨0|Z|0⟩ = 1."""
        sv = StateVector(1)  # |0⟩
        exp = sv.expectation(gates.Z)
        assert np.isclose(exp, 1.0)

    def test_expectation_z_on_one(self):
        """⟨1|Z|1⟩ = -1."""
        sv = StateVector(1)
        sv = sv.apply_gate(gates.X, [0])  # |1⟩
        exp = sv.expectation(gates.Z)
        assert np.isclose(exp, -1.0)

    def test_expectation_z_on_plus(self):
        """⟨+|Z|+⟩ = 0."""
        sv = StateVector(1)
        sv = sv.apply_gate(gates.H, [0])  # |+⟩
        exp = sv.expectation(gates.Z)
        assert np.isclose(exp, 0.0, atol=1e-10)


class TestBlochSphere:
    """Test Bloch sphere calculations."""

    def test_bloch_zero_state(self):
        """|0⟩ should be at north pole (0, 0, 1)."""
        sv = StateVector(1)
        x, y, z = sv.bloch_vector(0)
        assert np.allclose([x, y, z], [0, 0, 1], atol=1e-10)

    def test_bloch_one_state(self):
        """|1⟩ should be at south pole (0, 0, -1)."""
        sv = StateVector(1)
        sv = sv.apply_gate(gates.X, [0])
        x, y, z = sv.bloch_vector(0)
        assert np.allclose([x, y, z], [0, 0, -1], atol=1e-10)

    def test_bloch_plus_state(self):
        """|+⟩ should be on equator at (1, 0, 0)."""
        sv = StateVector(1)
        sv = sv.apply_gate(gates.H, [0])
        x, y, z = sv.bloch_vector(0)
        assert np.allclose([x, y, z], [1, 0, 0], atol=1e-10)


class TestFidelity:
    """Test fidelity calculations."""

    def test_fidelity_same_state(self):
        """Fidelity of state with itself should be 1."""
        sv = StateVector(2)
        sv = sv.apply_gate(gates.H, [0])
        assert np.isclose(sv.fidelity(sv), 1.0)

    def test_fidelity_orthogonal(self):
        """Fidelity of orthogonal states should be 0."""
        sv0 = StateVector(1)  # |0⟩
        sv1 = StateVector(1)
        sv1 = sv1.apply_gate(gates.X, [0])  # |1⟩
        assert np.isclose(sv0.fidelity(sv1), 0.0)


class TestSerialization:
    """Test state vector serialization."""

    def test_to_dict_from_dict(self):
        """State should survive serialization round-trip."""
        sv = StateVector.bell_state('phi+')
        data = sv.to_dict()
        sv_restored = StateVector.from_dict(data)
        assert np.allclose(sv.amplitudes, sv_restored.amplitudes)
