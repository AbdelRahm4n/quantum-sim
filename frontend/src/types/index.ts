// Gate types
export type SingleQubitGate =
  | 'I' | 'X' | 'Y' | 'Z' | 'H' | 'S' | 'Sdg' | 'T' | 'Tdg' | 'SX' | 'SXdg';

export type RotationGate =
  | 'Rx' | 'Ry' | 'Rz' | 'Phase' | 'U1' | 'U2' | 'U3';

export type TwoQubitGate =
  | 'CX' | 'CY' | 'CZ' | 'SWAP' | 'iSWAP' | 'CRx' | 'CRy' | 'CRz' | 'CPhase'
  | 'Rxx' | 'Ryy' | 'Rzz';

export type ThreeQubitGate = 'CCX' | 'CSWAP';

export type GateName = SingleQubitGate | RotationGate | TwoQubitGate | ThreeQubitGate;

// Gate operation
export interface GateOperation {
  gate_name: GateName;
  qubits: number[];
  params: number[];
  label?: string;
}

// Measurement operation
export interface MeasurementOperation {
  qubits: number[];
  classical_bits: number[];
  basis: 'X' | 'Y' | 'Z';
}

// Circuit operation
export interface CircuitOperation {
  type: 'gate' | 'measurement' | 'barrier' | 'reset';
  gate?: GateOperation;
  measurement?: MeasurementOperation;
  qubits: number[];
}

// Circuit
export interface Circuit {
  id: string;
  n_qubits: number;
  n_classical: number;
  name: string;
  operations: CircuitOperation[];
  depth: number;
  gate_count: Record<string, number>;
}

// Bloch vector
export interface BlochVector {
  qubit: number;
  x: number;
  y: number;
  z: number;
}

// State snapshot
export interface StateSnapshot {
  step: number;
  operation_type: string;
  operation_name: string;
  qubits: number[];
  params: number[];
  probabilities: number[];
  bloch_vectors: BlochVector[];
  measurement_outcome?: number;
}

// Execution result
export interface ExecutionResult {
  circuit_id: string;
  counts: Record<string, number>;
  shots: number;
  probabilities: Record<string, number>;
  execution_time_ms: number;
  snapshots: StateSnapshot[];
}

// State vector response
export interface StateVectorResponse {
  circuit_id: string;
  n_qubits: number;
  amplitudes_real: number[];
  amplitudes_imag: number[];
  probabilities: number[];
  bloch_vectors: BlochVector[];
}

// Noise configuration
export interface NoiseConfig {
  depolarizing_rate?: number;
  amplitude_damping?: number;
  phase_damping?: number;
  readout_error?: Record<number, [number, number]>;
}

// Execution request
export interface ExecutionRequest {
  shots: number;
  mode: 'statevector' | 'density_matrix';
  noise?: NoiseConfig;
  record_snapshots: boolean;
}

// Gate metadata for UI
export interface GateInfo {
  name: GateName;
  displayName: string;
  description: string;
  numQubits: number;
  numParams: number;
  paramNames?: string[];
  color: string;
  symbol: string;
}

// Gate palette categories
export const GATE_CATEGORIES = {
  basic: {
    name: 'Basic',
    gates: ['I', 'X', 'Y', 'Z', 'H', 'S', 'T'] as GateName[],
  },
  rotation: {
    name: 'Rotation',
    gates: ['Rx', 'Ry', 'Rz', 'Phase'] as GateName[],
  },
  controlled: {
    name: 'Controlled',
    gates: ['CX', 'CY', 'CZ', 'CRx', 'CRy', 'CRz'] as GateName[],
  },
  entangling: {
    name: 'Entangling',
    gates: ['SWAP', 'iSWAP', 'CCX', 'CSWAP'] as GateName[],
  },
};

// Gate info lookup
export const GATE_INFO: Record<GateName, GateInfo> = {
  I: { name: 'I', displayName: 'Identity', description: 'Identity gate', numQubits: 1, numParams: 0, color: '#64748b', symbol: 'I' },
  X: { name: 'X', displayName: 'Pauli-X', description: 'NOT gate, bit flip', numQubits: 1, numParams: 0, color: '#ef4444', symbol: 'X' },
  Y: { name: 'Y', displayName: 'Pauli-Y', description: 'Y rotation by π', numQubits: 1, numParams: 0, color: '#22c55e', symbol: 'Y' },
  Z: { name: 'Z', displayName: 'Pauli-Z', description: 'Phase flip', numQubits: 1, numParams: 0, color: '#3b82f6', symbol: 'Z' },
  H: { name: 'H', displayName: 'Hadamard', description: 'Creates superposition', numQubits: 1, numParams: 0, color: '#f59e0b', symbol: 'H' },
  S: { name: 'S', displayName: 'S Gate', description: 'π/2 phase', numQubits: 1, numParams: 0, color: '#8b5cf6', symbol: 'S' },
  Sdg: { name: 'Sdg', displayName: 'S†', description: '-π/2 phase', numQubits: 1, numParams: 0, color: '#8b5cf6', symbol: 'S†' },
  T: { name: 'T', displayName: 'T Gate', description: 'π/4 phase', numQubits: 1, numParams: 0, color: '#ec4899', symbol: 'T' },
  Tdg: { name: 'Tdg', displayName: 'T†', description: '-π/4 phase', numQubits: 1, numParams: 0, color: '#ec4899', symbol: 'T†' },
  SX: { name: 'SX', displayName: '√X', description: 'Square root of X', numQubits: 1, numParams: 0, color: '#ef4444', symbol: '√X' },
  SXdg: { name: 'SXdg', displayName: '√X†', description: 'Inverse √X', numQubits: 1, numParams: 0, color: '#ef4444', symbol: '√X†' },
  Rx: { name: 'Rx', displayName: 'Rx(θ)', description: 'X-axis rotation', numQubits: 1, numParams: 1, paramNames: ['θ'], color: '#ef4444', symbol: 'Rx' },
  Ry: { name: 'Ry', displayName: 'Ry(θ)', description: 'Y-axis rotation', numQubits: 1, numParams: 1, paramNames: ['θ'], color: '#22c55e', symbol: 'Ry' },
  Rz: { name: 'Rz', displayName: 'Rz(θ)', description: 'Z-axis rotation', numQubits: 1, numParams: 1, paramNames: ['θ'], color: '#3b82f6', symbol: 'Rz' },
  Phase: { name: 'Phase', displayName: 'P(θ)', description: 'Phase gate', numQubits: 1, numParams: 1, paramNames: ['θ'], color: '#8b5cf6', symbol: 'P' },
  U1: { name: 'U1', displayName: 'U1(λ)', description: 'U1 gate', numQubits: 1, numParams: 1, paramNames: ['λ'], color: '#06b6d4', symbol: 'U1' },
  U2: { name: 'U2', displayName: 'U2(φ,λ)', description: 'U2 gate', numQubits: 1, numParams: 2, paramNames: ['φ', 'λ'], color: '#06b6d4', symbol: 'U2' },
  U3: { name: 'U3', displayName: 'U3(θ,φ,λ)', description: 'Universal gate', numQubits: 1, numParams: 3, paramNames: ['θ', 'φ', 'λ'], color: '#06b6d4', symbol: 'U3' },
  CX: { name: 'CX', displayName: 'CNOT', description: 'Controlled-X', numQubits: 2, numParams: 0, color: '#ef4444', symbol: '⊕' },
  CY: { name: 'CY', displayName: 'CY', description: 'Controlled-Y', numQubits: 2, numParams: 0, color: '#22c55e', symbol: 'CY' },
  CZ: { name: 'CZ', displayName: 'CZ', description: 'Controlled-Z', numQubits: 2, numParams: 0, color: '#3b82f6', symbol: 'CZ' },
  SWAP: { name: 'SWAP', displayName: 'SWAP', description: 'Swap qubits', numQubits: 2, numParams: 0, color: '#f59e0b', symbol: '×' },
  iSWAP: { name: 'iSWAP', displayName: 'iSWAP', description: 'iSWAP gate', numQubits: 2, numParams: 0, color: '#f59e0b', symbol: 'i×' },
  CRx: { name: 'CRx', displayName: 'CRx(θ)', description: 'Controlled Rx', numQubits: 2, numParams: 1, paramNames: ['θ'], color: '#ef4444', symbol: 'CRx' },
  CRy: { name: 'CRy', displayName: 'CRy(θ)', description: 'Controlled Ry', numQubits: 2, numParams: 1, paramNames: ['θ'], color: '#22c55e', symbol: 'CRy' },
  CRz: { name: 'CRz', displayName: 'CRz(θ)', description: 'Controlled Rz', numQubits: 2, numParams: 1, paramNames: ['θ'], color: '#3b82f6', symbol: 'CRz' },
  CPhase: { name: 'CPhase', displayName: 'CP(θ)', description: 'Controlled Phase', numQubits: 2, numParams: 1, paramNames: ['θ'], color: '#8b5cf6', symbol: 'CP' },
  Rxx: { name: 'Rxx', displayName: 'Rxx(θ)', description: 'XX rotation', numQubits: 2, numParams: 1, paramNames: ['θ'], color: '#ef4444', symbol: 'Rxx' },
  Ryy: { name: 'Ryy', displayName: 'Ryy(θ)', description: 'YY rotation', numQubits: 2, numParams: 1, paramNames: ['θ'], color: '#22c55e', symbol: 'Ryy' },
  Rzz: { name: 'Rzz', displayName: 'Rzz(θ)', description: 'ZZ rotation', numQubits: 2, numParams: 1, paramNames: ['θ'], color: '#3b82f6', symbol: 'Rzz' },
  CCX: { name: 'CCX', displayName: 'Toffoli', description: 'CC-NOT gate', numQubits: 3, numParams: 0, color: '#ef4444', symbol: '⊕' },
  CSWAP: { name: 'CSWAP', displayName: 'Fredkin', description: 'Controlled SWAP', numQubits: 3, numParams: 0, color: '#f59e0b', symbol: 'C×' },
};
