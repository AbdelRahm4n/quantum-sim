import axios from 'axios';
import type {
  Circuit,
  CircuitOperation,
  ExecutionRequest,
  ExecutionResult,
  StateVectorResponse,
  BlochVector,
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || '';

const api = axios.create({
  baseURL: `${API_BASE}/api`,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Circuit API
export async function createCircuit(
  nQubits: number,
  name: string,
  operations: CircuitOperation[]
): Promise<Circuit> {
  const response = await api.post('/circuit', {
    n_qubits: nQubits,
    name,
    operations,
  });
  return response.data;
}

export async function getCircuit(circuitId: string): Promise<Circuit> {
  const response = await api.get(`/circuit/${circuitId}`);
  return response.data;
}

export async function updateCircuit(
  circuitId: string,
  operations: CircuitOperation[]
): Promise<Circuit> {
  const response = await api.put(`/circuit/${circuitId}`, { operations });
  return response.data;
}

export async function deleteCircuit(circuitId: string): Promise<void> {
  await api.delete(`/circuit/${circuitId}`);
}

// Execution API
export async function runCircuit(
  circuitId: string,
  request: ExecutionRequest
): Promise<ExecutionResult> {
  const response = await api.post(`/circuit/${circuitId}/run`, request);
  return response.data;
}

export async function getStateVector(
  circuitId: string
): Promise<StateVectorResponse> {
  const response = await api.get(`/circuit/${circuitId}/state`);
  return response.data;
}

export async function getBlochVectors(
  circuitId: string
): Promise<{ circuit_id: string; bloch_vectors: BlochVector[] }> {
  const response = await api.get(`/circuit/${circuitId}/bloch`);
  return response.data;
}

export async function getProbabilities(
  circuitId: string
): Promise<{ circuit_id: string; probabilities: Record<string, number> }> {
  const response = await api.get(`/circuit/${circuitId}/probabilities`);
  return response.data;
}

// Analysis API
export async function getCircuitStats(circuitId: string): Promise<{
  circuit_id: string;
  n_qubits: number;
  depth: number;
  total_gates: number;
  gate_counts: Record<string, number>;
  two_qubit_gates: number;
  measurements: number;
}> {
  const response = await api.get(`/circuit/${circuitId}/stats`);
  return response.data;
}

// Export API
export async function exportOpenQASM(
  circuitId: string,
  version: string = '2.0'
): Promise<{ circuit_id: string; qasm: string; version: string }> {
  const response = await api.get(`/circuit/${circuitId}/openqasm`, {
    params: { version },
  });
  return response.data;
}

// WebSocket for real-time execution
export function createExecutionWebSocket(
  circuitId: string,
  onMessage: (data: unknown) => void,
  onError?: (error: Event) => void,
  onClose?: () => void
): WebSocket {
  const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/circuit/${circuitId}/execute`;
  const ws = new WebSocket(wsUrl);

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    onMessage(data);
  };

  ws.onerror = (event) => {
    console.error('WebSocket error:', event);
    onError?.(event);
  };

  ws.onclose = () => {
    onClose?.();
  };

  return ws;
}

// Health check
export async function healthCheck(): Promise<{ status: string; version: string }> {
  const response = await axios.get(`${API_BASE}/health`);
  return response.data;
}
