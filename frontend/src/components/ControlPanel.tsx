import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Play, Pause, RotateCcw, StepForward, Download, Trash2 } from 'lucide-react';
import { useCircuitStore } from '../stores/circuitStore';
import { createCircuit, runCircuit, exportOpenQASM } from '../api/client';

export function ControlPanel() {
  const [isRunning, setIsRunning] = useState(false);

  const {
    nQubits,
    name,
    operations,
    shots,
    setShots,
    mode,
    setMode,
    recordSnapshots,
    setRecordSnapshots,
    circuitId,
    setCircuitId,
    setResult,
    setBlochVectors,
    setProbabilities,
    clearOperations,
    resetExecution,
  } = useCircuitStore();

  // Create and run circuit mutation
  const runMutation = useMutation({
    mutationFn: async () => {
      // Create circuit
      const circuit = await createCircuit(nQubits, name, operations);
      setCircuitId(circuit.id);

      // Run circuit
      const result = await runCircuit(circuit.id, {
        shots,
        mode,
        record_snapshots: recordSnapshots,
      });

      return result;
    },
    onSuccess: (result) => {
      setResult(result);

      // Update visualization from snapshots
      if (result.snapshots.length > 0) {
        const lastSnapshot = result.snapshots[result.snapshots.length - 1];
        setBlochVectors(lastSnapshot.bloch_vectors);
        setProbabilities(lastSnapshot.probabilities);
      }
    },
    onError: (error) => {
      console.error('Error running circuit:', error);
      alert('Error running circuit. Check console for details.');
    },
  });

  // Export QASM mutation
  const exportMutation = useMutation({
    mutationFn: async () => {
      if (!circuitId) {
        // Create circuit first
        const circuit = await createCircuit(nQubits, name, operations);
        setCircuitId(circuit.id);
        return exportOpenQASM(circuit.id);
      }
      return exportOpenQASM(circuitId);
    },
    onSuccess: (data) => {
      // Download as file
      const blob = new Blob([data.qasm], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${name}.qasm`;
      a.click();
      URL.revokeObjectURL(url);
    },
  });

  const handleRun = () => {
    setIsRunning(true);
    runMutation.mutate();
    setIsRunning(false);
  };

  const handleReset = () => {
    resetExecution();
    setCircuitId(null);
  };

  const handleClear = () => {
    clearOperations();
    resetExecution();
    setCircuitId(null);
  };

  return (
    <div className="flex items-center justify-between gap-4">
      {/* Left: Run controls */}
      <div className="flex items-center gap-2">
        <button
          onClick={handleRun}
          disabled={runMutation.isPending || operations.length === 0}
          className="flex items-center gap-2 px-4 py-2 bg-quantum-600 hover:bg-quantum-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-medium transition-colors"
        >
          {runMutation.isPending ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Running...
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Run
            </>
          )}
        </button>

        <button
          onClick={handleReset}
          className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
          title="Reset execution"
        >
          <RotateCcw className="w-4 h-4" />
        </button>

        <button
          onClick={handleClear}
          className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
          title="Clear circuit"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      {/* Center: Settings */}
      <div className="flex items-center gap-4">
        {/* Shots */}
        <div className="flex items-center gap-2">
          <label className="text-sm text-slate-400">Shots:</label>
          <select
            value={shots}
            onChange={(e) => setShots(parseInt(e.target.value))}
            className="bg-slate-700 rounded px-2 py-1 text-sm"
          >
            <option value={100}>100</option>
            <option value={1024}>1024</option>
            <option value={4096}>4096</option>
            <option value={10000}>10000</option>
          </select>
        </div>

        {/* Mode */}
        <div className="flex items-center gap-2">
          <label className="text-sm text-slate-400">Mode:</label>
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as 'statevector' | 'density_matrix')}
            className="bg-slate-700 rounded px-2 py-1 text-sm"
          >
            <option value="statevector">State Vector</option>
            <option value="density_matrix">Density Matrix</option>
          </select>
        </div>

        {/* Record snapshots toggle */}
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={recordSnapshots}
            onChange={(e) => setRecordSnapshots(e.target.checked)}
            className="rounded border-slate-500"
          />
          <span className="text-sm text-slate-400">Record steps</span>
        </label>
      </div>

      {/* Right: Export */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => exportMutation.mutate()}
          disabled={exportMutation.isPending || operations.length === 0}
          className="flex items-center gap-2 px-3 py-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded-lg text-sm transition-colors"
        >
          <Download className="w-4 h-4" />
          Export QASM
        </button>

        {/* Circuit stats */}
        <div className="text-xs text-slate-500 ml-2">
          {operations.length} operations
        </div>
      </div>
    </div>
  );
}
