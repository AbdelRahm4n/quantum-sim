import { useState } from 'react';
import { GATE_CATEGORIES, GATE_INFO, type GateName } from '../../types';
import { useCircuitStore } from '../../stores/circuitStore';
import { ChevronDown, ChevronRight, Plus, Minus } from 'lucide-react';

export function GatePalette() {
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(['basic', 'controlled'])
  );
  const [selectedGate, setSelectedGate] = useState<GateName | null>(null);
  const [params, setParams] = useState<number[]>([Math.PI / 2]);
  const [targetQubits, setTargetQubits] = useState<number[]>([0]);

  const { nQubits, setNQubits, addGate, addMeasurement, addBarrier } = useCircuitStore();

  const toggleCategory = (category: string) => {
    const next = new Set(expandedCategories);
    if (next.has(category)) {
      next.delete(category);
    } else {
      next.add(category);
    }
    setExpandedCategories(next);
  };

  const handleGateClick = (gate: GateName) => {
    const info = GATE_INFO[gate];
    setSelectedGate(gate);

    // Initialize params
    if (info.numParams > 0) {
      setParams(Array(info.numParams).fill(Math.PI / 2));
    } else {
      setParams([]);
    }

    // Initialize target qubits
    setTargetQubits(Array(info.numQubits).fill(0).map((_, i) => i % nQubits));
  };

  const handleAddGate = () => {
    if (!selectedGate) return;
    addGate(selectedGate, targetQubits, params);
    setSelectedGate(null);
  };

  const handleAddMeasurement = () => {
    for (let i = 0; i < nQubits; i++) {
      addMeasurement(i, i);
    }
  };

  return (
    <div className="p-4">
      {/* Qubit count control */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-slate-300 mb-2">
          Number of Qubits
        </label>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setNQubits(Math.max(1, nQubits - 1))}
            disabled={nQubits <= 1}
            className="p-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded-lg transition-colors"
          >
            <Minus className="w-4 h-4" />
          </button>
          <span className="w-12 text-center text-lg font-medium">{nQubits}</span>
          <button
            onClick={() => setNQubits(Math.min(10, nQubits + 1))}
            disabled={nQubits >= 10}
            className="p-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded-lg transition-colors"
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Gate categories */}
      <div className="space-y-2">
        {Object.entries(GATE_CATEGORIES).map(([key, category]) => (
          <div key={key} className="bg-slate-700/50 rounded-lg overflow-hidden">
            <button
              onClick={() => toggleCategory(key)}
              className="w-full flex items-center justify-between px-3 py-2 hover:bg-slate-700 transition-colors"
            >
              <span className="font-medium text-sm">{category.name}</span>
              {expandedCategories.has(key) ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
            </button>

            {expandedCategories.has(key) && (
              <div className="p-2 grid grid-cols-4 gap-1">
                {category.gates.map((gate) => {
                  const info = GATE_INFO[gate];
                  return (
                    <button
                      key={gate}
                      onClick={() => handleGateClick(gate)}
                      title={info.description}
                      className={`aspect-square flex items-center justify-center text-xs font-bold rounded transition-all ${
                        selectedGate === gate
                          ? 'ring-2 ring-quantum-400 scale-105'
                          : 'hover:scale-105'
                      }`}
                      style={{ backgroundColor: info.color }}
                    >
                      {info.symbol}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Selected gate configuration */}
      {selectedGate && (
        <div className="mt-4 p-3 bg-slate-700/50 rounded-lg">
          <h3 className="font-medium mb-2">{GATE_INFO[selectedGate].displayName}</h3>
          <p className="text-xs text-slate-400 mb-3">
            {GATE_INFO[selectedGate].description}
          </p>

          {/* Target qubits */}
          <div className="space-y-2 mb-3">
            {targetQubits.map((q, idx) => (
              <div key={idx} className="flex items-center gap-2">
                <label className="text-xs text-slate-400 w-16">
                  {GATE_INFO[selectedGate].numQubits === 2 && idx === 0
                    ? 'Control'
                    : `Qubit ${idx + 1}`}
                </label>
                <select
                  value={q}
                  onChange={(e) => {
                    const next = [...targetQubits];
                    next[idx] = parseInt(e.target.value);
                    setTargetQubits(next);
                  }}
                  className="flex-1 bg-slate-600 rounded px-2 py-1 text-sm"
                >
                  {Array.from({ length: nQubits }, (_, i) => (
                    <option key={i} value={i}>
                      q{i}
                    </option>
                  ))}
                </select>
              </div>
            ))}
          </div>

          {/* Parameters */}
          {GATE_INFO[selectedGate].numParams > 0 && (
            <div className="space-y-2 mb-3">
              {params.map((p, idx) => (
                <div key={idx} className="flex items-center gap-2">
                  <label className="text-xs text-slate-400 w-16">
                    {GATE_INFO[selectedGate].paramNames?.[idx] || `θ${idx + 1}`}
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={2 * Math.PI}
                    step={0.01}
                    value={p}
                    onChange={(e) => {
                      const next = [...params];
                      next[idx] = parseFloat(e.target.value);
                      setParams(next);
                    }}
                    className="flex-1"
                  />
                  <span className="text-xs text-slate-400 w-12">
                    {(p / Math.PI).toFixed(2)}π
                  </span>
                </div>
              ))}
            </div>
          )}

          <button
            onClick={handleAddGate}
            className="w-full py-2 bg-quantum-600 hover:bg-quantum-500 rounded-lg font-medium transition-colors"
          >
            Add Gate
          </button>
        </div>
      )}

      {/* Quick actions */}
      <div className="mt-4 space-y-2">
        <button
          onClick={handleAddMeasurement}
          className="w-full py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm font-medium transition-colors"
        >
          Measure All
        </button>
        <button
          onClick={() => addBarrier(Array.from({ length: nQubits }, (_, i) => i))}
          className="w-full py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm font-medium transition-colors"
        >
          Add Barrier
        </button>
      </div>
    </div>
  );
}
