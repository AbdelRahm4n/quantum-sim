import { useState } from 'react';
import { CircuitBuilder } from './components/CircuitBuilder/CircuitBuilder';
import { GatePalette } from './components/CircuitBuilder/GatePalette';
import { BlochSphere } from './components/Visualizations/BlochSphere';
import { Histogram } from './components/Visualizations/Histogram';
import { StateVector } from './components/Visualizations/StateVector';
import { ControlPanel } from './components/ControlPanel';
import { useCircuitStore } from './stores/circuitStore';
import { Settings, Play, RotateCcw, Download } from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState<'histogram' | 'state' | 'bloch'>('histogram');
  const { nQubits, result, executionState } = useCircuitStore();

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      {/* Header */}
      <header className="bg-slate-800 border-b border-slate-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="text-2xl font-bold text-quantum-400">
              Quantum Circuit Simulator
            </h1>
            <span className="text-sm text-slate-400">
              {nQubits} qubit{nQubits > 1 ? 's' : ''}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button className="p-2 hover:bg-slate-700 rounded-lg transition-colors">
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      <div className="flex h-[calc(100vh-73px)]">
        {/* Left sidebar - Gate Palette */}
        <aside className="w-64 bg-slate-800 border-r border-slate-700 overflow-y-auto">
          <GatePalette />
        </aside>

        {/* Main content */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Circuit Builder */}
          <div className="flex-1 overflow-auto p-4">
            <CircuitBuilder />
          </div>

          {/* Control Panel */}
          <div className="border-t border-slate-700 bg-slate-800 p-4">
            <ControlPanel />
          </div>
        </main>

        {/* Right sidebar - Visualizations */}
        <aside className="w-96 bg-slate-800 border-l border-slate-700 flex flex-col">
          {/* Visualization tabs */}
          <div className="flex border-b border-slate-700">
            <button
              onClick={() => setActiveTab('histogram')}
              className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                activeTab === 'histogram'
                  ? 'bg-slate-700 text-white'
                  : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
              }`}
            >
              Histogram
            </button>
            <button
              onClick={() => setActiveTab('state')}
              className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                activeTab === 'state'
                  ? 'bg-slate-700 text-white'
                  : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
              }`}
            >
              State
            </button>
            <button
              onClick={() => setActiveTab('bloch')}
              className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                activeTab === 'bloch'
                  ? 'bg-slate-700 text-white'
                  : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
              }`}
            >
              Bloch
            </button>
          </div>

          {/* Visualization content */}
          <div className="flex-1 overflow-auto p-4">
            {activeTab === 'histogram' && <Histogram />}
            {activeTab === 'state' && <StateVector />}
            {activeTab === 'bloch' && <BlochSphere />}
          </div>

          {/* Results summary */}
          {result && (
            <div className="border-t border-slate-700 p-4">
              <h3 className="text-sm font-medium text-slate-300 mb-2">
                Execution Results
              </h3>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="bg-slate-700/50 rounded p-2">
                  <div className="text-slate-400">Shots</div>
                  <div className="font-medium">{result.shots}</div>
                </div>
                <div className="bg-slate-700/50 rounded p-2">
                  <div className="text-slate-400">Time</div>
                  <div className="font-medium">
                    {result.execution_time_ms.toFixed(2)} ms
                  </div>
                </div>
              </div>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

export default App;
