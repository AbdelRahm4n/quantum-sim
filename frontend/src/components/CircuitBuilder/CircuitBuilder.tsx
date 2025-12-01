import { useCircuitStore } from '../../stores/circuitStore';
import { GATE_INFO } from '../../types';
import { Trash2 } from 'lucide-react';

const CELL_WIDTH = 60;
const CELL_HEIGHT = 60;
const WIRE_Y_OFFSET = CELL_HEIGHT / 2;

export function CircuitBuilder() {
  const { nQubits, operations, removeOperation, currentStep, executionState } = useCircuitStore();

  // Calculate circuit width
  const circuitWidth = Math.max(operations.length + 2, 10) * CELL_WIDTH;

  return (
    <div className="bg-slate-800/50 rounded-xl p-4 overflow-auto">
      <svg
        width={circuitWidth + 100}
        height={nQubits * CELL_HEIGHT + 40}
        className="select-none"
      >
        {/* Qubit labels */}
        {Array.from({ length: nQubits }, (_, i) => (
          <g key={`label-${i}`}>
            <text
              x={40}
              y={20 + i * CELL_HEIGHT + WIRE_Y_OFFSET}
              textAnchor="end"
              dominantBaseline="middle"
              className="fill-slate-400 text-sm font-mono"
            >
              q{i}
            </text>
            <text
              x={40}
              y={20 + i * CELL_HEIGHT + WIRE_Y_OFFSET + 12}
              textAnchor="end"
              dominantBaseline="middle"
              className="fill-slate-500 text-xs font-mono"
            >
              |0⟩
            </text>
          </g>
        ))}

        {/* Qubit wires */}
        {Array.from({ length: nQubits }, (_, i) => (
          <line
            key={`wire-${i}`}
            x1={60}
            y1={20 + i * CELL_HEIGHT + WIRE_Y_OFFSET}
            x2={circuitWidth + 60}
            y2={20 + i * CELL_HEIGHT + WIRE_Y_OFFSET}
            stroke="#475569"
            strokeWidth={2}
          />
        ))}

        {/* Operations */}
        {operations.map((op, opIndex) => {
          const x = 80 + opIndex * CELL_WIDTH;
          const isCurrentStep = executionState === 'running' && opIndex === currentStep;

          if (op.type === 'gate' && op.gate) {
            const info = GATE_INFO[op.gate.gate_name];
            const qubits = op.gate.qubits;

            if (qubits.length === 1) {
              // Single-qubit gate
              const y = 20 + qubits[0] * CELL_HEIGHT + WIRE_Y_OFFSET;
              return (
                <g
                  key={`op-${opIndex}`}
                  className={`cursor-pointer transition-opacity ${
                    isCurrentStep ? 'opacity-100' : 'hover:opacity-80'
                  }`}
                  onClick={() => removeOperation(opIndex)}
                >
                  {isCurrentStep && (
                    <rect
                      x={x - 25}
                      y={y - 25}
                      width={50}
                      height={50}
                      rx={8}
                      fill="none"
                      stroke="#0c8ee6"
                      strokeWidth={2}
                      className="animate-pulse"
                    />
                  )}
                  <rect
                    x={x - 20}
                    y={y - 20}
                    width={40}
                    height={40}
                    rx={4}
                    fill={info?.color || '#64748b'}
                  />
                  <text
                    x={x}
                    y={y}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    className="fill-white text-xs font-bold pointer-events-none"
                  >
                    {info?.symbol || op.gate.gate_name}
                  </text>
                  {op.gate.params.length > 0 && (
                    <text
                      x={x}
                      y={y + 28}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      className="fill-slate-400 text-[10px] pointer-events-none"
                    >
                      {(op.gate.params[0] / Math.PI).toFixed(1)}π
                    </text>
                  )}
                </g>
              );
            } else if (qubits.length === 2) {
              // Two-qubit gate (e.g., CNOT)
              const y1 = 20 + qubits[0] * CELL_HEIGHT + WIRE_Y_OFFSET;
              const y2 = 20 + qubits[1] * CELL_HEIGHT + WIRE_Y_OFFSET;
              const minY = Math.min(y1, y2);
              const maxY = Math.max(y1, y2);
              const isCNOT = ['CX', 'CY', 'CZ'].includes(op.gate.gate_name);

              return (
                <g
                  key={`op-${opIndex}`}
                  className="cursor-pointer hover:opacity-80"
                  onClick={() => removeOperation(opIndex)}
                >
                  {/* Connection line */}
                  <line
                    x1={x}
                    y1={minY}
                    x2={x}
                    y2={maxY}
                    stroke={info?.color || '#64748b'}
                    strokeWidth={2}
                  />

                  {isCNOT ? (
                    <>
                      {/* Control dot */}
                      <circle
                        cx={x}
                        cy={y1}
                        r={6}
                        fill={info?.color || '#64748b'}
                      />
                      {/* Target (XOR symbol) */}
                      <circle
                        cx={x}
                        cy={y2}
                        r={16}
                        fill="none"
                        stroke={info?.color || '#64748b'}
                        strokeWidth={2}
                      />
                      <line
                        x1={x - 16}
                        y1={y2}
                        x2={x + 16}
                        y2={y2}
                        stroke={info?.color || '#64748b'}
                        strokeWidth={2}
                      />
                      <line
                        x1={x}
                        y1={y2 - 16}
                        x2={x}
                        y2={y2 + 16}
                        stroke={info?.color || '#64748b'}
                        strokeWidth={2}
                      />
                    </>
                  ) : (
                    <>
                      {/* Generic two-qubit gate boxes */}
                      <rect
                        x={x - 20}
                        y={y1 - 20}
                        width={40}
                        height={40}
                        rx={4}
                        fill={info?.color || '#64748b'}
                      />
                      <rect
                        x={x - 20}
                        y={y2 - 20}
                        width={40}
                        height={40}
                        rx={4}
                        fill={info?.color || '#64748b'}
                      />
                      <text
                        x={x}
                        y={(y1 + y2) / 2}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        className="fill-white text-xs font-bold pointer-events-none"
                      >
                        {info?.symbol || op.gate.gate_name}
                      </text>
                    </>
                  )}
                </g>
              );
            } else if (qubits.length === 3) {
              // Three-qubit gate (Toffoli)
              const ys = qubits.map(q => 20 + q * CELL_HEIGHT + WIRE_Y_OFFSET);
              const minY = Math.min(...ys);
              const maxY = Math.max(...ys);

              return (
                <g
                  key={`op-${opIndex}`}
                  className="cursor-pointer hover:opacity-80"
                  onClick={() => removeOperation(opIndex)}
                >
                  <line
                    x1={x}
                    y1={minY}
                    x2={x}
                    y2={maxY}
                    stroke={info?.color || '#64748b'}
                    strokeWidth={2}
                  />
                  {/* Control dots */}
                  <circle cx={x} cy={ys[0]} r={6} fill={info?.color || '#64748b'} />
                  <circle cx={x} cy={ys[1]} r={6} fill={info?.color || '#64748b'} />
                  {/* Target */}
                  <circle
                    cx={x}
                    cy={ys[2]}
                    r={16}
                    fill="none"
                    stroke={info?.color || '#64748b'}
                    strokeWidth={2}
                  />
                  <line
                    x1={x - 16}
                    y1={ys[2]}
                    x2={x + 16}
                    y2={ys[2]}
                    stroke={info?.color || '#64748b'}
                    strokeWidth={2}
                  />
                  <line
                    x1={x}
                    y1={ys[2] - 16}
                    x2={x}
                    y2={ys[2] + 16}
                    stroke={info?.color || '#64748b'}
                    strokeWidth={2}
                  />
                </g>
              );
            }
          } else if (op.type === 'measurement') {
            // Measurement
            const qubits = op.measurement?.qubits || op.qubits;
            return qubits.map((q, idx) => {
              const y = 20 + q * CELL_HEIGHT + WIRE_Y_OFFSET;
              return (
                <g
                  key={`op-${opIndex}-m-${idx}`}
                  className="cursor-pointer hover:opacity-80"
                  onClick={() => removeOperation(opIndex)}
                >
                  <rect
                    x={x - 20}
                    y={y - 20}
                    width={40}
                    height={40}
                    rx={4}
                    fill="#1e293b"
                    stroke="#64748b"
                    strokeWidth={2}
                  />
                  {/* Meter symbol */}
                  <path
                    d={`M ${x - 10} ${y + 5} Q ${x} ${y - 15} ${x + 10} ${y + 5}`}
                    fill="none"
                    stroke="#64748b"
                    strokeWidth={2}
                  />
                  <line
                    x1={x}
                    y1={y - 5}
                    x2={x + 8}
                    y2={y - 12}
                    stroke="#64748b"
                    strokeWidth={2}
                  />
                </g>
              );
            });
          } else if (op.type === 'barrier') {
            // Barrier
            const qubits = op.qubits;
            const minQ = Math.min(...qubits);
            const maxQ = Math.max(...qubits);
            const y1 = 20 + minQ * CELL_HEIGHT + WIRE_Y_OFFSET - 25;
            const y2 = 20 + maxQ * CELL_HEIGHT + WIRE_Y_OFFSET + 25;

            return (
              <g
                key={`op-${opIndex}`}
                className="cursor-pointer hover:opacity-80"
                onClick={() => removeOperation(opIndex)}
              >
                <line
                  x1={x}
                  y1={y1}
                  x2={x}
                  y2={y2}
                  stroke="#64748b"
                  strokeWidth={2}
                  strokeDasharray="5,5"
                />
              </g>
            );
          }

          return null;
        })}

        {/* Empty slot indicator */}
        {operations.length === 0 && (
          <text
            x={circuitWidth / 2 + 60}
            y={(nQubits * CELL_HEIGHT) / 2 + 20}
            textAnchor="middle"
            dominantBaseline="middle"
            className="fill-slate-500 text-sm"
          >
            Add gates from the palette
          </text>
        )}
      </svg>
    </div>
  );
}
