import { useMemo } from 'react';
import { useCircuitStore } from '../../stores/circuitStore';

export function StateVector() {
  const { nQubits, probabilities } = useCircuitStore();

  // Calculate amplitudes from probabilities (assuming pure state)
  // Note: This loses phase information - for full amplitudes, need API call
  const amplitudes = useMemo(() => {
    return probabilities.map((p, i) => ({
      index: i,
      label: i.toString(2).padStart(nQubits, '0'),
      magnitude: Math.sqrt(p),
      probability: p,
      // Phase would need to come from actual state
      phase: 0,
    }));
  }, [probabilities, nQubits]);

  const nonZeroAmps = amplitudes.filter(a => a.magnitude > 0.001);
  const maxMag = Math.max(...amplitudes.map(a => a.magnitude), 0.01);

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-medium text-slate-300">
        State Vector Amplitudes
      </h3>

      {nonZeroAmps.length === 0 ? (
        <div className="flex items-center justify-center h-32 text-slate-400">
          |0...0⟩
        </div>
      ) : (
        <div className="space-y-3">
          {nonZeroAmps.slice(0, 16).map(({ label, magnitude, probability, phase }) => (
            <div key={label} className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm text-slate-400 w-16">
                  |{label}⟩
                </span>
                <div className="flex-1 flex gap-1">
                  {/* Amplitude bar (real part visualization) */}
                  <div className="flex-1 h-6 bg-slate-700/50 rounded-l overflow-hidden">
                    <div
                      className="h-full bg-blue-500 transition-all duration-300"
                      style={{ width: `${(magnitude / maxMag) * 100}%` }}
                    />
                  </div>
                  {/* Phase indicator */}
                  <div
                    className="w-6 h-6 rounded-r flex items-center justify-center text-xs"
                    style={{
                      background: `hsl(${(phase * 180) / Math.PI}, 70%, 50%)`,
                    }}
                    title={`Phase: ${((phase * 180) / Math.PI).toFixed(0)}°`}
                  >
                    ∠
                  </div>
                </div>
                <span className="w-20 text-right font-mono text-xs text-slate-300">
                  {magnitude.toFixed(3)}
                </span>
              </div>

              {/* Detailed values */}
              <div className="ml-[72px] flex gap-4 text-xs text-slate-500">
                <span>|α|² = {probability.toFixed(4)}</span>
                <span>φ = {((phase * 180) / Math.PI).toFixed(1)}°</span>
              </div>
            </div>
          ))}

          {nonZeroAmps.length > 16 && (
            <div className="text-xs text-slate-500 text-center">
              ... and {nonZeroAmps.length - 16} more states
            </div>
          )}
        </div>
      )}

      {/* State summary */}
      <div className="pt-4 border-t border-slate-700">
        <div className="text-xs text-slate-400">
          <div className="flex justify-between">
            <span>Dimension:</span>
            <span>2^{nQubits} = {Math.pow(2, nQubits)}</span>
          </div>
          <div className="flex justify-between">
            <span>Non-zero amplitudes:</span>
            <span>{nonZeroAmps.length}</span>
          </div>
          <div className="flex justify-between">
            <span>Normalization:</span>
            <span>
              {probabilities.reduce((sum, p) => sum + p, 0).toFixed(6)}
            </span>
          </div>
        </div>
      </div>

      {/* State notation */}
      {nonZeroAmps.length > 0 && nonZeroAmps.length <= 4 && (
        <div className="pt-2 text-sm font-mono text-center text-slate-300">
          |ψ⟩ ={' '}
          {nonZeroAmps.map((a, i) => (
            <span key={a.label}>
              {i > 0 && ' + '}
              {a.magnitude.toFixed(2)}|{a.label}⟩
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
