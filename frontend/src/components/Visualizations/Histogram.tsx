import { useMemo } from 'react';
import { useCircuitStore } from '../../stores/circuitStore';

export function Histogram() {
  const { result, nQubits, probabilities } = useCircuitStore();

  // Use result counts if available, otherwise use state probabilities
  const data = useMemo(() => {
    if (result) {
      // Sort by count descending, take top 16
      const entries = Object.entries(result.counts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 16);

      const total = result.shots;
      return entries.map(([label, count]) => ({
        label,
        value: count / total,
        count,
      }));
    }

    // Use state probabilities
    const entries: { label: string; value: number; count: number }[] = [];
    for (let i = 0; i < probabilities.length; i++) {
      if (probabilities[i] > 0.001) {
        entries.push({
          label: i.toString(2).padStart(nQubits, '0'),
          value: probabilities[i],
          count: 0,
        });
      }
    }

    return entries.sort((a, b) => b.value - a.value).slice(0, 16);
  }, [result, probabilities, nQubits]);

  const maxValue = Math.max(...data.map(d => d.value), 0.01);

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-400">
        No data to display. Run the circuit first.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-medium text-slate-300">
        {result ? 'Measurement Results' : 'Probability Distribution'}
      </h3>

      <div className="space-y-2">
        {data.map(({ label, value, count }) => (
          <div key={label} className="group">
            <div className="flex items-center gap-2 text-xs">
              <span className="font-mono text-slate-400 w-20">|{label}⟩</span>
              <div className="flex-1 h-6 bg-slate-700/50 rounded overflow-hidden">
                <div
                  className="h-full bg-quantum-500 transition-all duration-300"
                  style={{ width: `${(value / maxValue) * 100}%` }}
                />
              </div>
              <span className="w-16 text-right font-mono text-slate-300">
                {(value * 100).toFixed(1)}%
              </span>
            </div>
            {result && (
              <div className="text-xs text-slate-500 ml-[88px]">
                {count} / {result.shots}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="pt-4 border-t border-slate-700">
        <div className="grid grid-cols-2 gap-2 text-xs text-slate-400">
          <div>States: {data.length}</div>
          <div>
            Max: {(maxValue * 100).toFixed(1)}%
          </div>
        </div>
      </div>
    </div>
  );
}
