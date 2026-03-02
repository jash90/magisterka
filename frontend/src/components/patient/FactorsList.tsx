import type { DemoFactor } from '../../lib/demo';

export function FactorsList({ factors }: { factors: DemoFactor[] }) {
  const sorted = [...factors].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  const top5 = sorted.slice(0, 5);

  return (
    <div className="space-y-2">
      {top5.map((factor) => {
        const isRisk = factor.contribution > 0;
        return (
          <div
            key={factor.feature}
            className={`flex items-center gap-2 rounded-lg p-2 text-sm ${
              isRisk ? 'bg-red-900/20 text-red-300' : 'bg-green-900/20 text-green-300'
            }`}
          >
            <span className="text-lg">{isRisk ? '↑' : '↓'}</span>
            <span className="flex-1 font-medium">{factor.feature}</span>
            <span className="text-xs opacity-70">
              {isRisk ? 'zwiększa' : 'zmniejsza'} ryzyko ({factor.contribution >= 0 ? '+' : ''}
              {factor.contribution.toFixed(3)})
            </span>
          </div>
        );
      })}
    </div>
  );
}
