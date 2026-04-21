import type { BatchResultRow } from '../../api/types';
import { pl } from '../../i18n/pl';

export function BatchSummary({ results }: { results: BatchResultRow[] }) {
  const total = results.length;
  const low = results.filter((r) => r.risk_level === 'low').length;
  const moderate = results.filter((r) => r.risk_level === 'moderate').length;
  const high = results.filter((r) => r.risk_level === 'high').length;

  const cards = [
    { label: pl.batch.totalPatients, value: total, color: 'text-blue-400' },
    { label: pl.batch.lowRisk, value: low, pct: ((low / total) * 100).toFixed(1), color: 'text-green-400' },
    { label: pl.batch.moderateRisk, value: moderate, pct: ((moderate / total) * 100).toFixed(1), color: 'text-yellow-400' },
    { label: pl.batch.highRisk, value: high, pct: ((high / total) * 100).toFixed(1), color: 'text-red-400' },
  ];

  return (
    <div>
      <h2 className="mb-4 text-xl font-bold text-gray-200">{pl.batch.summary}</h2>
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {cards.map((card) => (
          <div key={card.label} className="rounded-lg bg-gray-800 p-4 text-center">
            <p className={`text-3xl font-bold ${card.color}`}>{card.value}</p>
            <p className="mt-1 text-sm text-gray-400">{card.label}</p>
            {card.pct && <p className="text-xs text-gray-500">{card.pct}%</p>}
          </div>
        ))}
      </div>
    </div>
  );
}
