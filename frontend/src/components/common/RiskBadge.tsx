import type { RiskLevel } from '../../api/types';
import { pl } from '../../i18n/pl';

const STYLES: Record<RiskLevel, { bg: string; border: string; text: string }> = {
  low: { bg: 'bg-green-900/30', border: 'border-green-500', text: 'text-green-300' },
  moderate: { bg: 'bg-yellow-900/30', border: 'border-yellow-500', text: 'text-yellow-300' },
  high: { bg: 'bg-red-900/30', border: 'border-red-500', text: 'text-red-300' },
};

const DESCS: Record<RiskLevel, string> = {
  low: pl.risk.lowDesc,
  moderate: pl.risk.moderateDesc,
  high: pl.risk.highDesc,
};

const LABELS: Record<RiskLevel, string> = {
  low: pl.risk.low,
  moderate: pl.risk.moderate,
  high: pl.risk.high,
};

export function RiskBadge({ level }: { level: RiskLevel }) {
  const s = STYLES[level];
  return (
    <div className={`rounded-lg border-l-4 p-4 ${s.bg} ${s.border}`}>
      <strong className={s.text}>{LABELS[level]}</strong>
      <br />
      <span className="text-sm text-gray-300">{DESCS[level]}</span>
    </div>
  );
}

export function RiskPill({ level }: { level: RiskLevel }) {
  const colors: Record<RiskLevel, string> = {
    low: 'bg-green-600 text-white',
    moderate: 'bg-yellow-500 text-gray-900',
    high: 'bg-red-600 text-white',
  };
  return (
    <span className={`inline-block rounded-full px-3 py-0.5 text-xs font-medium ${colors[level]}`}>
      {LABELS[level]}
    </span>
  );
}
