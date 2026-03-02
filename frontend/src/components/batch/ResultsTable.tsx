import { useState, useMemo } from 'react';
import type { BatchResultRow, RiskLevel } from '../../api/types';
import { RiskPill } from '../common/RiskBadge';
import { pl } from '../../i18n/pl';

export function ResultsTable({ results }: { results: BatchResultRow[] }) {
  const [riskFilter, setRiskFilter] = useState<RiskLevel[]>(['low', 'moderate', 'high']);
  const [sortBy, setSortBy] = useState<'probability' | 'wiek_rozpoznania' | 'patient_id'>('probability');

  const filtered = useMemo(() => {
    const f = results.filter((r) => riskFilter.includes(r.risk_level));
    return f.sort((a, b) => {
      if (sortBy === 'patient_id') return a.patient_id.localeCompare(b.patient_id);
      if (sortBy === 'probability') return b.probability - a.probability;
      return b.wiek_rozpoznania - a.wiek_rozpoznania;
    });
  }, [results, riskFilter, sortBy]);

  const toggleFilter = (level: RiskLevel) => {
    setRiskFilter((prev) =>
      prev.includes(level) ? prev.filter((l) => l !== level) : [...prev, level],
    );
  };

  return (
    <div>
      <h2 className="mb-4 text-xl font-bold text-gray-200">{pl.batch.detailedResults}</h2>

      <div className="mb-4 flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-400">{pl.batch.filterByRisk}:</span>
          {(['low', 'moderate', 'high'] as RiskLevel[]).map((level) => {
            const labels: Record<RiskLevel, string> = { low: 'Niskie', moderate: 'Umiarkowane', high: 'Wysokie' };
            const isActive = riskFilter.includes(level);
            return (
              <button
                key={level}
                onClick={() => toggleFilter(level)}
                className={`rounded-full px-3 py-1 text-xs font-medium transition ${
                  isActive ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-400'
                }`}
              >
                {labels[level]}
              </button>
            );
          })}
        </div>

        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-400">{pl.batch.sortBy}:</span>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
            className="rounded-md border border-gray-600 bg-gray-700 px-3 py-1 text-sm text-white"
          >
            <option value="probability">Prawdopodobienstwo</option>
            <option value="wiek_rozpoznania">Wiek rozpoznania</option>
            <option value="patient_id">ID pacjenta</option>
          </select>
        </div>
      </div>

      <div className="max-h-[400px] overflow-auto rounded-lg border border-gray-700">
        <table className="w-full text-left text-sm">
          <thead className="sticky top-0 bg-gray-800 text-xs uppercase text-gray-400">
            <tr>
              <th className="px-4 py-3">ID</th>
              <th className="px-4 py-3">Wiek rozp.</th>
              <th className="px-4 py-3">Narzady</th>
              <th className="px-4 py-3">Ryzyko</th>
              <th className="px-4 py-3">Poziom</th>
              <th className="px-4 py-3">Glowne czynniki</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-700">
            {filtered.map((r) => (
              <tr key={r.patient_id} className="text-gray-300 hover:bg-gray-800/50">
                <td className="px-4 py-2 font-medium">{r.patient_id}</td>
                <td className="px-4 py-2">{r.wiek_rozpoznania}</td>
                <td className="px-4 py-2">{r.liczba_narzadow}</td>
                <td className="px-4 py-2">{r.probability_pct}</td>
                <td className="px-4 py-2">
                  <RiskPill level={r.risk_level} />
                </td>
                <td className="px-4 py-2 text-xs text-gray-400">{r.top_factors}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-2 text-xs text-gray-500">
        Wyswietlono {filtered.length} z {results.length} pacjentow
      </p>
    </div>
  );
}
