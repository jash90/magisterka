import type { BatchResultRow } from '../../api/types';
import { RiskPieChart } from '../charts/PieChart';
import { ProbabilityHistogram } from '../charts/Histogram';
import { AgeRiskScatter } from '../charts/ScatterChart';
import { pl } from '../../i18n/pl';

export function BatchCharts({ results }: { results: BatchResultRow[] }) {
  return (
    <div>
      <h2 className="mb-4 text-xl font-bold text-gray-200">{pl.batch.charts}</h2>
      <div className="grid gap-6 md:grid-cols-2">
        <RiskPieChart results={results} />
        <ProbabilityHistogram results={results} />
      </div>
      <div className="mt-6">
        <h3 className="mb-2 text-lg font-semibold text-gray-300">{pl.batch.ageVsRisk}</h3>
        <AgeRiskScatter results={results} />
      </div>
    </div>
  );
}
