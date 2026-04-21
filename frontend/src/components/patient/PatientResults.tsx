import { GaugeChart } from '../charts/GaugeChart';
import { RiskBadge } from '../common/RiskBadge';
import { FactorsList } from './FactorsList';
import type { PredictionOutput } from '../../api/types';
import type { DemoFactor } from '../../lib/demo';
import { pl } from '../../i18n/pl';

interface PatientResultsProps {
  prediction: PredictionOutput;
  factors: DemoFactor[];
  isDemo: boolean;
}

export function PatientResults({ prediction, factors, isDemo }: PatientResultsProps) {
  return (
    <div>
      {isDemo && (
        <div className="mb-4 rounded-lg bg-yellow-900/20 p-3 text-sm text-yellow-300">
          Tryb demo - predykcje symulowane
        </div>
      )}

      <div className="grid gap-6 md:grid-cols-[1fr_2fr]">
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-200">{pl.results.title}</h3>
          <GaugeChart probability={prediction.probability} title={pl.results.riskTitle} />
          <RiskBadge level={prediction.risk_level} />
        </div>

        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-200">{pl.results.keyFactors}</h3>
          <FactorsList factors={factors} />
        </div>
      </div>
    </div>
  );
}
