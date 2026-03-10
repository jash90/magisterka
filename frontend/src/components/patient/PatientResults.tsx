import { GaugeChart } from '../charts/GaugeChart';
import { RiskBadge } from '../common/RiskBadge';
import { FactorsList } from './FactorsList';
import type { PredictionOutput, ModelPrediction } from '../../api/types';
import type { DemoFactor } from '../../lib/demo';
import { pl } from '../../i18n/pl';

interface PatientResultsProps {
  prediction: PredictionOutput;
  factors: DemoFactor[];
  models?: ModelPrediction[];
}

function ModelGauge({ model }: { model: ModelPrediction }) {
  return (
    <div className="rounded-xl border border-gray-700/60 bg-gray-800/30 p-4 text-center">
      <p className="mb-2 text-sm font-semibold text-gray-300">{model.model_name}</p>
      <GaugeChart probability={model.probability} title="" />
      <div className="mt-2 flex justify-center">
        <RiskBadge level={model.risk_level} />
      </div>
      <p className="mt-1 text-xs text-gray-500">{(model.probability * 100).toFixed(1)}%</p>
    </div>
  );
}

export function PatientResults({ prediction, factors, models }: PatientResultsProps) {
  return (
    <div className="space-y-8">
      {/* All model predictions */}
      {models && models.length > 0 ? (
        <div>
          <h3 className="mb-4 text-lg font-semibold text-blue-300">Predykcja modeli</h3>
          <div className={`grid gap-4 ${models.length === 3 ? 'md:grid-cols-3' : models.length === 2 ? 'md:grid-cols-2' : ''}`}>
            {models.map((m) => (
              <ModelGauge key={m.model_name} model={m} />
            ))}
          </div>
          {/* Ensemble summary */}
          <div className="mt-4 rounded-xl border border-blue-500/30 bg-blue-900/10 p-4 text-center">
            <p className="text-sm font-medium text-gray-400">Ensemble (średnia)</p>
            <p className="mt-1 text-2xl font-bold text-blue-300">{(prediction.probability * 100).toFixed(1)}%</p>
            <div className="mt-2 flex justify-center">
              <RiskBadge level={prediction.risk_level} />
            </div>
          </div>
        </div>
      ) : (
        /* Fallback: single model */
        <div className="grid gap-6 md:grid-cols-[1fr_2fr]">
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-200">{pl.results.title}</h3>
            <GaugeChart probability={prediction.probability} title={pl.results.riskTitle} />
            <RiskBadge level={prediction.risk_level} />
          </div>
        </div>
      )}

      {/* Key factors */}
      {factors.length > 0 && (
        <div>
          <h3 className="mb-4 text-lg font-semibold text-gray-200">{pl.results.keyFactors}</h3>
          <FactorsList factors={factors} />
        </div>
      )}
    </div>
  );
}
