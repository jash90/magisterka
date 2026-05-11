import { useEffect } from 'react';
import { useExplainEbm } from '../../hooks/useApi';
import { HorizontalBarChart } from '../charts/HorizontalBarChart';
import { WaterfallChart } from '../charts/WaterfallChart';
import { RiskBadge } from '../common/RiskBadge';
import type { PatientInput } from '../../api/types';
import type { DemoFactor } from '../../lib/demo';

interface EbmTabProps {
  patient: PatientInput;
  factors: DemoFactor[];
}

export function EbmTab({ patient, factors }: EbmTabProps) {
  const mutation = useExplainEbm();

  useEffect(() => {
    mutation.mutate({ patient, method: 'ebm', num_features: 10 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patient]);

  const localFactors = mutation.data
    ? mutation.data.local_contributions.map((f) => ({ feature: f.feature, contribution: f.contribution }))
    : factors;

  const globalFactors = mutation.data
    ? Object.entries(mutation.data.global_importance).map(([feature, contribution]) => ({ feature, contribution }))
    : [];

  return (
    <div>
      <h3 className="mb-2 text-lg font-semibold text-gray-200">EBM - Explainable Boosting Machine</h3>
      <p className="mb-4 text-sm text-gray-400">
        EBM to inherentnie interpretowalny model bazujący na Generalized Additive Models (GAM).
        Łączy dokładność gradient boostingu z pełną transparentnością — każda cecha ma swoją 
        wyraźną funkcję kształtu. EBM jest wytrenowany niezależnie na tych samych danych
        (719 pacjentów, 20 cech) — jego wynik to predykcja własnego modelu, nie XGBoost/RF/LightGBM.
      </p>

      <div className="mb-4 flex items-center gap-2 rounded border border-gray-600 bg-gray-800/30 px-3 py-2">
        <span className="text-xs text-gray-500">Model:</span>
        <span className="rounded bg-purple-600/30 px-2 py-0.5 text-xs font-medium text-purple-300">
          EBM (własny model GAM)
        </span>
      </div>

      {mutation.isPending && (
        <div className="flex h-64 items-center justify-center text-gray-400">Trenowanie modelu EBM (może potrwać kilkanaście sekund)...</div>
      )}

      {mutation.data && (
        <div className="space-y-6">
          {/* Prediction summary */}
          <div className="flex items-center gap-4 rounded-lg border border-gray-600 bg-gray-800/50 p-4">
            <div>
              <p className="text-sm text-gray-400">Predykcja EBM</p>
              <p className="text-2xl font-bold text-white">{(mutation.data.probability * 100).toFixed(1)}%</p>
            </div>
            <RiskBadge level={mutation.data.risk_level} />
            {mutation.data.interactions.length > 0 && (
              <div className="ml-auto text-xs text-gray-500">
                Wykryte interakcje: {mutation.data.interactions.length}
              </div>
            )}
          </div>

          {/* Local contributions */}
          <div>
            <h4 className="mb-3 text-sm font-semibold text-gray-300">Lokalny wkład cech</h4>
            <WaterfallChart factors={localFactors} title="" />
          </div>

          {/* Global importance */}
          {globalFactors.length > 0 && (
            <div>
              <h4 className="mb-3 text-sm font-semibold text-gray-300">Globalna ważność cech (EBM)</h4>
              <HorizontalBarChart factors={globalFactors} title="" />
            </div>
          )}

          {/* Interactions */}
          {mutation.data.interactions.length > 0 && (
            <div>
              <h4 className="mb-2 text-sm font-semibold text-gray-300">Wykryte interakcje między cechami</h4>
              <div className="flex flex-wrap gap-2">
                {mutation.data.interactions.map((inter) => (
                  <span key={inter} className="rounded-full bg-gray-700 px-3 py-1 text-xs text-gray-300">{inter}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {mutation.isError && (
        <p className="mt-2 text-xs text-yellow-400">EBM niedostępny (wymaga biblioteki interpret)</p>
      )}
    </div>
  );
}
