import { useEffect, useState } from 'react';
import { useExplainDalex } from '../../hooks/useApi';
import { HorizontalBarChart } from '../charts/HorizontalBarChart';
import { WaterfallChart } from '../charts/WaterfallChart';
import { ModelSelector } from './ModelSelector';
import type { PatientInput } from '../../api/types';
import type { DemoFactor } from '../../lib/demo';

interface DalexTabProps {
  patient: PatientInput;
  factors: DemoFactor[];
}

export function DalexTab({ patient, factors }: DalexTabProps) {
  const mutation = useExplainDalex();
  const [modelKey, setModelKey] = useState('xgboost');

  useEffect(() => {
    mutation.mutate({ patient, method: 'dalex', num_features: 10, model_key: modelKey });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patient, modelKey]);

  const bdFactors = mutation.data
    ? [
        ...mutation.data.risk_factors.map((f) => ({ feature: f.feature, contribution: f.contribution })),
        ...mutation.data.protective_factors.map((f) => ({ feature: f.feature, contribution: f.contribution })),
      ]
    : factors;

  const vi = mutation.data?.variable_importance;

  return (
    <div>
      <h3 className="mb-2 text-lg font-semibold text-gray-200">DALEX - Break Down Analysis</h3>
      <p className="mb-4 text-sm text-gray-400">
        DALEX (Descriptive mAchine Learning EXplanations) generuje wyjaśnienia poprzez break-down 
        — rozkłada predykcję na wkłady poszczególnych cech, pokazując jak każda z nich wpłynęła na wynik.
        Dodatkowo wyświetlana jest globalna ważność cech obliczona metodą permutacji.
      </p>

      <ModelSelector value={modelKey} onChange={setModelKey} />

      {mutation.isPending && (
        <div className="flex h-64 items-center justify-center text-gray-400">Ładowanie wyjaśnienia DALEX...</div>
      )}

      {mutation.data && (
        <div className="space-y-6">
          {/* Break Down */}
          <div>
            <h4 className="mb-3 text-sm font-semibold text-gray-300">Break Down — wkład cech do predykcji</h4>
            <p className="mb-2 text-xs text-gray-500">
              Intercept: {mutation.data.intercept.toFixed(4)} | Predykcja: {mutation.data.prediction.toFixed(4)}
            </p>
            <WaterfallChart factors={bdFactors} title="" />
          </div>

          {/* Variable Importance */}
          {vi && Object.keys(vi).length > 0 && (
            <div>
              <h4 className="mb-3 text-sm font-semibold text-gray-300">Permutation Variable Importance</h4>
              <p className="mb-2 text-xs text-gray-500">
                Spadek AUC po permutacji cechy — im wyższa wartość, tym ważniejsza cecha.
              </p>
              <HorizontalBarChart
                factors={Object.entries(vi).map(([feature, contribution]) => ({ feature, contribution }))}
                title=""
              />
            </div>
          )}
        </div>
      )}

      {mutation.isError && (
        <p className="mt-2 text-xs text-yellow-400">DALEX niedostępny (wymaga biblioteki dalex)</p>
      )}
    </div>
  );
}
