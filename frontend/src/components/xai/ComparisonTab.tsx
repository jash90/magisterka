import { useEffect, useState } from 'react';
import { useExplainComparison } from '../../hooks/useApi';
import { ModelSelector } from './ModelSelector';
import { pl } from '../../i18n/pl';
import type { PatientInput } from '../../api/types';
import type { DemoFactor } from '../../lib/demo';

interface ComparisonTabProps {
  patient: PatientInput;
  factors: DemoFactor[];
}

export function ComparisonTab({ patient, factors }: ComparisonTabProps) {
  const mutation = useExplainComparison();
  const [modelKey, setModelKey] = useState('xgboost');

  useEffect(() => {
    mutation.mutate({ patient, method: 'shap', num_features: 10, model_key: modelKey });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patient, modelKey]);

  const sorted = [...factors].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  const shapRanking = mutation.data?.individual_rankings?.SHAP ?? sorted.slice(0, 5).map((f) => f.feature);
  const limeRanking = mutation.data?.individual_rankings?.LIME ?? sorted.slice(0, 5).map((f) => f.feature);

  return (
    <div>
      <h3 className="mb-4 text-lg font-semibold text-gray-200">{pl.xai.compTitle}</h3>

      <ModelSelector value={modelKey} onChange={setModelKey} />

      <div className="grid gap-6 md:grid-cols-2">
        <div>
          <h4 className="mb-2 font-medium text-gray-300">Ranking SHAP:</h4>
          <ol className="list-inside list-decimal space-y-1 text-sm text-gray-400">
            {(Array.isArray(shapRanking) ? shapRanking : []).slice(0, 5).map((f, i) => (
              <li key={i}>{f}</li>
            ))}
          </ol>
        </div>

        <div>
          <h4 className="mb-2 font-medium text-gray-300">Ranking LIME:</h4>
          <ol className="list-inside list-decimal space-y-1 text-sm text-gray-400">
            {(Array.isArray(limeRanking) ? limeRanking : []).slice(0, 5).map((f, i) => (
              <li key={i}>{f}</li>
            ))}
          </ol>
        </div>
      </div>

      {mutation.data && (
        <div className="mt-4 rounded-lg bg-blue-900/20 p-3 text-sm text-blue-300">
          Zgodność rankingów: {(mutation.data.ranking_agreement * 100).toFixed(0)}%
          {mutation.data.common_top_features?.length > 0 && (
            <> | Wspólne cechy: {mutation.data.common_top_features.join(', ')}</>
          )}
        </div>
      )}

      <div className="mt-4 rounded-lg bg-blue-900/10 p-3 text-sm text-blue-400">
        {pl.xai.compInfo}
      </div>

      {mutation.isPending && <p className="mt-2 text-xs text-gray-500">Ładowanie porównania...</p>}
      {mutation.isError && <p className="mt-2 text-xs text-yellow-400">Błąd porównania</p>}
    </div>
  );
}
