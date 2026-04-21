import { useEffect } from 'react';
import { useExplainShap } from '../../hooks/useApi';
import { WaterfallChart } from '../charts/WaterfallChart';
import { pl } from '../../i18n/pl';
import type { PatientInput } from '../../api/types';
import type { DemoFactor } from '../../lib/demo';

interface ShapTabProps {
  patient: PatientInput;
  factors: DemoFactor[];
}

export function ShapTab({ patient, factors }: ShapTabProps) {
  const mutation = useExplainShap();

  useEffect(() => {
    mutation.mutate({ patient, method: 'shap', num_features: 10 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patient]);

  const chartFactors = mutation.data
    ? [...mutation.data.risk_factors, ...mutation.data.protective_factors].map((f) => ({
        feature: f.feature,
        contribution: f.contribution,
      }))
    : factors;

  return (
    <div>
      <h3 className="mb-2 text-lg font-semibold text-gray-200">{pl.xai.shapTitle}</h3>
      <p className="mb-4 text-sm text-gray-400">{pl.xai.shapDesc}</p>

      {mutation.isPending && (
        <div className="flex h-64 items-center justify-center text-gray-400">Ładowanie wyjaśnienia SHAP...</div>
      )}

      <WaterfallChart factors={chartFactors} title="Wpływ czynników (SHAP)" />

      {mutation.isError && (
        <p className="mt-2 text-xs text-yellow-400">Użyto danych demo (API SHAP niedostępne)</p>
      )}
    </div>
  );
}
