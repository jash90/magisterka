import { useEffect } from 'react';
import { useExplainLime } from '../../hooks/useApi';
import { HorizontalBarChart } from '../charts/HorizontalBarChart';
import { pl } from '../../i18n/pl';
import type { PatientInput } from '../../api/types';
import type { DemoFactor } from '../../lib/demo';

interface LimeTabProps {
  patient: PatientInput;
  factors: DemoFactor[];
}

export function LimeTab({ patient, factors }: LimeTabProps) {
  const mutation = useExplainLime();

  useEffect(() => {
    mutation.mutate({ patient, method: 'lime', num_features: 10 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patient]);

  const chartFactors = mutation.data
    ? [...(mutation.data.risk_factors ?? []), ...(mutation.data.protective_factors ?? [])].map((f) => ({
        feature: String(f.feature ?? f.name ?? ''),
        contribution: Number(f.contribution ?? f.weight ?? 0),
      }))
    : factors;

  return (
    <div>
      <h3 className="mb-2 text-lg font-semibold text-gray-200">{pl.xai.limeTitle}</h3>
      <p className="mb-4 text-sm text-gray-400">{pl.xai.limeDesc}</p>

      {mutation.isPending && (
        <div className="flex h-64 items-center justify-center text-gray-400">Ładowanie wyjaśnienia LIME...</div>
      )}

      <HorizontalBarChart factors={chartFactors} title="Ważność czynników (LIME)" />

      {mutation.isError && (
        <p className="mt-2 text-xs text-yellow-400">Użyto danych demo (API LIME niedostępne)</p>
      )}
    </div>
  );
}
