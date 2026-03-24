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

const FEATURE_LABELS: Record<string, string> = {
  Wiek_rozpoznania: 'Wiek rozpoznania',
  Opoznienie_Rozpoznia: 'Opóźnienie diagnozy',
  Manifestacja_Miesno_Szkiel: 'Mięśniowo-szkieletowy',
  Manifestacja_Skora: 'Skóra',
  Manifestacja_Wzrok: 'Wzrok',
  Manifestacja_Sercowo_Naczyniowy: 'Serce/naczynia',
  Manifestacja_Sercowo_Naczyniowy: 'Serce/naczynia',
  Manifestacja_Pokarmowy: 'Układ pokarmowy',
  Manifestacja_Nerki: 'Nerki',
  Manifestacja_Moczowo_Plciowy: 'Moczowo-płciowy',
  Manifestacja_Zajecie_CSN: 'CSN (mózg)',
  Manifestacja_Neurologiczny: 'Neurologiczny',
  Liczba_Zajetych_Narzadow: 'Liczba zajętych narządów',
  Zaostrz_Wymagajace_Hospital: 'Hospitalizacja',
  Zaostrz_Wymagajace_OIT: 'OIT',
  Kreatynina: 'Kreatynina',
  Pulsy: 'Pulsy sterydowe',
  Czas_Sterydow: 'Czas sterydów',
  Plazmaferezy: 'Plazmaferezy',
  Eozynofilia_Krwi_Obwodowej_Wartosc: 'Eozynofilia',
  Biopsja_Wynik: 'Biopsja',
};

function label(feature: string): string {
  return FEATURE_LABELS[feature] || feature.replace(/_/g, ' ');
}

export function ShapTab({ patient, factors }: ShapTabProps) {
  const mutation = useExplainShap();

  useEffect(() => {
    mutation.mutate({ patient, method: 'shap', num_features: 10 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patient]);

  const chartFactors = mutation.data
    ? [...mutation.data.risk_factors, ...mutation.data.protective_factors].map((f) => ({
        feature: label(f.feature),
        contribution: f.contribution,
      }))
    : factors.map((f) => ({ feature: label(f.feature), contribution: f.contribution }));

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
