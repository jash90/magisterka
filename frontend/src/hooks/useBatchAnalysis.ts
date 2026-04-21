import { useState, useCallback } from 'react';
import type { PatientInput, BatchResultRow, RiskLevel } from '../api/types';
import type { ParsedPatient } from '../lib/fileParser';
import { parseFile } from '../lib/fileParser';
import { predictBatch } from '../api/endpoints';
import { getDemoPrediction, getDemoExplanation } from '../lib/demo';
import { RISK_LEVEL_PL } from '../lib/columnMapping';

export function useBatchAnalysis() {
  const [results, setResults] = useState<BatchResultRow[] | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const [mode, setMode] = useState<string>('');

  const processFile = useCallback(async (file: File) => {
    setIsProcessing(true);
    setProgress(0);
    setError(null);
    setFileName(file.name);

    try {
      const patients = await parseFile(file);
      if (patients.length === 0) {
        setError('Plik jest pusty');
        setIsProcessing(false);
        return;
      }

      // Try batch API first
      const patientInputs: PatientInput[] = patients.map(({ patient_id: _, ...rest }) => rest);
      const CHUNK_SIZE = 1000;
      const allResults: BatchResultRow[] = [];

      let apiSuccess = false;

      for (let i = 0; i < patientInputs.length; i += CHUNK_SIZE) {
        const chunk = patientInputs.slice(i, i + CHUNK_SIZE);
        const chunkPatients = patients.slice(i, i + CHUNK_SIZE);

        try {
          const batchResult = await predictBatch({
            patients: chunk,
            include_risk_factors: true,
            top_n_factors: 3,
          });

          apiSuccess = true;
          setMode(batchResult.mode);

          for (let j = 0; j < batchResult.results.length; j++) {
            const item = batchResult.results[j];
            const pred = item.prediction;
            const topFactors = item.top_risk_factors?.map((f) => f.feature).slice(0, 3).join(', ') ?? '';

            allResults.push({
              patient_id: chunkPatients[j].patient_id,
              wiek_rozpoznania: chunk[j].wiek_rozpoznania ?? 0,
              liczba_narzadow: chunk[j].liczba_zajetych_narzadow,
              probability: pred.probability,
              probability_pct: `${(pred.probability * 100).toFixed(1)}%`,
              risk_level: pred.risk_level,
              risk_level_pl: RISK_LEVEL_PL[pred.risk_level] ?? 'Niskie',
              prediction: pred.prediction,
              top_factors: topFactors,
              processing_mode: batchResult.mode,
            });
          }
        } catch {
          // Fallback to demo mode
          for (let j = 0; j < chunk.length; j++) {
            const prediction = getDemoPrediction(chunk[j]);
            const explanation = getDemoExplanation(chunk[j]);
            const allFactors = [...explanation.risk_factors, ...explanation.protective_factors]
              .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

            allResults.push({
              patient_id: chunkPatients[j].patient_id,
              wiek_rozpoznania: chunk[j].wiek_rozpoznania ?? 0,
              liczba_narzadow: chunk[j].liczba_zajetych_narzadow,
              probability: prediction.probability,
              probability_pct: `${(prediction.probability * 100).toFixed(1)}%`,
              risk_level: prediction.risk_level as RiskLevel,
              risk_level_pl: RISK_LEVEL_PL[prediction.risk_level] ?? 'Niskie',
              prediction: prediction.prediction,
              top_factors: allFactors.slice(0, 3).map((f) => f.feature).join(', '),
              processing_mode: 'demo',
            });
          }
          if (!apiSuccess) setMode('demo');
        }

        setProgress((i + chunk.length) / patientInputs.length);
      }

      setProgress(1);
      setResults(allResults);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Błąd przetwarzania pliku');
    } finally {
      setIsProcessing(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResults(null);
    setProgress(0);
    setError(null);
    setFileName('');
    setMode('');
  }, []);

  return { results, isProcessing, progress, error, fileName, mode, processFile, reset };
}
