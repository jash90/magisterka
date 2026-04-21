import { useState, useCallback } from 'react';
import { Sidebar } from './Sidebar';
import { Footer } from './Footer';
import { WelcomePage } from '../welcome/WelcomePage';
import { BatchWelcome } from '../welcome/BatchWelcome';
import { PatientResults } from '../patient/PatientResults';
import { XaiTabs } from '../xai/XaiTabs';
import { BatchSummary } from '../batch/BatchSummary';
import { BatchCharts } from '../batch/BatchCharts';
import { ResultsTable } from '../batch/ResultsTable';
import { ExportButtons } from '../batch/ExportButtons';
import { BatchProgress } from '../batch/BatchProgress';
import { LoadingSkeleton } from '../common/LoadingSkeleton';
import { DemoModeIndicator } from '../common/DemoModeIndicator';
import { usePredict } from '../../hooks/useApi';
import { useBatchAnalysis } from '../../hooks/useBatchAnalysis';
import { explainShap } from '../../api/endpoints';
import { getDemoPrediction, getDemoExplanation, type DemoFactor } from '../../lib/demo';
import { pl } from '../../i18n/pl';
import type { PatientInput, PredictionOutput } from '../../api/types';

export function AppLayout() {
  const [mode, setMode] = useState<'single' | 'batch'>('single');
  const [prediction, setPrediction] = useState<PredictionOutput | null>(null);
  const [factors, setFactors] = useState<DemoFactor[]>([]);
  const [currentPatient, setCurrentPatient] = useState<PatientInput | null>(null);
  const [isDemo, setIsDemo] = useState(false);

  const predictMutation = usePredict();
  const batch = useBatchAnalysis();

  const handleAnalyze = useCallback(
    async (patient: PatientInput) => {
      setCurrentPatient(patient);
      try {
        const result = await predictMutation.mutateAsync(patient);
        setPrediction(result);
        setIsDemo(false);

        // Fetch real SHAP explanation from API for initial factors
        try {
          const shapData = await explainShap({ patient, method: 'shap', num_features: 10 });
          const apiFactors = [...shapData.risk_factors, ...shapData.protective_factors].map((f) => ({
            feature: f.feature,
            contribution: f.contribution,
          }));
          setFactors(apiFactors);
        } catch {
          // SHAP unavailable — fall back to demo factors
          const expl = getDemoExplanation(patient);
          setFactors([...expl.risk_factors, ...expl.protective_factors]);
        }
      } catch {
        // API unavailable — full demo mode
        const demoPred = getDemoPrediction(patient);
        setPrediction(demoPred);
        setIsDemo(true);
        const expl = getDemoExplanation(patient);
        setFactors([...expl.risk_factors, ...expl.protective_factors]);
      }
    },
    [predictMutation],
  );

  const handleFileSelect = useCallback(
    (file: File) => {
      batch.processFile(file);
    },
    [batch],
  );

  const handleModeChange = useCallback(
    (newMode: 'single' | 'batch') => {
      setMode(newMode);
      if (newMode === 'single') {
        batch.reset();
      } else {
        setPrediction(null);
        setFactors([]);
        setCurrentPatient(null);
      }
    },
    [batch],
  );

  const isAnalyzing = predictMutation.isPending || batch.isProcessing;

  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b border-gray-700 bg-gray-900/80 px-6 py-4">
        <h1 className="text-center text-2xl font-bold text-blue-300 md:text-3xl">{pl.app.title}</h1>
        <DemoModeIndicator />
      </header>

      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          mode={mode}
          onModeChange={handleModeChange}
          onAnalyze={handleAnalyze}
          onFileSelect={handleFileSelect}
          isAnalyzing={isAnalyzing}
        />

        <main className="flex-1 overflow-y-auto p-6">
          {mode === 'single' && !prediction && <WelcomePage />}

          {mode === 'single' && predictMutation.isPending && <LoadingSkeleton />}

          {mode === 'single' && prediction && currentPatient && (
            <div className="space-y-8">
              <PatientResults prediction={prediction} factors={factors} isDemo={isDemo} />
              <hr className="border-gray-700" />
              <XaiTabs patient={currentPatient} prediction={prediction} factors={factors} />
            </div>
          )}

          {mode === 'batch' && !batch.results && !batch.isProcessing && <BatchWelcome />}

          {mode === 'batch' && batch.isProcessing && <BatchProgress progress={batch.progress} />}

          {mode === 'batch' && batch.error && (
            <div className="rounded-lg border border-red-500/50 bg-red-900/20 p-4 text-red-300">{batch.error}</div>
          )}

          {mode === 'batch' && batch.results && (
            <div className="space-y-8">
              {batch.mode && (
                <div
                  className={`rounded-lg p-3 text-sm ${
                    batch.mode === 'api'
                      ? 'bg-green-900/20 text-green-300'
                      : 'bg-yellow-900/20 text-yellow-300'
                  }`}
                >
                  {batch.mode === 'api'
                    ? 'Tryb API - Predykcje z wytrenowanego modelu'
                    : 'Tryb Demo - Predykcje symulowane (bez modelu ML)'}
                </div>
              )}
              <BatchSummary results={batch.results} />
              <hr className="border-gray-700" />
              <BatchCharts results={batch.results} />
              <hr className="border-gray-700" />
              <ResultsTable results={batch.results} />
              <hr className="border-gray-700" />
              <ExportButtons results={batch.results} />
            </div>
          )}
        </main>
      </div>

      <Footer />
    </div>
  );
}
