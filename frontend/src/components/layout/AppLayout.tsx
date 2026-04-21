import { useState, useCallback } from 'react';
import { Footer } from './Footer';
import { WelcomePage } from '../welcome/WelcomePage';
import { PatientResults } from '../patient/PatientResults';
import { XaiTabs } from '../xai/XaiTabs';
import { AgentChatView } from '../xai/AgentChatView';
import { PatientForm } from '../patient/PatientForm';
import { FileUpload } from '../batch/FileUpload';
import { BatchSummary } from '../batch/BatchSummary';
import { BatchCharts } from '../batch/BatchCharts';
import { ResultsTable } from '../batch/ResultsTable';
import { ExportButtons } from '../batch/ExportButtons';
import { BatchProgress } from '../batch/BatchProgress';
import { LoadingSkeleton } from '../common/LoadingSkeleton';
import { DemoModeIndicator } from '../common/DemoModeIndicator';
import { usePredictAll } from '../../hooks/useApi';
import { useBatchAnalysis } from '../../hooks/useBatchAnalysis';
import { explainShap } from '../../api/endpoints';
import type { DemoFactor } from '../../lib/demo';
import { pl } from '../../i18n/pl';
import type { PatientInput, PredictionOutput, MultiModelPredictionOutput, ModelPrediction } from '../../api/types';

type AppMode = 'single' | 'agent' | 'batch';

const NAV_ITEMS: { id: AppMode; label: string }[] = [
  { id: 'single', label: pl.sidebar.singlePatient },
  { id: 'agent', label: 'Agent AI' },
  { id: 'batch', label: pl.sidebar.batchAnalysis },
];

const SAMPLE_CSV = `id,wiek,plec,wiek_rozpoznania,liczba_narzadow,nerki,serce,oit,dializa,kreatynina,crp
P001,65,M,60,3,1,0,0,0,120,45
P002,45,K,40,2,0,0,0,0,85,22
P003,72,M,68,4,1,1,1,0,180,88
P004,38,K,35,1,0,0,0,0,75,15
P005,55,M,50,2,1,0,0,0,110,35
P006,68,K,62,3,1,1,0,1,220,95
P007,42,M,38,2,0,0,0,0,90,28
P008,78,K,70,5,1,1,1,1,250,120
P009,51,M,48,2,0,0,0,0,95,30
P010,63,K,58,3,1,0,1,0,145,55`;

export function AppLayout() {
  const [mode, setMode] = useState<AppMode>('single');
  const [prediction, setPrediction] = useState<PredictionOutput | null>(null);
  const [factors, setFactors] = useState<DemoFactor[]>([]);
  const [currentPatient, setCurrentPatient] = useState<PatientInput | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [multiModel, setMultiModel] = useState<MultiModelPredictionOutput | null>(null);
  const predictMutation = usePredictAll();
  const batch = useBatchAnalysis();

  const handleAnalyze = useCallback(
    async (patient: PatientInput) => {
      setCurrentPatient(patient);
      setPrediction(null);
      setMultiModel(null);
      setFactors([]);
      setError(null);
      try {
        const multiResult = await predictMutation.mutateAsync(patient);
        setMultiModel(multiResult);
        // Primary prediction from ensemble
        setPrediction({
          probability: multiResult.ensemble_probability,
          risk_level: multiResult.ensemble_risk_level,
          prediction: multiResult.ensemble_probability > 0.5 ? 1 : 0,
        });

        try {
          const shapData = await explainShap({ patient, method: 'shap', num_features: 10 });
          const apiFactors = [...shapData.risk_factors, ...shapData.protective_factors].map((f) => ({
            feature: f.feature,
            contribution: f.contribution,
          }));
          setFactors(apiFactors);
        } catch {
          setFactors([]);
        }
      } catch {
        setPrediction(null);
        setError(pl.common.predictionError);
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
    (newMode: AppMode) => {
      setMode(newMode);
      setError(null);
      if (newMode === 'single') {
        batch.reset();
      } else if (newMode === 'batch') {
        setPrediction(null);
        setFactors([]);
        setCurrentPatient(null);
      }
    },
    [batch],
  );

  function handleSampleDownload() {
    const blob = new Blob([SAMPLE_CSV], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'przykladowi_pacjenci.csv';
    a.click();
    URL.revokeObjectURL(url);
  }

  const isAnalyzing = predictMutation.isPending || batch.isProcessing;

  return (
    <div className="flex min-h-screen flex-col">
      {/* ====== HEADER ====== */}
      <header className="sticky top-0 z-30 border-b border-gray-700/80 bg-gray-900/95 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
          <h1 className="text-lg font-bold text-blue-300 md:text-xl">{pl.app.title}</h1>
          <DemoModeIndicator />
        </div>

        <nav className="mx-auto flex max-w-7xl px-6">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              onClick={() => handleModeChange(item.id)}
              className={`relative px-5 py-2.5 text-sm font-medium transition ${
                mode === item.id
                  ? 'text-blue-400'
                  : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              {item.label}
              {mode === item.id && (
                <span className="absolute inset-x-0 bottom-0 h-0.5 bg-blue-500" />
              )}
            </button>
          ))}
        </nav>
      </header>

      {/* ====== MAIN CONTENT — centered, no sidebar ====== */}
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-5xl px-6 py-8">

          {/* ========== SINGLE PATIENT ========== */}
          {mode === 'single' && !prediction && !predictMutation.isPending && !error && (
            <div className="space-y-10">
              <WelcomePage />
              <hr className="border-gray-700/50" />
              <div className="mx-auto max-w-2xl">
                          <div className="rounded-xl border border-gray-700 bg-gray-800/40 p-6">
                  <PatientForm onSubmit={handleAnalyze} isSubmitting={isAnalyzing} />
                </div>
              </div>
            </div>
          )}

          {mode === 'single' && predictMutation.isPending && (
            <div className="mx-auto max-w-2xl">
              <LoadingSkeleton />
            </div>
          )}

          {mode === 'single' && error && !predictMutation.isPending && (
            <div className="mx-auto max-w-2xl">
              <div className="rounded-xl border border-red-500/40 bg-red-900/15 p-8 text-center">
                <p className="text-lg font-semibold text-red-300">{error}</p>
                <p className="mt-3 text-sm text-gray-400">
                  Upewnij się, że serwer API jest uruchomiony i spróbuj ponownie.
                </p>
              </div>
            </div>
          )}

          {mode === 'single' && prediction && currentPatient && (
            <div className="space-y-10">
              <PatientResults prediction={prediction} factors={factors} models={multiModel?.models} />
              <hr className="border-gray-700/50" />
              <XaiTabs patient={currentPatient} prediction={prediction} factors={factors} />
            </div>
          )}

          {/* ========== AGENT AI ========== */}
          {mode === 'agent' && (
            <AgentChatView />
          )}

          {/* ========== BATCH ANALYSIS ========== */}
          {mode === 'batch' && !batch.results && !batch.isProcessing && (
            <div className="mx-auto max-w-3xl space-y-10">
              {/* Upload area */}
              <div className="rounded-xl border border-gray-700 bg-gray-800/40 p-8 text-center">
                <h2 className="mb-2 text-xl font-bold text-blue-300">{pl.batch.title}</h2>
                <p className="mb-6 text-sm text-gray-400">{pl.batch.subtitle}</p>
                <div className="mx-auto max-w-md">
                  <FileUpload onFileSelect={handleFileSelect} disabled={isAnalyzing} />
                </div>
                <button
                  onClick={handleSampleDownload}
                  className="mt-4 text-sm text-blue-400 underline decoration-blue-400/30 transition hover:text-blue-300"
                >
                  {pl.batch.downloadSample}
                </button>
                <p className="mt-2 text-xs text-gray-500">Obsługiwane formaty: CSV, JSON | Max 100 MB | Do 50 000+ pacjentów</p>
              </div>

              {/* Format info */}
              <div>
                <h3 className="mb-4 text-lg font-semibold text-gray-200">{pl.batch.fileFormats}</h3>
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="rounded-xl border border-gray-700 bg-gray-800/30 p-5">
                    <h4 className="mb-2 font-medium text-gray-200">CSV</h4>
                    <ul className="space-y-1.5 text-sm text-gray-400">
                      <li className="flex items-start gap-2"><span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-400" />Pierwszy wiersz: nagłówki kolumn</li>
                      <li className="flex items-start gap-2"><span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-400" />Separatory: przecinek, średnik, tab</li>
                      <li className="flex items-start gap-2"><span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-400" />Kodowanie: UTF-8</li>
                    </ul>
                  </div>
                  <div className="rounded-xl border border-gray-700 bg-gray-800/30 p-5">
                    <h4 className="mb-2 font-medium text-gray-200">JSON</h4>
                    <ul className="space-y-1.5 text-sm text-gray-400">
                      <li className="flex items-start gap-2"><span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-400" />Tablica obiektów: {'[{...}, {...}]'}</li>
                      <li className="flex items-start gap-2"><span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-400" />Obiekt z kluczem patients lub data</li>
                      <li className="flex items-start gap-2"><span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-400" />Kodowanie: UTF-8</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          )}

          {mode === 'batch' && batch.isProcessing && (
            <div className="mx-auto max-w-3xl">
              <BatchProgress progress={batch.progress} />
            </div>
          )}

          {mode === 'batch' && batch.error && (
            <div className="mx-auto max-w-3xl">
              <div className="rounded-xl border border-red-500/40 bg-red-900/15 p-6 text-red-300">{batch.error}</div>
            </div>
          )}

          {mode === 'batch' && batch.results && (
            <div className="space-y-10">
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
              <hr className="border-gray-700/50" />
              <BatchCharts results={batch.results} />
              <hr className="border-gray-700/50" />
              <ResultsTable results={batch.results} />
              <hr className="border-gray-700/50" />
              <ExportButtons results={batch.results} />
            </div>
          )}
        </div>
      </main>

      <Footer />
    </div>
  );
}
