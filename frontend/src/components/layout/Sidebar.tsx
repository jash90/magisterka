import { pl } from '../../i18n/pl';
import { PatientForm } from '../patient/PatientForm';
import { FileUpload } from '../batch/FileUpload';
import type { PatientInput } from '../../api/types';

interface SidebarProps {
  mode: 'single' | 'batch';
  onModeChange: (mode: 'single' | 'batch') => void;
  onAnalyze: (patient: PatientInput) => void;
  onFileSelect: (file: File) => void;
  isAnalyzing: boolean;
}

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

export function Sidebar({ mode, onModeChange, onAnalyze, onFileSelect, isAnalyzing }: SidebarProps) {
  function handleSampleDownload() {
    const blob = new Blob([SAMPLE_CSV], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'przykladowi_pacjenci.csv';
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <aside className="w-80 shrink-0 overflow-y-auto border-r border-gray-700 bg-gray-900 p-4">
      <h2 className="mb-4 text-lg font-bold text-blue-300">{pl.sidebar.analysisMode}</h2>

      <div className="mb-4 flex rounded-lg bg-gray-800 p-1">
        <button
          onClick={() => onModeChange('single')}
          className={`flex-1 rounded-md px-3 py-2 text-sm font-medium transition ${
            mode === 'single' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'
          }`}
        >
          {pl.sidebar.singlePatient}
        </button>
        <button
          onClick={() => onModeChange('batch')}
          className={`flex-1 rounded-md px-3 py-2 text-sm font-medium transition ${
            mode === 'batch' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'
          }`}
        >
          {pl.sidebar.batchAnalysis}
        </button>
      </div>

      <hr className="my-4 border-gray-700" />

      {mode === 'single' ? (
        <PatientForm onSubmit={onAnalyze} isSubmitting={isAnalyzing} />
      ) : (
        <div className="space-y-4">
          <h2 className="text-lg font-bold text-blue-300">{pl.sidebar.uploadFile}</h2>
          <FileUpload onFileSelect={onFileSelect} disabled={isAnalyzing} />

          <p className="text-xs text-gray-500">Obsługiwane: do 50,000+ pacjentów | Max 100MB</p>

          <hr className="border-gray-700" />
          <button
            onClick={handleSampleDownload}
            className="w-full rounded-lg border border-gray-600 px-4 py-2 text-sm text-gray-300 hover:bg-gray-800"
          >
            {pl.batch.downloadSample}
          </button>
        </div>
      )}
    </aside>
  );
}
