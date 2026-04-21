import { pl } from '../../i18n/pl';
import { PatientForm } from '../patient/PatientForm';
import { FileUpload } from '../batch/FileUpload';
import type { PatientInput } from '../../api/types';

interface SidebarProps {
  mode: 'single' | 'agent' | 'batch';
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

export function Sidebar({ mode, onAnalyze, onFileSelect, isAnalyzing }: SidebarProps) {
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
      {mode === 'single' && (
        <>
          <h2 className="mb-4 text-lg font-bold text-blue-300">{pl.sidebar.patientData}</h2>
          <PatientForm onSubmit={onAnalyze} isSubmitting={isAnalyzing} />
        </>
      )}

      {mode === 'agent' && (
        <div className="space-y-4">
          <h2 className="text-lg font-bold text-blue-300">Agent AI</h2>
          <p className="text-sm text-gray-400">
            Asystent AI przeprowadzi Cię przez proces zbierania danych pacjenta w formie rozmowy,
            wykona predykcję ryzyka śmiertelności i przedstawi wyniki z interaktywnymi wykresami.
          </p>
          <div className="rounded-lg bg-blue-900/20 p-3 text-xs text-blue-300">
            <p className="font-semibold">Jak to działa?</p>
            <ol className="mt-1 list-inside list-decimal space-y-1 text-blue-400">
              <li>Kliknij &quot;Rozpocznij rozmowę&quot;</li>
              <li>Odpowiadaj na pytania agenta</li>
              <li>Otrzymasz predykcję z wykresami</li>
              <li>Możesz pytać o chorobę i czynniki</li>
            </ol>
          </div>
        </div>
      )}

      {mode === 'batch' && (
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
