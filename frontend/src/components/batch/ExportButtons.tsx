import type { BatchResultRow } from '../../api/types';
import { exportToCSV, exportToJSON, downloadBlob } from '../../lib/exportUtils';
import { pl } from '../../i18n/pl';

export function ExportButtons({ results }: { results: BatchResultRow[] }) {
  const ts = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, (m) => (m === 'T' ? '_' : ''));

  return (
    <div>
      <h2 className="mb-4 text-xl font-bold text-gray-200">{pl.batch.exportResults}</h2>
      <div className="flex gap-4">
        <button
          onClick={() => downloadBlob(exportToCSV(results), `wyniki_analizy_${ts}.csv`, 'text/csv;charset=utf-8')}
          className="flex-1 rounded-lg bg-blue-600 py-2.5 text-sm font-medium text-white hover:bg-blue-700"
        >
          {pl.batch.downloadCsv}
        </button>
        <button
          onClick={() => downloadBlob(exportToJSON(results), `wyniki_analizy_${ts}.json`, 'application/json')}
          className="flex-1 rounded-lg border border-gray-600 py-2.5 text-sm font-medium text-gray-300 hover:bg-gray-800"
        >
          {pl.batch.downloadJson}
        </button>
      </div>
    </div>
  );
}
