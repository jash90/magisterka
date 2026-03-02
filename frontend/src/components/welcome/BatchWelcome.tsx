import { pl } from '../../i18n/pl';

export function BatchWelcome() {
  return (
    <div className="mx-auto max-w-2xl space-y-8">
      <div className="rounded-lg border-2 border-dashed border-gray-600 p-8 text-center">
        <h2 className="mb-2 text-xl font-bold text-blue-300">{pl.batch.title}</h2>
        <p className="text-gray-400">{pl.batch.subtitle}</p>
        <hr className="my-4 border-gray-600" />
        <p className="text-sm text-gray-500">&larr; {pl.batch.useSidebar}</p>
      </div>

      <div>
        <h3 className="mb-4 text-lg font-semibold text-gray-200">{pl.batch.fileFormats}</h3>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4">
            <h4 className="mb-2 font-medium text-gray-200">CSV</h4>
            <ul className="space-y-1 text-sm text-gray-400">
              <li>Pierwszy wiersz: nagłówki kolumn</li>
              <li>Separatory: przecinek, średnik, tab</li>
              <li>Kodowanie: UTF-8</li>
            </ul>
          </div>
          <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4">
            <h4 className="mb-2 font-medium text-gray-200">JSON</h4>
            <ul className="space-y-1 text-sm text-gray-400">
              <li>{'Tablica obiektów: [{...}, {...}]'}</li>
              <li>{'Obiekt z kluczem patients lub data'}</li>
              <li>Kodowanie: UTF-8</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
