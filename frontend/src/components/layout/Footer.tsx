import { pl } from '../../i18n/pl';

export function Footer() {
  return (
    <footer className="mt-auto border-t border-gray-700 bg-gray-900/50">
      <div className="mx-auto max-w-7xl px-4 py-6">
        <div className="rounded-lg bg-gray-800 p-4 text-sm text-gray-400">
          <strong className="text-gray-300">Ważne:</strong> {pl.footer.disclaimer}
        </div>
        <p className="mt-4 text-center text-xs text-gray-500">{pl.app.version} | &copy; 2024</p>
      </div>
    </footer>
  );
}
