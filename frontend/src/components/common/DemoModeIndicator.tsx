import { useDemoMode } from '../../hooks/useApi';

export function DemoModeIndicator() {
  const { data, isError } = useDemoMode();

  const mode = isError ? 'unavailable' : data?.current_mode ?? 'unknown';

  if (mode === 'api') return null;

  const label = mode === 'demo' ? 'DEMO' : mode === 'unavailable' ? 'API niedostępne' : '';
  const color = mode === 'demo' ? 'bg-yellow-600' : 'bg-red-600';

  if (!label) return null;

  return (
    <div className="mt-1 flex justify-center">
      <span className={`rounded-full px-3 py-0.5 text-xs font-bold text-white ${color}`}>{label}</span>
    </div>
  );
}
