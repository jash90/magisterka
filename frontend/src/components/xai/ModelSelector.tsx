import { pl } from '../../i18n/pl';

interface ModelSelectorProps {
  value: string;
  onChange: (key: string) => void;
}

const MODELS = [
  { key: 'xgboost', label: 'XGBoost' },
  { key: 'random_forest', label: 'Random Forest' },
  { key: 'lightgbm', label: 'LightGBM' },
];

export function ModelSelector({ value, onChange }: ModelSelectorProps) {
  return (
    <div className="flex items-center gap-2 mb-4">
      <span className="text-xs text-gray-500">Model:</span>
      <div className="flex rounded-lg border border-gray-600 overflow-hidden">
        {MODELS.map((m) => (
          <button
            key={m.key}
            onClick={() => onChange(m.key)}
            className={`px-3 py-1.5 text-xs font-medium transition ${
              value === m.key
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-200'
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>
    </div>
  );
}
