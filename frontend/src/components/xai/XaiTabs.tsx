import { useState } from 'react';
import { ShapTab } from './ShapTab';
import { LimeTab } from './LimeTab';
import { ComparisonTab } from './ComparisonTab';
import { ChatTab } from './ChatTab';
import { pl } from '../../i18n/pl';
import type { PatientInput, PredictionOutput } from '../../api/types';
import type { DemoFactor } from '../../lib/demo';

interface XaiTabsProps {
  patient: PatientInput;
  prediction: PredictionOutput;
  factors: DemoFactor[];
}

const TABS = [
  { id: 'shap', label: pl.xai.shap },
  { id: 'lime', label: pl.xai.lime },
  { id: 'comparison', label: pl.xai.comparison },
  { id: 'chat', label: pl.xai.chat },
] as const;

type TabId = (typeof TABS)[number]['id'];

export function XaiTabs({ patient, prediction, factors }: XaiTabsProps) {
  const [activeTab, setActiveTab] = useState<TabId>('shap');

  return (
    <div>
      <div className="flex border-b border-gray-700">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2.5 text-sm font-medium transition ${
              activeTab === tab.id
                ? 'border-b-2 border-blue-500 text-blue-400'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="mt-4">
        {activeTab === 'shap' && <ShapTab patient={patient} factors={factors} />}
        {activeTab === 'lime' && <LimeTab patient={patient} factors={factors} />}
        {activeTab === 'comparison' && <ComparisonTab patient={patient} factors={factors} />}
        {activeTab === 'chat' && <ChatTab patient={patient} prediction={prediction} factors={factors} />}
      </div>
    </div>
  );
}
