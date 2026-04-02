import { useState } from 'react';
import { ShapTab } from './ShapTab';
import { LimeTab } from './LimeTab';
import { ComparisonTab } from './ComparisonTab';
import { DalexTab } from './DalexTab';
import { EbmTab } from './EbmTab';
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
  { id: 'dalex', label: 'DALEX' },
  { id: 'ebm', label: 'EBM' },
  { id: 'comparison', label: pl.xai.comparison },
] as const;

type TabId = (typeof TABS)[number]['id'];

export function XaiTabs({ patient, prediction: _prediction, factors }: XaiTabsProps) {
  const [activeTab, setActiveTab] = useState<TabId>('shap');

  return (
    <div className="space-y-10">
      {/* XAI method tabs */}
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
          {activeTab === 'dalex' && <DalexTab patient={patient} factors={factors} />}
          {activeTab === 'ebm' && <EbmTab patient={patient} factors={factors} />}
          {activeTab === 'comparison' && <ComparisonTab patient={patient} factors={factors} />}
        </div>
      </div>

      {/* Chat AI — always visible below */}
      <hr className="border-gray-700/50" />
      <div>
        <h3 className="mb-3 text-lg font-semibold text-blue-300">{pl.xai.chatTitle}</h3>
        <ChatTab patient={patient} />
      </div>
    </div>
  );
}
