import { useState, useRef, useEffect } from 'react';
import { useAgentChat } from '../../hooks/useApi';
import { GaugeChart } from '../charts/GaugeChart';
import { WaterfallChart } from '../charts/WaterfallChart';
import { RiskBadge } from '../common/RiskBadge';
import { MarkdownMessage } from '../common/MarkdownMessage';
import type { ChatPredictionData } from '../../api/types';
import type { AgentChatPayload } from '../../api/endpoints';

interface FieldMeta {
  field: string;
  type: 'number' | 'boolean';
  widget: 'slider' | 'buttons' | 'input';
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  options?: string[];
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  prediction_data?: ChatPredictionData | null;
}

interface CollectedSummaryProps {
  data: Record<string, number | string>;
}

const FIELD_LABELS: Record<string, string> = {
  wiek_rozpoznania: 'Wiek',
  opoznienie_rozpoznia: 'Opóźnienie diagnozy (mies.)',
  manifestacja_nerki: 'Nerki',
  manifestacja_sercowo_naczyniowy: 'Serce/naczynia',
  manifestacja_zajecie_csn: 'CSN',
  manifestacja_neurologiczny: 'Neurologiczny',
  liczba_zajetych_narzadow: 'Narządy',
  zaostrz_wymagajace_hospital: 'Hospitalizacja',
  zaostrz_wymagajace_oit: 'OIT',
  kreatynina: 'Kreatynina',
  czas_sterydow: 'Sterydy (mies.)',
  plazmaferezy: 'Plazmaferezy',
  biopsja_wynik: 'Biopsja',
};

const BOOL_FIELDS = new Set([
  'manifestacja_nerki', 'manifestacja_sercowo_naczyniowy', 'manifestacja_zajecie_csn',
  'manifestacja_neurologiczny', 'plazmaferezy', 'biopsja_wynik',
  'zaostrz_wymagajace_hospital', 'zaostrz_wymagajace_oit',
]);

function CollectedSummary({ data }: CollectedSummaryProps) {
  const entries = Object.entries(data);
  if (entries.length === 0) return null;

  return (
    <div className="rounded-lg border border-gray-600 bg-gray-800/80 p-3">
      <h4 className="mb-2 text-xs font-semibold text-blue-300">Zebrane dane pacjenta:</h4>
      <div className="flex flex-wrap gap-2">
        {entries.map(([key, val]) => {
          const label = FIELD_LABELS[key] || key;
          const displayVal = typeof val === 'number' && BOOL_FIELDS.has(key)
            ? (val ? 'Tak' : 'Nie')
            : String(val);
          return (
            <span key={key} className="rounded bg-gray-700 px-2 py-0.5 text-xs text-gray-300">
              {label}: <strong>{displayVal}</strong>
            </span>
          );
        })}
      </div>
    </div>
  );
}

/* ---------- Slider input widget ---------- */
function SliderWidget({ meta, onSend, disabled }: {
  meta: FieldMeta;
  onSend: (val: string) => void;
  disabled: boolean;
}) {
  const min = meta.min ?? 0;
  const max = meta.max ?? 100;
  const step = meta.step ?? 1;
  const mid = Math.round((min + max) / 2);
  const [val, setVal] = useState(mid);

  useEffect(() => { setVal(mid); }, [min, max, mid]);

  return (
    <div className="rounded-lg border border-blue-500/30 bg-blue-900/10 p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs text-gray-400">{min}</span>
        <span className="rounded bg-gray-800 px-3 py-1 text-lg font-bold text-white tabular-nums">
          {val}{meta.unit ? ` ${meta.unit}` : ''}
        </span>
        <span className="text-xs text-gray-400">{max}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={val}
        onChange={(e) => setVal(Number(e.target.value))}
        disabled={disabled}
        className="w-full accent-blue-500"
      />
      <button
        onClick={() => onSend(String(val))}
        disabled={disabled}
        className="mt-2 w-full rounded-lg bg-blue-600 py-2 text-sm font-medium text-white transition hover:bg-blue-700 disabled:opacity-50"
      >
        Potwierdź: {val}{meta.unit ? ` ${meta.unit}` : ''}
      </button>
    </div>
  );
}

/* ---------- Buttons input widget ---------- */
function ButtonsWidget({ options, onSend, disabled }: {
  options: string[];
  onSend: (val: string) => void;
  disabled: boolean;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {options.map((opt) => (
        <button
          key={opt}
          onClick={() => onSend(opt)}
          disabled={disabled}
          className="rounded-full border border-blue-500/50 bg-blue-900/20 px-4 py-1.5 text-sm text-blue-300 transition hover:bg-blue-900/40 disabled:opacity-50"
        >
          {opt}
        </button>
      ))}
    </div>
  );
}

export function AgentChatView() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [collectedData, setCollectedData] = useState<Record<string, number | string>>({});
  const [currentStep, setCurrentStep] = useState(0);
  const [phase, setPhase] = useState<'collecting' | 'prediction' | 'discussion'>('collecting');
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [fieldMeta, setFieldMeta] = useState<FieldMeta | null>(null);
  const [started, setStarted] = useState(false);
  const agentMutation = useAgentChat();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, fieldMeta]);

  const sendToAgent = async (msg: string) => {
    const userMsg: Message = { role: 'user', content: msg };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setFieldMeta(null);

    try {
      const history = messages.slice(-20).map((m) => ({ role: m.role, content: m.content }));
      const payload: AgentChatPayload = {
        message: msg,
        conversation_history: history,
        collected_data: collectedData,
        current_step: currentStep,
        phase: phase,
      };
      const res = await agentMutation.mutateAsync(payload);

      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: res.response, prediction_data: res.prediction_data },
      ]);
      setCollectedData(res.collected_data);
      setCurrentStep(res.current_step);
      setPhase(res.phase as typeof phase);
      setSuggestions(res.follow_up_suggestions || []);
      setFieldMeta(res.field_meta ?? null);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Nie udało się uzyskać odpowiedzi z serwera. Spróbuj ponownie.' },
      ]);
    }
  };

  const startConversation = async () => {
    setStarted(true);
    try {
      const payload: AgentChatPayload = {
        message: 'start',
        conversation_history: [],
        collected_data: {},
        current_step: 0,
        phase: 'collecting',
      };
      const res = await agentMutation.mutateAsync(payload);
      setMessages([{ role: 'assistant', content: res.response }]);
      setCurrentStep(res.current_step);
      setPhase(res.phase as typeof phase);
      setSuggestions(res.follow_up_suggestions || []);
      setFieldMeta(res.field_meta ?? null);
    } catch {
      setMessages([{ role: 'assistant', content: 'Nie udało się połączyć z serwerem. Sprawdź API i spróbuj ponownie.' }]);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (input.trim()) sendToAgent(input.trim());
    }
  };

  if (!started) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="max-w-lg text-center">
          <h2 className="mb-3 text-2xl font-bold text-blue-300">Agent AI — Rozmowa o pacjencie</h2>
          <p className="mb-6 text-gray-400">
            Rozpocznij rozmowę z asystentem AI, który krok po kroku zbierze dane pacjenta,
            wykona predykcję ryzyka śmiertelności i pokaże wyniki z interaktywnymi wykresami.
          </p>
          <button
            onClick={startConversation}
            className="rounded-lg bg-blue-600 px-8 py-3 text-lg font-bold text-white transition hover:bg-blue-700"
          >
            Rozpocznij rozmowę
          </button>
        </div>
      </div>
    );
  }

  const isLoading = agentMutation.isPending;

  return (
    <div className="flex h-full flex-col">
      {/* Collected data summary */}
      {Object.keys(collectedData).length > 0 && (
        <div className="mb-3">
          <CollectedSummary data={collectedData} />
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 space-y-4 overflow-y-auto pb-4">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`${msg.role === 'user' ? 'ml-auto max-w-[80%]' : 'mr-auto max-w-[95%]'}`}
          >
            <div
              className={`rounded-lg px-4 py-3 text-sm leading-relaxed ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-200'
              }`}
            >
              {msg.role === 'user' ? (
                msg.content.split('\n').map((line, j) => (
                  <p key={j} className={j > 0 ? 'mt-1' : ''}>{line}</p>
                ))
              ) : (
                <MarkdownMessage content={msg.content} />
              )}
            </div>

            {/* Inline prediction charts */}
            {msg.role === 'assistant' && msg.prediction_data && (
              <div className="mt-3 rounded-lg border border-gray-600 bg-gray-800 p-4">
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <GaugeChart
                      probability={msg.prediction_data.prediction.probability}
                      title="Ryzyko śmiertelności"
                    />
                    <div className="mt-2 flex justify-center">
                      <RiskBadge level={msg.prediction_data.prediction.risk_level} />
                    </div>
                  </div>
                  <div>
                    {msg.prediction_data.factors.length > 0 && (
                      <WaterfallChart
                        factors={msg.prediction_data.factors}
                        title="Wpływ czynników (SHAP)"
                      />
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
        {isLoading && (
          <div className="max-w-[80%] rounded-lg bg-gray-700 px-4 py-3 text-sm text-gray-400">
            Piszę odpowiedź...
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input area — contextual widgets or text suggestions */}
      <div className="border-t border-gray-700 pt-3">
        {/* Collecting phase: show slider or button widgets */}
        {phase === 'collecting' && fieldMeta && !isLoading && (
          <div className="mb-2">
            {fieldMeta.widget === 'slider' && (
              <SliderWidget
                meta={fieldMeta}
                onSend={(val) => sendToAgent(val)}
                disabled={isLoading}
              />
            )}
            {fieldMeta.widget === 'buttons' && fieldMeta.options && (
              <ButtonsWidget
                options={fieldMeta.options}
                onSend={(val) => sendToAgent(val)}
                disabled={isLoading}
              />
            )}
          </div>
        )}

        {/* Discussion phase: show text suggestion buttons */}
        {phase === 'discussion' && suggestions.length > 0 && !isLoading && (
          <div className="mb-2 flex flex-wrap gap-2">
            {suggestions.map((s) => (
              <button
                key={s}
                onClick={() => sendToAgent(s)}
                className="rounded-full border border-blue-500/50 bg-blue-900/20 px-4 py-1.5 text-sm text-blue-300 transition hover:bg-blue-900/40"
              >
                {s}
              </button>
            ))}
          </div>
        )}

        {/* Text input — hidden when widget (slider/buttons) is shown */}
        {!(phase === 'collecting' && fieldMeta) && (
          <div className="flex">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                phase === 'collecting'
                  ? 'Lub wpisz własną wartość...'
                  : 'Zadaj pytanie...'
              }
              className="flex-1 rounded-l-lg border border-gray-600 bg-gray-700 px-4 py-3 text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
              disabled={isLoading}
            />
            <button
              onClick={() => input.trim() && sendToAgent(input.trim())}
              disabled={isLoading || !input.trim()}
              className="rounded-r-lg bg-blue-600 px-6 py-3 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
            >
              Wyślij
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
