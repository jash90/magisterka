import { useState, useRef, useEffect } from 'react';
import { useChat } from '../../hooks/useApi';
import { GaugeChart } from '../charts/GaugeChart';
import { WaterfallChart } from '../charts/WaterfallChart';
import { RiskBadge } from '../common/RiskBadge';
import { MarkdownMessage } from '../common/MarkdownMessage';
import { pl } from '../../i18n/pl';
import type { PatientInput, ChatPredictionData } from '../../api/types';

interface ChatTabProps {
  patient: PatientInput;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  prediction_data?: ChatPredictionData | null;
}

export function ChatTab({ patient }: ChatTabProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const chatMutation = useChat();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    const prompt = input.trim();
    if (!prompt) return;

    const userMsg: Message = { role: 'user', content: prompt };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');

    try {
      const history = messages.slice(-20).map((m) => ({ role: m.role, content: m.content }));
      const res = await chatMutation.mutateAsync({
        message: prompt,
        patient,
        health_literacy: 'basic',
        conversation_history: history,
      });
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: res.response, prediction_data: res.prediction_data },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'Nie udało się uzyskać odpowiedzi z serwera. Sprawdź połączenie z API i spróbuj ponownie.',
        },
      ]);
    }
  };

  return (
    <div>
      <h3 className="mb-4 text-lg font-semibold text-gray-200">{pl.xai.chatTitle}</h3>

      <div className="flex h-[32rem] flex-col rounded-lg border border-gray-700 bg-gray-800/50">
        <div className="flex-1 space-y-3 overflow-y-auto p-4">
          {messages.length === 0 && (
            <p className="text-center text-sm text-gray-500">Zadaj pytanie o wyniki analizy</p>
          )}
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`${
                msg.role === 'user' ? 'ml-auto max-w-[80%]' : 'mr-auto max-w-[95%]'
              }`}
            >
              <div
                className={`rounded-lg px-3 py-2 text-sm leading-relaxed ${
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

              {/* Inline prediction charts when agent returns prediction data */}
              {msg.role === 'assistant' && msg.prediction_data && (
                <div className="mt-2 rounded-lg border border-gray-600 bg-gray-800 p-3">
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
          {chatMutation.isPending && (
            <div className="max-w-[80%] rounded-lg bg-gray-700 px-3 py-2 text-sm text-gray-400">
              Piszę odpowiedź...
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <div className="flex border-t border-gray-700 p-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
            placeholder={pl.xai.chatPlaceholder}
            className="flex-1 rounded-l-lg border border-gray-600 bg-gray-700 px-3 py-2 text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
            disabled={chatMutation.isPending}
          />
          <button
            onClick={handleSend}
            disabled={chatMutation.isPending || !input.trim()}
            className="rounded-r-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            Wyślij
          </button>
        </div>
      </div>
    </div>
  );
}
