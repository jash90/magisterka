import { useState, useRef, useEffect } from 'react';
import { useChat } from '../../hooks/useApi';
import { pl } from '../../i18n/pl';
import type { PatientInput, PredictionOutput } from '../../api/types';
import type { DemoFactor } from '../../lib/demo';

interface ChatTabProps {
  patient: PatientInput;
  prediction: PredictionOutput;
  factors: DemoFactor[];
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

function getDemoResponse(prompt: string, prediction: PredictionOutput, factors: DemoFactor[]): string {
  const sorted = [...factors].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

  if (['wynik', 'ryzyko', 'analiza'].some((w) => prompt.toLowerCase().includes(w))) {
    return `Na podstawie analizy, poziom ryzyka wynosi **${prediction.risk_level}** (prawdopodobieństwo: ${(prediction.probability * 100).toFixed(1)}%).\n\nGłówne czynniki to:\n- ${sorted[0]?.feature ?? 'brak'}\n- ${sorted[1]?.feature ?? 'brak'}\n\nCzy masz dodatkowe pytania?\n\n*Pamiętaj: to narzędzie informacyjne, skonsultuj się z lekarzem.*`;
  }
  if (['czynnik', 'dlaczego'].some((w) => prompt.toLowerCase().includes(w))) {
    let resp = 'Główne czynniki wpływające na ocenę to:\n\n';
    sorted.slice(0, 3).forEach((f) => {
      resp += `- **${f.feature}**: wpływ ${f.contribution >= 0 ? '+' : ''}${f.contribution.toFixed(3)}\n`;
    });
    return resp + '\n\n*Pamiętaj: to narzędzie informacyjne, skonsultuj się z lekarzem.*';
  }
  return 'Mogę pomóc Ci zrozumieć:\n- Wyniki analizy ryzyka\n- Czynniki wpływające na ocenę\n- Znaczenie poszczególnych wskaźników\n\nO czym chciałbyś porozmawiać?\n\n*Pamiętaj: to narzędzie informacyjne, skonsultuj się z lekarzem.*';
}

export function ChatTab({ patient, prediction, factors }: ChatTabProps) {
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
      setMessages((prev) => [...prev, { role: 'assistant', content: res.response }]);
    } catch {
      const demoResp = getDemoResponse(prompt, prediction, factors);
      setMessages((prev) => [...prev, { role: 'assistant', content: demoResp }]);
    }
  };

  return (
    <div>
      <h3 className="mb-4 text-lg font-semibold text-gray-200">{pl.xai.chatTitle}</h3>

      <div className="flex h-96 flex-col rounded-lg border border-gray-700 bg-gray-800/50">
        <div className="flex-1 space-y-3 overflow-y-auto p-4">
          {messages.length === 0 && (
            <p className="text-center text-sm text-gray-500">Zadaj pytanie o wyniki analizy</p>
          )}
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`max-w-[80%] rounded-lg px-3 py-2 text-sm ${
                msg.role === 'user'
                  ? 'ml-auto bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-200'
              }`}
            >
              {msg.content.split('\n').map((line, j) => (
                <p key={j} className={j > 0 ? 'mt-1' : ''}>
                  {line}
                </p>
              ))}
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
