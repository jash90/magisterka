import { pl } from '../../i18n/pl';

function InfoCard({ title, items }: { title: string; items: readonly string[] }) {
  return (
    <div className="rounded-xl border border-gray-700/60 bg-gradient-to-br from-gray-800/60 to-gray-900/60 p-5">
      <h3 className="mb-3 text-base font-semibold text-blue-300">{title}</h3>
      <ul className="space-y-1.5 text-sm text-gray-300">
        {items.map((item) => (
          <li key={item} className="flex items-start gap-2">
            <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-400" />
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function WelcomePage() {
  const w = pl.welcome;

  return (
    <div className="mx-auto max-w-3xl space-y-8 text-center">
      <div>
        <h2 className="mb-2 text-2xl font-bold text-blue-300">{w.title}</h2>
        <p className="text-gray-400">{w.subtitle}</p>
      </div>

      <div className="flex justify-center gap-8">
        {w.features.map((f) => (
          <div key={f.title} className="text-gray-300">
            <strong className="text-blue-200">{f.title}</strong>
            <span className="text-gray-500"> — </span>
            {f.desc}
          </div>
        ))}
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <InfoCard title={w.models.title} items={w.models.items} />
        <InfoCard title={w.xai.title} items={w.xai.items} />
        <InfoCard title={w.metrics.title} items={w.metrics.items} />
      </div>
    </div>
  );
}
