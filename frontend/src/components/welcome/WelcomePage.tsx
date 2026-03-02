import { pl } from '../../i18n/pl';

function InfoCard({ title, items }: { title: string; items: string[] }) {
  return (
    <div className="rounded-lg border border-gray-700 bg-gradient-to-br from-gray-800 to-gray-900 p-5">
      <h3 className="mb-3 text-base font-semibold text-blue-300">{title}</h3>
      <ul className="space-y-1 text-sm text-gray-300">
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

export function WelcomePage() {
  const w = pl.welcome;

  return (
    <div className="mx-auto max-w-3xl space-y-8">
      <div>
        <h2 className="mb-2 text-2xl font-bold text-blue-300">{w.title}</h2>
        <p className="text-gray-400">{w.subtitle}</p>
        <ul className="mt-3 space-y-2">
          {w.features.map((f) => (
            <li key={f.title} className="text-gray-300">
              <strong>{f.title}</strong> - {f.desc}
            </li>
          ))}
        </ul>
      </div>

      <div>
        <h3 className="mb-2 text-lg font-semibold text-blue-300">{w.howTo}</h3>
        <ol className="list-inside list-decimal space-y-1 text-gray-400">
          {w.steps.map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ol>
      </div>

      <hr className="border-gray-700" />

      <div className="grid gap-4 md:grid-cols-3">
        <InfoCard title={w.models.title} items={w.models.items} />
        <InfoCard title={w.xai.title} items={w.xai.items} />
        <InfoCard title={w.metrics.title} items={w.metrics.items} />
      </div>
    </div>
  );
}
