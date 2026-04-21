import Plot from './Plot';
import type { BatchResultRow, RiskLevel } from '../../api/types';

export function AgeRiskScatter({ results }: { results: BatchResultRow[] }) {
  const colors: Record<RiskLevel, string> = { low: '#28a745', moderate: '#ffc107', high: '#dc3545' };
  const labels: Record<RiskLevel, string> = { low: 'Niskie', moderate: 'Umiarkowane', high: 'Wysokie' };

  const traces = (['low', 'moderate', 'high'] as RiskLevel[]).map((level) => {
    const filtered = results.filter((r) => r.risk_level === level);
    return {
      type: 'scatter' as const,
      mode: 'markers' as const,
      x: filtered.map((r) => r.wiek_rozpoznania),
      y: filtered.map((r) => r.probability * 100),
      marker: { size: 10, color: colors[level], opacity: 0.7 },
      name: labels[level],
      text: filtered.map((r) => r.patient_id),
      hovertemplate: '<b>%{text}</b><br>Wiek rozp.: %{x}<br>Ryzyko: %{y:.1f}%<extra></extra>',
    };
  });

  return (
    <Plot
      data={traces}
      layout={{
        title: { text: 'Wiek a ryzyko zgonu', font: { size: 18, color: '#ffffff' } },
        xaxis: {
          title: { text: 'Wiek (lata)', font: { color: '#ffffff' } },
          tickfont: { color: '#ffffff' },
          gridcolor: '#444444',
        },
        yaxis: {
          title: { text: 'Prawdopodobieństwo (%)', font: { color: '#ffffff' } },
          tickfont: { color: '#ffffff' },
          gridcolor: '#444444',
        },
        font: { color: '#ffffff' },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        template: 'plotly_dark' as unknown as undefined,
        height: 400,
        legend: { font: { color: '#ffffff' } },
      }}
      config={{ displayModeBar: false, responsive: true }}
      useResizeHandler
      className="w-full"
    />
  );
}
