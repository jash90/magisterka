import Plot from './Plot';
import type { BatchResultRow } from '../../api/types';

export function RiskPieChart({ results }: { results: BatchResultRow[] }) {
  const counts = { low: 0, moderate: 0, high: 0 };
  results.forEach((r) => {
    if (r.risk_level in counts) counts[r.risk_level as keyof typeof counts]++;
  });

  const labels = ['Niskie', 'Umiarkowane', 'Wysokie'];
  const values = [counts.low, counts.moderate, counts.high];
  const colors = ['#28a745', '#ffc107', '#dc3545'];

  return (
    <Plot
      data={[
        {
          type: 'pie',
          labels,
          values,
          marker: { colors },
          hole: 0.4,
          textinfo: 'label+percent+value',
          textfont: { size: 14, color: 'white' },
        },
      ]}
      layout={{
        title: { text: 'Rozkład poziomów ryzyka', font: { size: 18, color: '#ffffff' } },
        font: { color: '#ffffff' },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        template: 'plotly_dark' as unknown as undefined,
        height: 350,
        showlegend: true,
        legend: { font: { color: '#ffffff' } },
      }}
      config={{ displayModeBar: false, responsive: true }}
      useResizeHandler
      className="w-full"
    />
  );
}
