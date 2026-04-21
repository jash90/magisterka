import Plot from './Plot';
import type { BatchResultRow } from '../../api/types';

export function ProbabilityHistogram({ results }: { results: BatchResultRow[] }) {
  const probabilities = results.map((r) => r.probability * 100);

  return (
    <Plot
      data={[
        {
          type: 'histogram',
          x: probabilities,
          nbinsx: 20,
          marker: { color: '#2874a6', opacity: 0.8 },
          name: 'Pacjenci',
        },
      ]}
      layout={{
        title: { text: 'Rozkład prawdopodobieństw ryzyka', font: { size: 18, color: '#ffffff' } },
        xaxis: {
          title: { text: 'Prawdopodobieństwo (%)', font: { color: '#ffffff' } },
          tickfont: { color: '#ffffff' },
          gridcolor: '#444444',
        },
        yaxis: {
          title: { text: 'Liczba pacjentów', font: { color: '#ffffff' } },
          tickfont: { color: '#ffffff' },
          gridcolor: '#444444',
        },
        shapes: [
          { type: 'line', x0: 30, x1: 30, y0: 0, y1: 1, yref: 'paper', line: { color: '#28a745', dash: 'dash' } },
          { type: 'line', x0: 70, x1: 70, y0: 0, y1: 1, yref: 'paper', line: { color: '#dc3545', dash: 'dash' } },
        ],
        font: { color: '#ffffff' },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        template: 'plotly_dark' as unknown as undefined,
        height: 350,
        bargap: 0.1,
      }}
      config={{ displayModeBar: false, responsive: true }}
      useResizeHandler
      className="w-full"
    />
  );
}
