import Plot from './Plot';

interface GaugeChartProps {
  probability: number;
  title?: string;
}

export function GaugeChart({ probability, title = 'Ryzyko' }: GaugeChartProps) {
  const color = probability < 0.3 ? '#28a745' : probability < 0.7 ? '#ffc107' : '#dc3545';

  return (
    <Plot
      data={[
        {
          type: 'indicator',
          mode: 'gauge+number',
          value: probability * 100,
          domain: { x: [0, 1], y: [0, 1] },
          title: { text: title, font: { size: 20, color: '#ffffff' } },
          number: { suffix: '%', font: { size: 40, color: '#ffffff' } },
          gauge: {
            axis: { range: [0, 100], tickwidth: 1, tickcolor: '#ffffff', tickfont: { color: '#ffffff' } },
            bar: { color },
            bgcolor: '#2d2d2d',
            borderwidth: 2,
            bordercolor: '#555555',
            steps: [
              { range: [0, 30], color: '#1e4620' },
              { range: [30, 70], color: '#5c4a1e' },
              { range: [70, 100], color: '#5c1e1e' },
            ],
            threshold: {
              line: { color: '#ffffff', width: 4 },
              thickness: 0.75,
              value: probability * 100,
            },
          },
        },
      ]}
      layout={{
        height: 300,
        margin: { l: 20, r: 20, t: 50, b: 20 },
        font: { color: '#ffffff', size: 14 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        template: 'plotly_dark' as unknown as undefined,
      }}
      config={{ displayModeBar: false, responsive: true }}
      useResizeHandler
      className="w-full"
    />
  );
}
