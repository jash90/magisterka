import Plot from './Plot';

interface Factor {
  feature: string;
  contribution: number;
}

interface WaterfallChartProps {
  factors: Factor[];
  title?: string;
}

export function WaterfallChart({ factors, title = 'Wpływ czynników' }: WaterfallChartProps) {
  if (!factors.length) return null;

  const names = factors.map((f) => f.feature);
  const values = factors.map((f) => f.contribution);

  return (
    <Plot
      data={[
        {
          type: 'waterfall',
          name: '',
          orientation: 'h',
          y: names,
          x: values,
          connector: { line: { color: 'rgb(63, 63, 63)' } },
          decreasing: { marker: { color: '#28a745' } },
          increasing: { marker: { color: '#dc3545' } },
          text: values.map((v) => (v >= 0 ? `+${v.toFixed(3)}` : v.toFixed(3))),
          textposition: 'outside',
          textfont: { size: 13, color: '#ffffff', family: 'Arial Black' },
        },
      ]}
      layout={{
        title: { text: title, font: { size: 18, color: '#ffffff' } },
        xaxis: {
          title: { text: 'Wpływ na ryzyko', font: { size: 14, color: '#ffffff' } },
          tickfont: { color: '#ffffff' },
          gridcolor: '#444444',
          zerolinecolor: '#888888',
        },
        yaxis: {
          title: { text: 'Czynnik', font: { size: 14, color: '#ffffff' } },
          tickfont: { color: '#ffffff' },
        },
        height: 400,
        margin: { l: 20, r: 20, t: 50, b: 20 },
        font: { color: '#ffffff', size: 13 },
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
