import Plot from './Plot';

interface Factor {
  feature: string;
  contribution: number;
}

interface HorizontalBarChartProps {
  factors: Factor[];
  title?: string;
}

export function HorizontalBarChart({ factors, title = 'Ważność czynników' }: HorizontalBarChartProps) {
  if (!factors.length) return null;

  const names = factors.map((f) => f.feature);
  const values = factors.map((f) => Math.abs(f.contribution));
  const colors = factors.map((f) => (f.contribution > 0 ? '#dc3545' : '#28a745'));

  return (
    <Plot
      data={[
        {
          type: 'bar',
          y: names,
          x: values,
          orientation: 'h',
          marker: { color: colors },
          text: values.map((v) => v.toFixed(3)),
          textposition: 'outside',
          textfont: { size: 13, color: '#ffffff', family: 'Arial Black' },
        },
      ]}
      layout={{
        title: { text: title, font: { size: 18, color: '#ffffff' } },
        xaxis: {
          title: { text: 'Bezwzględny wpływ', font: { size: 14, color: '#ffffff' } },
          tickfont: { color: '#ffffff' },
          gridcolor: '#444444',
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
