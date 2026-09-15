import type { Comparison } from '../api/client';
import { points } from '../lib/format';

export function DeltaChart({ comparison }: { comparison: Comparison }) {
  const rows = [
    { name: 'Overall accuracy', delta: comparison.delta, ci95: comparison.ci95 },
    ...comparison.slices.map((s) => ({ ...s, name: `${s.name.replaceAll('_', ' ')} · n=${s.n}` })),
  ];
  const extent =
    Math.max(0.025, comparison.margin * 1.7, ...rows.flatMap((r) => r.ci95.map(Math.abs))) * 1.2;
  const x = (value: number) => 205 + ((value + extent) / (2 * extent)) * 490;
  const height = rows.length * 52 + 72;
  const ticks = [-extent, -extent / 2, 0, extent / 2, extent];
  return (
    <div className="chart-wrap">
      <svg
        viewBox={`0 0 740 ${height}`}
        role="img"
        aria-label={`${comparison.verdict}. Accuracy difference ${points(comparison.delta)} points, 95 percent interval ${points(comparison.ci95[0])} to ${points(comparison.ci95[1])}, margin ${points(-comparison.margin)} points.`}
      >
        <rect
          x="205"
          y="16"
          width={x(-comparison.margin) - 205}
          height={height - 50}
          fill="#fef2f2"
          rx="4"
        />
        {ticks.map((tick) => (
          <g key={tick}>
            <line x1={x(tick)} x2={x(tick)} y1="16" y2={height - 34} stroke="#e5e7eb" />
            <text x={x(tick)} y={height - 12} textAnchor="middle" className="chart-tick">
              {points(tick)}
            </text>
          </g>
        ))}
        <line
          x1={x(-comparison.margin)}
          x2={x(-comparison.margin)}
          y1="16"
          y2={height - 34}
          stroke="#d97757"
          strokeDasharray="5 4"
        />
        <line x1={x(0)} x2={x(0)} y1="16" y2={height - 34} stroke="#9ca3af" />
        {rows.map((row, index) => {
          const y = 46 + index * 52;
          return (
            <g key={row.name}>
              <text x="0" y={y + 5} className={index === 0 ? 'chart-label strong' : 'chart-label'}>
                {row.name}
              </text>
              <line
                x1={x(row.ci95[0])}
                x2={x(row.ci95[1])}
                y1={y}
                y2={y}
                stroke={index === 0 ? '#334155' : '#94a3b8'}
                strokeWidth="3"
              />
              <line
                x1={x(row.ci95[0])}
                x2={x(row.ci95[0])}
                y1={y - 6}
                y2={y + 6}
                stroke="#64748b"
              />
              <line
                x1={x(row.ci95[1])}
                x2={x(row.ci95[1])}
                y1={y - 6}
                y2={y + 6}
                stroke="#64748b"
              />
              <circle
                cx={x(row.delta)}
                cy={y}
                r={index === 0 ? 6 : 4}
                fill={index === 0 ? '#d97757' : '#64748b'}
              />
            </g>
          );
        })}
      </svg>
      <div className="chart-legend">
        <span>
          <i className="legend-dot" />
          Observed difference
        </span>
        <span>
          <i className="legend-line" />
          95% paired interval
        </span>
        <span>
          <i className="legend-margin" />
          Accepted margin
        </span>
      </div>
    </div>
  );
}
