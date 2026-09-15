import { useState } from 'react';
import type { Comparison } from '../api/client';
import { EmptyState } from './Feedback';
import { ModelOutput } from './ModelOutput';

export function FlipTable({ comparison }: { comparison: Comparison }) {
  const [kind, setKind] = useState<'regression' | 'fix'>('regression');
  const [page, setPage] = useState(0);
  const [expanded, setExpanded] = useState<number | null>(null);
  const rows = comparison.flips.filter((flip) => flip.kind === kind);
  function selectTab(value: 'regression' | 'fix') {
    setKind(value);
    setPage(0);
    setExpanded(null);
  }
  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <h2>Where behavior changed</h2>
          <p>Inspect the examples behind the difference.</p>
        </div>
        <span className="subtle mono">{comparison.flips.length} flipped examples</span>
      </div>
      <div className="tabs" role="tablist" aria-label="Flipped examples">
        {(['regression', 'fix'] as const).map((value) => (
          <button
            key={value}
            role="tab"
            id={`tab-${value}`}
            aria-selected={kind === value}
            aria-controls="flip-panel"
            tabIndex={kind === value ? 0 : -1}
            onClick={() => selectTab(value)}
            onKeyDown={(event) => {
              if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
              event.preventDefault();
              const next =
                event.key === 'Home'
                  ? 'regression'
                  : event.key === 'End'
                    ? 'fix'
                    : kind === 'regression'
                      ? 'fix'
                      : 'regression';
              selectTab(next);
              document.getElementById(`tab-${next}`)?.focus();
            }}
          >
            {value === 'regression' ? 'Regressions' : 'Fixes'}
            <span>{value === 'regression' ? comparison.regressions : comparison.fixes}</span>
          </button>
        ))}
      </div>
      <div id="flip-panel" role="tabpanel" aria-labelledby={`tab-${kind}`}>
        {rows.length === 0 ? (
          <EmptyState title={`No ${kind === 'regression' ? 'regressions' : 'fixes'}`}>
            Both models agree on correctness for these examples.
          </EmptyState>
        ) : (
          <div className="table-scroll">
            <table className="flip-table">
              <thead>
                <tr>
                  <th>Example / true sentiment</th>
                  <th>Baseline output</th>
                  <th>Candidate output</th>
                </tr>
              </thead>
              <tbody>
                {rows.slice(page * 10, page * 10 + 10).map((row) => (
                  <tr key={row.idx}>
                    <td>
                      <button
                        className="text-button example-text"
                        aria-expanded={expanded === row.idx}
                        onClick={() => setExpanded(expanded === row.idx ? null : row.idx)}
                      >
                        {expanded === row.idx || row.text.length < 155
                          ? row.text
                          : `${row.text.slice(0, 155)}…`}
                      </button>
                      <div className="tag-row">
                        <span className={`sentiment ${row.label ? 'positive' : 'negative'}`}>
                          {row.label ? 'Positive' : 'Negative'}
                        </span>
                        <span className="mono subtle">#{row.idx}</span>
                        {row.slices.map((tag) => (
                          <span className="tag" key={tag}>
                            {tag.replaceAll('_', ' ')}
                          </span>
                        ))}
                        {row.near_boundary && <span className="tag">Near boundary</span>}
                      </div>
                    </td>
                    <td>
                      <ModelOutput
                        label={row.baseline_pred}
                        probability={row.baseline_prob}
                        raw={row.baseline_output}
                      />
                    </td>
                    <td>
                      <ModelOutput
                        label={row.candidate_pred}
                        probability={row.candidate_prob}
                        raw={row.candidate_output}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {rows.length > 10 && (
          <div className="pagination">
            <span>
              {page * 10 + 1}–{Math.min(rows.length, page * 10 + 10)} of {rows.length}
            </span>
            <button disabled={page === 0} onClick={() => setPage(page - 1)}>
              Previous
            </button>
            <button disabled={(page + 1) * 10 >= rows.length} onClick={() => setPage(page + 1)}>
              Next
            </button>
          </div>
        )}
      </div>
    </section>
  );
}
