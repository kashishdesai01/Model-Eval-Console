import { useMutation, useQuery } from '@tanstack/react-query';
import { ArrowRight, GitCompareArrows, Info, Terminal } from 'lucide-react';
import { useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { api } from '../api/client';
import { DeltaChart } from '../components/DeltaChart';
import { EmptyState, ErrorMessage, Loading } from '../components/Feedback';
import { FlipTable } from '../components/FlipTable';
import { VerdictBadge } from '../components/VerdictBadge';
import { percent, points, shortId } from '../lib/format';

export function ComparePage() {
  const [params, setParams] = useSearchParams();
  const id = params.get('comparison');
  const runs = useQuery({ queryKey: ['runs'], queryFn: api.runs });
  const saved = useQuery({
    queryKey: ['comparisons', id],
    queryFn: () => (id ? api.comparison(id).then((data) => [data]) : api.comparisons()),
  });
  const completed = runs.data?.filter((run) => run.status === 'succeeded') ?? [];
  const [selection, setSelection] = useState<{
    context: string;
    baseline?: string;
    candidate?: string;
    margin?: string;
  }>({ context: '' });
  const comparison = saved.data?.[0];
  const context = comparison?.id || 'new';
  const controls: Partial<typeof selection> = selection.context === context ? selection : {};
  const margin = controls.margin ?? String((comparison?.margin ?? 0.01) * 100);
  const baselineId =
    controls.baseline ||
    comparison?.baseline.run_id ||
    completed.find((r) => r.candidate_name === 'baseline')?.id ||
    completed[0]?.id ||
    '';
  const candidateId =
    controls.candidate ||
    comparison?.candidate.run_id ||
    completed.find((r) => r.candidate_name === 'trunc16')?.id ||
    completed.find((r) => r.id !== baselineId)?.id ||
    '';
  const setBaseline = (baseline: string) => setSelection({ ...controls, context, baseline });
  const setCandidate = (candidate: string) => setSelection({ ...controls, context, candidate });
  const setMargin = (margin: string) => setSelection({ ...controls, context, margin });
  const mutation = useMutation({
    mutationFn: () => api.compare(baselineId, candidateId, Number(margin) / 100),
    onSuccess: (result) => setParams({ comparison: result.id }),
  });
  const changed =
    comparison &&
    (comparison.baseline.run_id !== baselineId ||
      comparison.candidate.run_id !== candidateId ||
      Math.abs(comparison.margin - Number(margin) / 100) > 1e-9);
  const validMargin = Number(margin) > 0 && Number(margin) <= 100;
  const titles = {
    pass: 'Within the accepted quality margin',
    fail: 'Regression exceeds the accepted margin',
    inconclusive: 'More evidence is needed',
  };
  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">EVALUATE → UNDERSTAND → DECIDE</div>
          <h1>Compare models</h1>
          <p>A release decision, backed by evidence.</p>
        </div>
        <span className="page-label">
          <GitCompareArrows size={15} />
          Paired evaluation
        </span>
      </div>
      <section className="panel comparison-controls" aria-label="Select runs to compare">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            mutation.mutate();
          }}
        >
          <label>
            Baseline run
            <select value={baselineId} onChange={(e) => setBaseline(e.target.value)} required>
              <option value="" disabled>
                Select a completed run
              </option>
              {completed.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.candidate_name} · {shortId(run.id)}
                </option>
              ))}
            </select>
          </label>
          <ArrowRight className="comparison-arrow" size={18} />
          <label>
            Candidate run
            <select value={candidateId} onChange={(e) => setCandidate(e.target.value)} required>
              <option value="" disabled>
                Select a completed run
              </option>
              {completed.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.candidate_name} · {shortId(run.id)}
                </option>
              ))}
            </select>
          </label>
          <label className="margin-control">
            Accepted loss{' '}
            <span className="input-unit">
              <input
                aria-label="Accepted loss in accuracy points"
                type="number"
                min="0.01"
                max="100"
                step="0.01"
                value={margin}
                onChange={(e) => setMargin(e.target.value)}
                required
              />
              <span>pts</span>
            </span>
          </label>
          <button
            className="button primary"
            disabled={
              !baselineId ||
              !candidateId ||
              !validMargin ||
              runs.isPending ||
              runs.isError ||
              saved.isPending ||
              mutation.isPending
            }
          >
            {mutation.isPending ? 'Computing…' : 'Compare runs'}
            <ArrowRight size={16} />
          </button>
        </form>
        <div className="control-footnote">
          <Info size={13} />
          Same benchmark. Same examples. 10,000 paired resamples.
          {changed && <strong>Selections changed — compare to update the result.</strong>}
        </div>
      </section>
      <ErrorMessage error={runs.error || saved.error || mutation.error} />
      {(runs.isPending || saved.isPending) && <Loading />}
      {!saved.isPending && !comparison && (
        <EmptyState title="Your next model decision starts here">
          Select two completed runs to inspect quality changes.{' '}
          <Link to="/candidates">Register a candidate</Link> to get started.
        </EmptyState>
      )}
      {comparison && (
        <>
          <section className={`decision-card ${comparison.verdict}`} aria-live="polite">
            <div>
              <div className="decision-top">
                <VerdictBadge verdict={comparison.verdict} />
                <span className="subtle">
                  {comparison.candidate.name} vs. {comparison.baseline.name}
                </span>
                {comparison.evidence === 'fixture' && (
                  <span className="fixture-badge">Synthetic CI fixture</span>
                )}
              </div>
              <h2>{titles[comparison.verdict]}</h2>
              <p>
                {comparison.verdict === 'pass'
                  ? 'The lower bound stays above the selected loss threshold on this benchmark.'
                  : comparison.verdict === 'fail'
                    ? 'Even the upper bound is below the selected loss threshold on this benchmark.'
                    : 'The interval crosses the loss threshold. This result is not a pass.'}
              </p>
            </div>
            <div className="decision-delta">
              <span>ACCURACY DIFFERENCE</span>
              <strong>
                {points(comparison.delta)}
                <small>pts</small>
              </strong>
              <span>
                95% CI [{points(comparison.ci95[0])}, {points(comparison.ci95[1])}]
              </span>
            </div>
          </section>
          <div className="metric-grid">
            <div className="metric-card">
              <span>Baseline accuracy</span>
              <strong>{percent(comparison.baseline.value, 2)}</strong>
              <small>{comparison.baseline.name}</small>
            </div>
            <div className="metric-card">
              <span>Candidate accuracy</span>
              <strong>{percent(comparison.candidate.value, 2)}</strong>
              <small>{comparison.candidate.name}</small>
            </div>
            <div className="metric-card">
              <span>Paired examples</span>
              <strong>{comparison.n_examples.toLocaleString()}</strong>
              <small>Pinned, identical benchmark</small>
            </div>
            <div className="metric-card">
              <span>Accepted loss</span>
              <strong>
                {(comparison.margin * 100).toFixed(2)}
                <small className="inline-unit">pts</small>
              </strong>
              <small>Policy choice, not a fitted threshold</small>
            </div>
          </div>
          <section className="panel">
            <div className="panel-heading">
              <div>
                <h2>Quality difference & uncertainty</h2>
                <p>Candidate minus baseline · accuracy points</p>
              </div>
              <span className="tag">Slices are diagnostic</span>
            </div>
            <DeltaChart comparison={comparison} />
            <div className="stats-strip">
              <span>
                McNemar p <strong className="mono">{comparison.mcnemar_p.toFixed(4)}</strong>
              </span>
              <span>
                Interval half-width{' '}
                <strong className="mono">
                  {((comparison.ci95[1] - comparison.ci95[0]) * 50).toFixed(2)} pts
                </strong>
              </span>
              <span>
                Warm batch latency{' '}
                <strong>
                  {comparison.latency_ratio === null
                    ? 'Not comparable'
                    : `${comparison.latency_ratio.toFixed(2)}× baseline · ${comparison.latency_budget_pass ? 'within budget' : 'over budget'}`}
                </strong>
              </span>
            </div>
          </section>
          {comparison.warnings.length > 0 && (
            <div className="warnings">
              <Info size={18} />
              <div>
                <strong>Interpretation notes</strong>
                {comparison.warnings.map((warning) => (
                  <p key={warning}>{warning}</p>
                ))}
              </div>
            </div>
          )}
          <FlipTable key={comparison.id} comparison={comparison} />
          <section className="ci-callout">
            <Terminal size={20} />
            <div>
              <h3>Make this decision part of your pipeline</h3>
              <code>
                mec gate --baseline {comparison.baseline.run_id} --candidate{' '}
                {comparison.candidate.run_id} --margin {comparison.margin}
              </code>
              <p>Exit codes: 0 pass · 1 fail · 2 inconclusive · 3 error</p>
            </div>
          </section>
          <div className="provenance">
            Comparison <span className="mono">{shortId(comparison.id)}</span> · seed{' '}
            {comparison.seed} · {comparison.n_resamples.toLocaleString()} resamples ·{' '}
            <Link to={`/runs/${comparison.candidate.run_id}`}>
              Inspect run provenance <ArrowRight size={12} />
            </Link>
          </div>
        </>
      )}
    </>
  );
}
