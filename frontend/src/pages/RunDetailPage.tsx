import { useQuery } from '@tanstack/react-query';
import { ArrowLeft, Check, Clock, Fingerprint } from 'lucide-react';
import { useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { api } from '../api/client';
import { ErrorMessage, Loading } from '../components/Feedback';
import { StatusBadge } from '../components/VerdictBadge';
import { ModelOutput } from '../components/ModelOutput';
import { metricNumber, percent, shortId, time } from '../lib/format';

export function RunDetailPage() {
  const { id = '' } = useParams();
  const [offset, setOffset] = useState(0);
  const [filter, setFilter] = useState('all');
  const [slice, setSlice] = useState('');
  const run = useQuery({
    queryKey: ['run', id],
    queryFn: () => api.run(id),
    refetchInterval: (query) =>
      ['queued', 'running'].includes(query.state.data?.status ?? '') ? 2000 : false,
  });
  const predictions = useQuery({
    queryKey: ['predictions', id, offset, filter, slice, run.data?.status],
    queryFn: () =>
      api.predictions(
        id,
        offset,
        filter === 'all' ? undefined : filter === 'correct',
        slice || undefined,
      ),
    enabled: !!run.data,
    refetchInterval: run.data?.status === 'running' ? 2000 : false,
  });
  if (run.isPending) return <Loading />;
  if (!run.data) return <ErrorMessage error={run.error} />;
  const data = run.data;
  const accuracy = metricNumber(data.metrics, 'accuracy');
  const latency = metricNumber(data.metrics, 'latency_p95_ms');
  const weights = metricNumber(data.metrics, 'weights_bytes');
  return (
    <>
      <Link className="back-link" to="/runs">
        <ArrowLeft size={14} />
        Evaluation runs
      </Link>
      <div className="page-heading">
        <div>
          <div className="eyebrow">RUN {shortId(data.id).toUpperCase()}</div>
          <h1>{data.candidate_name}</h1>
          <p>
            {data.dataset_name} · created {time(data.created_at)}
          </p>
        </div>
        <StatusBadge status={data.status} />
      </div>
      <ErrorMessage error={run.error || predictions.error} />
      {data.error && <ErrorMessage error={new Error(data.error)} />}
      <section className="panel run-progress">
        <div>
          <Clock size={18} />
          <h2>Execution progress</h2>
          <span>
            {data.progress} / {data.total} predictions
          </span>
        </div>
        <progress
          max={data.total || 1}
          value={data.progress}
          aria-label="Evaluation prediction progress"
        />
        <div className="timeline">
          {['queued', 'running', 'succeeded'].map((status, index) => (
            <span
              className={
                data.status === status || data.status === 'succeeded' || index === 0
                  ? 'reached'
                  : ''
              }
              key={status}
            >
              <Check size={13} />
              {status === 'succeeded' ? 'completed' : status}
            </span>
          ))}
          <span>
            Attempt {data.attempts} of {data.max_attempts}
          </span>
        </div>
      </section>
      <div className="metric-grid">
        <div className="metric-card">
          <span>Accuracy</span>
          <strong>{accuracy === undefined ? '—' : percent(accuracy, 2)}</strong>
          <small>Binary sentiment classification</small>
        </div>
        <div className="metric-card">
          <span>Macro F1</span>
          <strong>{metricNumber(data.metrics, 'macro_f1')?.toFixed(4) ?? '—'}</strong>
          <small>Equal weight per class</small>
        </div>
        <div className="metric-card">
          <span>Warm batch p95</span>
          <strong>
            {latency?.toFixed(1) ?? '—'}
            <small className="inline-unit">ms</small>
          </strong>
          <small>Same-machine microbenchmark</small>
        </div>
        <div className="metric-card">
          <span>Weight storage</span>
          <strong>
            {weights === undefined ? '—' : (weights / 1024 ** 2).toFixed(1)}
            <small className="inline-unit">MiB</small>
          </strong>
          <small>
            {data.metrics?.weights_size_method === 'tensor-storage'
              ? 'Parameter tensors, excludes tokenizer'
              : 'Serialized state dict, excludes tokenizer'}
          </small>
        </div>
      </div>
      <p className="subtle">
        Invalid outputs: {metricNumber(data.metrics, 'invalid_outputs') ?? 0}. Invalid answers count
        as incorrect; no examples are dropped.
      </p>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Per-example predictions</h2>
            <p>Stored evidence used by the paired comparison.</p>
          </div>
          <div className="filters">
            <label>
              <span className="sr-only">Correctness filter</span>
              <select
                value={filter}
                onChange={(e) => {
                  setFilter(e.target.value);
                  setOffset(0);
                }}
              >
                <option value="all">All predictions</option>
                <option value="correct">Correct only</option>
                <option value="incorrect">Incorrect only</option>
              </select>
            </label>
            <label>
              <span className="sr-only">Slice filter</span>
              <select
                value={slice}
                onChange={(e) => {
                  setSlice(e.target.value);
                  setOffset(0);
                }}
              >
                <option value="">All slices</option>
                <option value="len_long">Long inputs</option>
                <option value="has_negation">Negation</option>
                <option value="has_contrast">Contrast</option>
              </select>
            </label>
          </div>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Example</th>
                <th>True label</th>
                <th>Prediction</th>
                <th>Probability / generated evidence</th>
              </tr>
            </thead>
            <tbody>
              {predictions.data?.map((row) => (
                <tr key={row.idx}>
                  <td className="prediction-text">
                    <span className="mono subtle">#{row.idx} </span>
                    {row.text}
                  </td>
                  <td>{row.label ? 'Positive' : 'Negative'}</td>
                  <td>
                    <span className={row.correct ? 'correct-text' : 'incorrect-text'}>
                      {row.pred === -1 ? 'Invalid output' : row.pred ? 'Positive' : 'Negative'}
                    </span>
                  </td>
                  <td>
                    <ModelOutput label={row.pred} probability={row.prob_pos} raw={row.raw_output} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {predictions.data?.length === 0 && (
          <p className="no-predictions">No predictions match this filter yet.</p>
        )}
        <div className="pagination">
          <span>Page {offset / 25 + 1}</span>
          <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 25))}>
            Previous
          </button>
          <button
            disabled={(predictions.data?.length ?? 0) < 25}
            onClick={() => setOffset(offset + 25)}
          >
            Next
          </button>
        </div>
      </section>
      <section className="panel provenance-panel">
        <div className="panel-heading">
          <div>
            <h2>
              <Fingerprint size={17} />
              Execution provenance
            </h2>
            <p>Code and environment recorded by the worker.</p>
          </div>
        </div>
        <dl>
          <dt>Evaluator version</dt>
          <dd className="mono">{data.code_version}</dd>
          {typeof data.metrics?.prompt_sha256 === 'string' && (
            <>
              <dt>Prompt SHA256</dt>
              <dd className="mono">{data.metrics.prompt_sha256}</dd>
            </>
          )}
          {Object.entries(data.env).map(([key, value]) => (
            <div key={key}>
              <dt>{key.replaceAll('_', ' ')}</dt>
              <dd className="mono">{String(value)}</dd>
            </div>
          ))}
        </dl>
      </section>
    </>
  );
}
