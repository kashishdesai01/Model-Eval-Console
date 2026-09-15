import { useQuery } from '@tanstack/react-query';
import { ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import { api } from '../api/client';
import { EmptyState, ErrorMessage, Loading } from '../components/Feedback';
import { StatusBadge } from '../components/VerdictBadge';
import { metricNumber, percent, shortId, time } from '../lib/format';

export function RunsPage() {
  const runs = useQuery({
    queryKey: ['runs'],
    queryFn: api.runs,
    refetchInterval: (query) =>
      query.state.data?.some((run) => run.status === 'queued' || run.status === 'running')
        ? 2000
        : false,
  });
  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">EVALUATION HISTORY</div>
          <h1>Evaluation runs</h1>
          <p>Every result has a model, benchmark, and execution history.</p>
        </div>
        <Link to="/candidates" className="button primary">
          Start an evaluation
          <ArrowRight size={16} />
        </Link>
      </div>
      <ErrorMessage error={runs.error} />
      {runs.isPending && <Loading />}
      {runs.data?.length === 0 && (
        <EmptyState title="No evaluations yet">Start a run from the candidate registry.</EmptyState>
      )}
      {!!runs.data?.length && (
        <section className="panel table-scroll">
          <table>
            <thead>
              <tr>
                <th>Candidate / run</th>
                <th>Status</th>
                <th>Benchmark</th>
                <th>Accuracy</th>
                <th>Created</th>
                <th>
                  <span className="sr-only">Open</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {runs.data.map((run) => {
                const accuracy = metricNumber(run.metrics, 'accuracy');
                return (
                  <tr key={run.id}>
                    <td>
                      <Link className="row-title" to={`/runs/${run.id}`}>
                        {run.candidate_name}
                      </Link>
                      <span className="subtle mono block">{shortId(run.id)}</span>
                    </td>
                    <td>
                      <StatusBadge status={run.status} />
                    </td>
                    <td>{run.dataset_name}</td>
                    <td className="mono">{accuracy === undefined ? '—' : percent(accuracy, 2)}</td>
                    <td className="subtle">{time(run.created_at)}</td>
                    <td>
                      <Link
                        to={`/runs/${run.id}`}
                        aria-label={`Open ${run.candidate_name} run ${shortId(run.id)}`}
                      >
                        <ArrowRight size={17} />
                      </Link>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      )}
    </>
  );
}
