import createClient from 'openapi-fetch';
import type { components, paths } from './schema';

export type Candidate = components['schemas']['CandidateOut'];
export type Dataset = components['schemas']['DatasetOut'];
export type Run = components['schemas']['RunOut'];
export type Comparison = components['schemas']['ComparisonOut'];
export type CandidateInput = components['schemas']['CandidateCreate'];
export type Prediction = components['schemas']['PredictionOut'];

const client = createClient<paths>({ baseUrl: '' });

function unwrap<T>(result: { data?: T; error?: unknown; response: Response }): T {
  if (!result.response.ok || result.data === undefined) {
    const error = result.error;
    const detail =
      typeof error === 'object' && error !== null && 'detail' in error
        ? String(error.detail)
        : `Request failed (${result.response.status})`;
    throw new Error(detail);
  }
  return result.data;
}

export const api = {
  comparisons: async () => unwrap(await client.GET('/v1/comparisons')),
  candidates: async () => unwrap(await client.GET('/v1/candidates')),
  datasets: async () => unwrap(await client.GET('/v1/datasets')),
  runs: async () => unwrap(await client.GET('/v1/runs')),
  run: async (id: string) =>
    unwrap(await client.GET('/v1/runs/{run_id}', { params: { path: { run_id: id } } })),
  register: async (body: CandidateInput) => unwrap(await client.POST('/v1/candidates', { body })),
  enqueue: async (candidate_id: string, dataset_id: string) =>
    unwrap(await client.POST('/v1/runs', { body: { candidate_id, dataset_id } })),
  compare: async (baseline_run_id: string, candidate_run_id: string, margin: number) =>
    unwrap(
      await client.POST('/v1/comparisons', {
        body: {
          baseline_run_id,
          candidate_run_id,
          margin,
          seed: 42,
          n_resamples: 10000,
          metric: 'accuracy',
        },
      }),
    ),
  comparison: async (id: string) =>
    unwrap(
      await client.GET('/v1/comparisons/{comparison_id}', {
        params: { path: { comparison_id: id } },
      }),
    ),
  predictions: async (id: string, offset: number, correct?: boolean, slice?: string) =>
    unwrap(
      await client.GET('/v1/runs/{run_id}/predictions', {
        params: { path: { run_id: id }, query: { offset, limit: 25, correct, slice } },
      }),
    ),
};
