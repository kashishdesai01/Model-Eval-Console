import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { ArrowRight, Box, Plus, X } from 'lucide-react';
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, type CandidateInput } from '../api/client';
import { EmptyState, ErrorMessage, Loading } from '../components/Feedback';
import { shortId } from '../lib/format';

export function CandidatesPage() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const candidates = useQuery({ queryKey: ['candidates'], queryFn: api.candidates });
  const datasets = useQuery({ queryKey: ['datasets'], queryFn: api.datasets });
  const [showForm, setShowForm] = useState(false);
  const [inferenceKind, setInferenceKind] = useState<'classifier' | 'generative'>('classifier');
  const [datasetId, setDatasetId] = useState('');
  const selectedDataset = datasetId || datasets.data?.[0]?.id || '';
  const register = useMutation({
    mutationFn: api.register,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['candidates'] });
      setShowForm(false);
    },
  });
  const enqueue = useMutation({
    mutationFn: (id: string) => api.enqueue(id, selectedDataset),
    onSuccess: (run) => {
      queryClient.invalidateQueries({ queryKey: ['runs'] });
      navigate(`/runs/${run.id}`);
    },
  });
  function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const body: CandidateInput = {
      name: String(data.get('name')),
      hf_model_id: String(data.get('model')),
      hf_revision: String(data.get('revision')),
      serving_config: {
        max_length: Number(data.get('max_length')),
        batch_size: inferenceKind === 'generative' ? 1 : Number(data.get('batch_size')),
        quantization:
          inferenceKind === 'generative'
            ? 'none'
            : data.get('quantization') === 'int8'
              ? 'int8'
              : 'none',
        positive_label_id:
          inferenceKind === 'generative' ? 1 : (Number(data.get('positive_label_id')) as 0 | 1),
        inference_kind: inferenceKind,
        dtype: inferenceKind === 'generative' ? 'bfloat16' : 'float32',
        prompt_version:
          data.get('prompt_version') === 'sentiment-few-v1'
            ? 'sentiment-few-v1'
            : 'sentiment-zero-v1',
        max_new_tokens: inferenceKind === 'generative' ? Number(data.get('max_new_tokens')) : 8,
      },
    };
    register.mutate(body);
  }
  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">MODEL REGISTRY</div>
          <h1>Candidates</h1>
          <p>Pin the weights. Version the serving configuration.</p>
        </div>
        <button className="button primary" onClick={() => setShowForm(!showForm)}>
          {showForm ? <X size={16} /> : <Plus size={16} />}
          {showForm ? 'Close form' : 'Register candidate'}
        </button>
      </div>
      <ErrorMessage error={candidates.error || datasets.error || register.error || enqueue.error} />
      {showForm && (
        <section className="panel registration">
          <h2>Register an immutable candidate</h2>
          <p>
            Binary classifiers or chat-template causal LMs. A new configuration gets a new name.
            Generative candidates use BF16 weights, batch 1 and no quantization.
          </p>
          <form onSubmit={submit}>
            <div className="form-grid">
              <label>
                Name
                <input
                  name="name"
                  required
                  maxLength={100}
                  pattern="[\w .-]+"
                  placeholder="sentiment-int8"
                />
              </label>
              <label>
                Hugging Face model ID
                <input
                  name="model"
                  required
                  placeholder="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
                />
              </label>
              <label className="full-width">
                Model revision (40-character commit SHA)
                <input
                  name="revision"
                  className="mono"
                  required
                  pattern="[0-9a-f]{40}"
                  minLength={40}
                  maxLength={40}
                  placeholder="Exact commit SHA, not main"
                />
              </label>
              <label>
                Inference type
                <select
                  name="inference_kind"
                  value={inferenceKind}
                  onChange={(event) =>
                    setInferenceKind(event.target.value as 'classifier' | 'generative')
                  }
                >
                  <option value="classifier">Binary classifier</option>
                  <option value="generative">Generative sentiment</option>
                </select>
              </label>
              <label>
                Weight precision
                <select disabled value={inferenceKind === 'generative' ? 'bfloat16' : 'float32'}>
                  <option value="float32">FP32 · classifier</option>
                  <option value="bfloat16">BF16 · generative</option>
                </select>
              </label>
              <label>
                Prompt version (generative only)
                <select name="prompt_version" disabled={inferenceKind === 'classifier'}>
                  <option value="sentiment-zero-v1">Zero-shot v1</option>
                  <option value="sentiment-few-v1">Few-shot v1 · handwritten examples</option>
                </select>
              </label>
              <label>
                Output token limit (generative only)
                <input
                  type="number"
                  name="max_new_tokens"
                  disabled={inferenceKind === 'classifier'}
                  min={1}
                  max={32}
                  defaultValue={8}
                  required
                />
              </label>
              <label>
                Maximum tokens
                <input
                  type="number"
                  name="max_length"
                  min={4}
                  max={512}
                  defaultValue={128}
                  required
                />
              </label>
              <label>
                Batch size
                <input
                  type="number"
                  name="batch_size"
                  min={1}
                  max={64}
                  key={inferenceKind}
                  defaultValue={inferenceKind === 'generative' ? 1 : 8}
                  disabled={inferenceKind === 'generative'}
                  required
                />
              </label>
              <label>
                Quantization
                <select
                  name="quantization"
                  key={inferenceKind}
                  disabled={inferenceKind === 'generative'}
                >
                  <option value="none">FP32 · no quantization</option>
                  <option value="int8">Dynamic int8 · linear layers</option>
                </select>
              </label>
              <label>
                Positive class index
                <select
                  name="positive_label_id"
                  key={inferenceKind}
                  disabled={inferenceKind === 'generative'}
                >
                  <option value="1">1 · standard SST-2 checkpoint</option>
                  <option value="0">0 · reversed label mapping</option>
                </select>
              </label>
            </div>
            <button className="button primary" disabled={register.isPending}>
              {register.isPending ? 'Registering…' : 'Register candidate'}
            </button>
          </form>
        </section>
      )}
      <div className="dataset-selector">
        <label>
          Evaluation benchmark
          <select value={selectedDataset} onChange={(event) => setDatasetId(event.target.value)}>
            <option value="" disabled>
              No benchmark ingested
            </option>
            {datasets.data?.map((dataset) => (
              <option key={dataset.id} value={dataset.id}>
                {dataset.name} · {dataset.n_examples} examples
              </option>
            ))}
          </select>
        </label>
        {datasets.data?.find((d) => d.id === selectedDataset) && (
          <span className="subtle mono">
            SHA256 {shortId(datasets.data.find((d) => d.id === selectedDataset)!.content_hash)}
          </span>
        )}
      </div>
      {candidates.isPending && <Loading />}
      {candidates.data?.length === 0 && (
        <EmptyState title="No candidates yet">
          Register a pinned sentiment checkpoint, then start an evaluation.
        </EmptyState>
      )}
      <div className="candidate-grid">
        {candidates.data?.map((candidate) => (
          <article className="panel candidate-card" key={candidate.id}>
            <div className="candidate-card-top">
              <span className="model-icon">
                <Box size={22} />
              </span>
              <span className="tag">
                {candidate.serving_config.quantization === 'int8'
                  ? 'INT8'
                  : candidate.serving_config.dtype === 'bfloat16'
                    ? 'BF16'
                    : 'FP32'}
              </span>
            </div>
            <h2>{candidate.name}</h2>
            <p className="model-id">{candidate.hf_model_id}</p>
            <dl>
              <div>
                <dt>Inference</dt>
                <dd>{candidate.serving_config.inference_kind ?? 'classifier'}</dd>
              </div>
              {candidate.serving_config.inference_kind === 'generative' && (
                <>
                  <div>
                    <dt>Prompt</dt>
                    <dd>{candidate.serving_config.prompt_version}</dd>
                  </div>
                  <div>
                    <dt>Output cap</dt>
                    <dd>{candidate.serving_config.max_new_tokens} tokens · greedy</dd>
                  </div>
                </>
              )}
              <div>
                <dt>Revision</dt>
                <dd className="mono">{shortId(candidate.hf_revision)}</dd>
              </div>
              <div>
                <dt>Max tokens</dt>
                <dd>{candidate.serving_config.max_length}</dd>
              </div>
              <div>
                <dt>Batch size</dt>
                <dd>{candidate.serving_config.batch_size}</dd>
              </div>
            </dl>
            <button
              className="button secondary"
              disabled={!selectedDataset || enqueue.isPending}
              onClick={() => enqueue.mutate(candidate.id)}
            >
              Start evaluation
              <ArrowRight size={15} />
            </button>
          </article>
        ))}
      </div>
    </>
  );
}
