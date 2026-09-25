# Design decisions

## A complete, bounded product

The primary workflow is candidate registration → asynchronous run → paired comparison → investigation → pipeline gate. The comparison is the main product surface. Online `/predict` demonstrates real inference integration through the same tokenizer/model/config code as evaluations.

The current scope supports binary sequence-classification sentiment checkpoints and chat-template causal LMs with safetensors weights. `positive_label_id` explicitly maps classifier logits to dataset labels. Generative inference uses versioned sentiment prompts, greedy generation and strict answer parsing; invalid labels are stored as -1 with raw evidence and null probability. Remote model code is disabled. Registering a candidate pins an identity; loading and compatibility errors appear when it is evaluated or served.

The seed uses the standalone `stanfordnlp/sst2` dataset with config `default`, equivalent to the SST-2 task in GLUE. This avoids a multipurpose GLUE registry while still recording dataset configuration explicitly. The manifest pins exact commits and never falls back silently to `main`.

## Ownership of code

FastAPI routers validate HTTP input and translate errors. Dataset and comparison services own persistence workflows. The inference service owns tokenization, binary label mapping, CPU inference, quantization and cache eviction. The worker runner owns evaluation orchestration. Queue functions own transactions and lease ownership. The statistics package imports no model or database code.

React pages own product workflows; shared components own the interval visualization, example investigation, and feedback. TanStack Query owns server-state caching and polling. Form selections stay local; a persisted comparison URL restores the chosen runs and margin. Changing selections does not relabel the existing verdict: the UI explicitly asks the user to recompute.

No generic repository layer, broker abstraction, plugin evaluator framework, or custom state-management library is needed at this scale.

## Queue protocol

1. A short transaction selects the oldest queued/expired row with `FOR UPDATE SKIP LOCKED`, increments attempts, assigns a random claim token and a deadline using database time.
2. A separate heartbeat thread renews the lease at one-third of its duration. It uses its own session.
3. Inference runs outside database transactions.
4. Each prediction batch locks the run row, checks status/token and unexpired ownership using database time, then inserts predictions within that same transaction.
5. A unique `(run_id, example_idx)` key makes a repeated batch idempotent. Previously persisted examples are skipped on resume.
6. Completion verifies exact prediction IDs and writes terminal state under the same ownership fence.
7. A stale worker's writes, heartbeats and terminal transitions are rejected. Exhausted expired leases become failed runs.

This provides at-least-once computation and idempotent persistence. Leases do not prevent overlapping CPU work when a worker pauses; claim-token fencing prevents stale persistence. Transactions are short, so heartbeat contention is bounded. An unavailable database can stop a worker; Compose restarts it and lease recovery resumes eligible work.

The process-kill test executes the real runner with a slow deterministic test inference service, kills it after persisted batches, reclaims its lease, and completes exactly the expected IDs. It isolates queue/runner reliability from external downloads. Separate tests exercise actual PyTorch int8 operators and model inference is measured by the real demo.

## Provenance and immutability

Candidates, datasets and examples have no update API, and database triggers reject updates. A new configuration creates a new candidate. The content hash is SHA-256 of ordered, UTF-8 JSON `[idx,text,label]` rows separated by newlines. Dataset identity includes Hub ID, config, split and commit revision.

Workers record evaluator code version and runtime before predictions. A partially completed run resumes only under exactly matching provenance. If `MEC_CODE_VERSION` is supplied, it should be the image/build Git SHA; otherwise the evaluator hashes the application source files. The measured snapshot retains the actual version from its execution; later source edits do not rewrite historical results.

The configured thread count and selected quantization engine are process-wide. The bounded cache holds at most two model artifacts and serializes online calls/eviction; a generative cold load clears existing artifacts to reduce memory pressure. Cache identity is model ID, revision, inference kind and quantization; max length, batch size, prompt version and label mapping apply on every inference call.

## Latency and efficiency

Warm latency includes tokenization and CPU forward inference (or prompt preparation and generation) but excludes model loading, network transfer, DB writes, and queue wait. A fixed, evenly spaced subset of at most 16 benchmark examples is batched using the candidate configuration. After five warm-up batches, three repeats produce three p95 values; the reported value is their median. The original classifier snapshots used 64 examples, 20 warm-ups and five repeats, as recorded in their provenance. Small microbenchmark tails are noisy, so this is a local budget diagnostic.

The ratio is available only when environments, sample indices, and batch sizes match. Batch 8 versus batch 32 latency is therefore not compared as an equivalent-request budget. Different tokenizers, prompts and maximum lengths remain part of the candidate's serving behavior. Classifier weight size measures serialized state-dict bytes; LLM size measures parameter tensor bytes without duplicating multi-GB weights in RAM. The method is recorded; neither is total RSS or cost.

The UI's quality verdict and 1.10× latency budget are separate. The measured int8 result has smaller weights but worse latency on the tested engine. No cloud-cost or general throughput claim follows.

## API and operations

- `/v1` routes expose explicit Pydantic request/response schemas. OpenAPI-generated frontend types make contract drift reviewable.
- Validation/domain failures use `application/problem+json`; the frontend preserves problem details.
- Cursor-based candidate listing and bounded run lists keep this small registry simple. Predictions use filtered, offset-based pagination.
- `/healthz` tests process liveness; `/readyz` checks DB access and the current migration revision. It does not promise a cached or downloadable model.
- Metrics use bounded labels (quantization/status) rather than unbounded candidate UUIDs. API and worker are scraped separately; the API queries queue depth from shared state.
- Images run without root. Services bind host ports to loopback. Compose provisions the database, migration, shared writable model cache, API, worker, UI and dashboards.

## Official guidance used

- [FastAPI: bigger applications](https://fastapi.tiangolo.com/tutorial/bigger-applications/) — routers and shared dependencies.
- [FastAPI: lifespan](https://fastapi.tiangolo.com/advanced/events/) — initialize the inference service at process startup.
- [React: you might not need an effect](https://react.dev/learn/you-might-not-need-an-effect) — derive saved controls instead of synchronizing duplicate state with effects.
- [TanStack Query defaults](https://tanstack.com/query/latest/docs/framework/react/guides/important-defaults) — deliberate stale time, bounded retries, terminal polling.
- [PostgreSQL SELECT](https://www.postgresql.org/docs/16/sql-select.html) and [explicit locking](https://www.postgresql.org/docs/18/explicit-locking.html) — short row-lock transactions and queue claim semantics.
- [PyTorch dynamic quantization](https://docs.pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html) — CPU int8 linear layers; pinned below 2.10 because this eager API is deprecated.
- [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) — empirical paired resampling and interval interpretation.
