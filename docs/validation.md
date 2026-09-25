# Independent reviews and validation

Verified locally on 2026-10-04. This records execution evidence, not a hosted CI or production deployment claim.

## Independent correctness review

A separate reviewer inspected API contracts, dataset and candidate identity, statistical decisions, worker ownership, recovery, tests and deployment setup. Findings were fixed and re-reviewed:

- The run-detail prediction query now changes when the run reaches a terminal state, so the last persisted batch is fetched. A browser test exercises the transition.
- Opening a saved comparison restores its recorded margin and run selections. Controls wait for saved data before submitting. A browser test verifies a non-default margin.
- The Docker image creates the Hugging Face cache as the runtime user before a named volume is attached. A fresh-volume runtime check confirmed UID 10001 can write there.

The reviewer found no substantive remaining correctness issue in the final source. Runtime checks below independently verify the packaged setup.

## Independent engineering-practices review

A second reviewer inspected module boundaries, unnecessary complexity, duplication, typing, frontend state, tests, accessibility and documentation. Findings were fixed and re-reviewed:

- Strict mypy now covers all application modules, including the API and inference boundaries. Third-party untyped calls use narrow exceptions.
- The flipped-example tabs implement keyboard navigation and roving focus, with a browser test.
- The registration browser test now verifies an actual server-side duplicate conflict rather than claiming server validation from a successful submission.

No additional substantive cleanup finding remained. The design intentionally uses a PostgreSQL queue and a shared inference service rather than adding a broker or another serving system.

## Executed checks

| Check | Result |
|---|---|
| Backend tests, including coverage simulation, generation contracts and process-kill recovery | 41 passed |
| Ruff lint and formatting | Passed |
| Strict mypy across application | Passed, 31 source files |
| Alembic upgrade and schema drift check | Passed |
| Frontend ESLint, Prettier, TypeScript and production build | Passed |
| Component tests | 5 passed |
| Chromium browser workflows against development UI | 7 passed |
| Chromium browser workflows against Docker/nginx UI | 8 passed |
| Automated WCAG scans, keyboard tabs and narrow viewport | Passed within browser suite |
| Frontend dependency audit | No reported vulnerabilities |
| OpenAPI export and generated TypeScript contract | Regenerated without drift |
| Fresh Docker build and Compose startup | Passed; migration exited successfully; services healthy |
| Packaged real online inference | Positive/negative labels correct; cold and warm requests succeeded |
| Fresh-cache Docker real benchmark | Four runs completed, 872 predictions each; all three comparisons persisted |
| Packaged CLI synthetic fixtures | PASS 0, FAIL 1, INCONCLUSIVE 2, missing run 3 |
| Prometheus API and worker scrape targets | Both up |
| Provisioned Grafana dashboard | HTTP 200, Model Eval Console |

Backend tests ran against an isolated schema using the actual Alembic migration, including database immutability triggers. Tests do not truncate the demo schema. The process-kill test terminates a real worker subprocess during fixture inference, reclaims its expired lease, and completes the exact remaining prediction set. This proves recovery mechanics; it does not pretend that fixture inference is a Hugging Face benchmark.

## Measured ML evidence

The full pinned DistilBERT benchmark ran on all 872 SST-2 validation examples for four serving configurations. [The native CPU snapshot](results/local-cpu.json) records model/dataset revisions, hashes, code identity, environment, predictions-derived scores, seeded comparisons and warm latency measurements. The README reports this snapshot's results.

The documented Compose workflow was also executed with fresh database and model-cache volumes. [The container snapshot](results/compose-cpu.json) records Linux/aarch64, Python 3.12.14 and PyTorch 2.9.1+cpu. It reproduced PASS for batch 32, INCONCLUSIVE for int8, and FAIL for truncation. Int8 accuracy was 90.48%, with 13 regressions and 8 fixes; its latency ratio was 1.51. Native and container results are retained separately because numerical outputs and timings can differ. The running local console shows the container results.

The bootstrap coverage experiment uses 500 simulated datasets per regime and 10,000 resamples per interval. [Coverage results](results/coverage.json) show 95.6% and 94.4% empirical coverage in the two non-sparse regimes, but only 60.6% in the deliberately sparse improvement regime. That limitation is documented and reflected in low-disagreement warnings; it is not hidden behind a nominal 95% label.

The extension completed three native FP32 runs on the same balanced 400-review IMDb sample: DistilBERT scored 81.75%; both frozen Qwen3-0.6B prompt configurations scored 50%, with no invalid outputs and all answers negative. Both comparisons failed, and the real CLI returned exit 1 for each. [The complete snapshot](results/llm-native-cpu.json) contains all 1,200 predictions, immutable sample indices, raw answers, prompt hashes and environment details. Stored predictions were checked against the dataset indices and recomputed accuracy. Separate synthetic controls through the direct model API produced a positive sentiment answer, so the observed collapse is not explained by a hardcoded negative parser. The cause remains unproven. See [the protocol and limitations](llm-experiment.md).

The Docker FP32 attempts repeatedly restarted while the VM had little free RAM. BF16 runs were deliberately stopped after a CPU performance check; their partial records remain visible. Neither is reported as a completed benchmark. The native worker was stopped after measurement and the Docker worker restored.

## Evidence limits

The Qwen/IMDb extension received a separate follow-up correctness and practices review. No substantive correctness issue was found. Practices findings led to compatible generative registration controls and a download-free generation test of preserved instructions, review truncation, continuation parsing and greedy output limits. Additional tests verify strict invalid-answer parsing, paired API evidence with null probabilities, balanced sample reproducibility and invalid outputs in F1. Migration 0002 preserves old records and refuses a downgrade that would discard generative evidence.

- GitHub Actions is configured but has not run on a hosted repository in this workspace.
- Browser checks cover Chromium and automated accessibility rules; they are not a manual screen-reader or cross-browser certification.
- Tests emit upstream Starlette/httpx and PyTorch eager-quantization deprecation warnings. PyTorch is constrained below 2.10; migration to torchao requires a new measurement.
- Real model measurements are local CPU experiments, not production load, cloud-cost or generalization guarantees.
