# Five-minute demo

Seed the real benchmark before presenting. Keep the measured JSON and a recording as backup; first-time downloads should not be part of a live interview.

1. Open **Candidates**. Explain model/config identity, the commit SHA, explicit label mapping, and why a serving change deserves evaluation.
2. Open a completed **Evaluation run**. Show exact dataset size, stored predictions, evaluator/environment and measured serialized weights. If time permits, start another baseline run to show asynchronous progress.
3. **Batch 32 vs baseline:** PASS with identical correctness. Explain the zero-disagreement warning and why the tool does not claim general population equivalence. Latency is deliberately not compared across different batch sizes.
4. **Truncation vs baseline:** FAIL, −9.86 accuracy points, with a −20-point long-input slice. Open actual regressions and compare both probabilities. Diagnostic slices explain behavior without multiplying release gates.
5. **Int8 vs baseline:** INCONCLUSIVE at a one-point margin. Show the regressions and fixes (native snapshot: 16/9; container snapshot: 13/8), the interval crossing the margin, smaller weights and slower measured local latency. Say: “The proposed optimization does not earn promotion from this evidence.” Use the displayed environment's actual measurements throughout.
6. Copy the displayed **CLI command**. Demonstrate its exit code. Explain the separate inconclusive/error states and that the pipeline owner must define policy.
7. Open **Grafana**. Show queue depth, attempt outcomes and bounded-label inference latency. Explain the lease/claim-token recovery test.

For the extended ML demo, open the completed IMDb comparison. Show DistilBERT's 81.75% against the frozen Qwen zero-shot/few-shot configurations at 50%, inspect raw generated answers, and explain the failed gate. Discuss the different tokenizer exposure, greedy decoding and small timing sample before interpreting the result. [The experiment guide](llm-experiment.md) contains the exact protocol and full evidence; do not present these failed configurations as a general ranking of model capability.

## Defensible discussion points

- **Why pairing?** Models score the same examples; differences preserve shared example difficulty instead of pretending two independent samples.
- **Why a margin?** A deployment policy tolerates an explicit loss; statistical difference from zero answers another question.
- **Why a third verdict?** Insufficient precision is a useful product result, not permission to guess.
- **Why Postgres?** Few runs, transactional registry/results, one operational dependency. A broker adds little at this scale.
- **What if a worker pauses?** Leases allow recovery; claim tokens fence stale persistence. Unique predictions prevent duplicate rows. Computation can repeat.
- **Why not more model features?** The project proves a complete evaluation/platform workflow. Adding unrelated LLM features would weaken its evidence and polish under the time constraint.
- **What would scale change?** Partition work, separate inference pools, store large prediction sets outside relational rows, add access control and team-level evaluation policies. Those are future architecture choices, not claims about this deployment.

## Resume wording supported by the implementation

- Built a React/TypeScript and FastAPI model evaluation console integrating pinned Hugging Face/PyTorch sentiment inference, reproducible benchmarks, and a CLI quality gate.
- Implemented paired-bootstrap non-inferiority comparisons with explicit inconclusive outcomes, exact McNemar diagnostics, and empirical coverage tests including documented sparse-sample limitations.
- Engineered a PostgreSQL evaluation queue with renewable leases, stale-worker fencing, idempotent prediction batches, and tested process-kill recovery.
- Detected a measured 9.86-point truncation regression and investigated its long-input failures; measured an int8 configuration's smaller weights, inconclusive quality and worse local latency.
- Extended the platform to compare a specialized classifier with a pinned small generative model using versioned prompts, strict output validation and a reproducible balanced IMDb sample; retained all raw outputs and rejected two failing configurations.

Use only the bullets appropriate to the resume's available space. Do not describe this as an Apple deployment, production-scale service, or generally faster int8 serving.
