# SubsetSynth v2.0b Architecture / Training Freeze v1.2

This document supersedes v1.1 for implementation details while retaining v1.1 as provenance.

## Frozen scientific capability
- Vocabulary: T1 / T2 / FS only; closed set.
- 9 learned single-target primitive mappings = six `1->1` + three `2->1`.
- Three `1->2` configurations are compositional repeated target queries from one source-conditioned latent.
- No task ID or fixed source-target task embedding enters the model.

## Frozen target risk

For primitive mapping `m` and frozen supporting cohort set `C_m`:

`alpha(m,c) = 1 / (9 * |C_m|)`.

The sampler defines proposal probability:

`q(m,c) = P(cohort=c) / N_admissible_source_subsets(c)`.

Training importance weight:

`a(m,c) = alpha(m,c) / q(m,c)`.

No arbitrary rescaling is applied; therefore `sum q*a = 1` exactly.

The equal-cohort 25/25/25/25 sampler is a Monte Carlo proposal only. It is not the fairness definition. Under the target risk, expected cohort mass is A=B=42.5926%, C2=C3=7.4074%.

## Fixed accumulation normalization
- `effective_batch_size = micro_batch_size * accum_steps` is mandatory.
- Training sampler length is truncated to a complete effective-batch multiple using actual **slice-record** count.
- `drop_last=True`.
- Micro-batch loss is `weighted_numerator / effective_batch_size`.
- **Never divide by `accum_steps` again.**
- Contract D must prove gradient equivalence between full-batch and accumulated execution.

## Frozen support matrix
Training weights and validation aggregation use the same generated and approved `mapping_cohort_support_lock.json`. Empty expected validation cells fail closed; `|C_m|` may not silently adapt.

## Physics
Baseline physics is OFF and `PhysicsDecoder` is not instantiated. Physics-on is only the first prespecified diagnostic experiment after baseline NO-GO, not an automatic rescue model.

## NPY semantics
All real-data train/evaluate/export entry points require `npy_semantics_lock.json`, created only after structural audit + human visual review of modality-labelled contact sheets. Split-hash changes invalidate the lock.

## Decoder routing
Only samples with `tgt_mask[:,t]=True` are sent through target `t` decoder/RRDB.

## Gate safety
All case-metric tables must be unique on `(case_id, cohort, mapping)`, finite, and paired one-to-one. Primary bootstrap and whole-model guardrail share the same validation function and frozen support matrix.

## Required audits
- raw proposal exposure vs theoretical q;
- weighted exposure vs alpha;
- exact `sum q*a = 1`;
- accumulation gradient equivalence;
- per-cohort validation curves;
- per-cohort draw / unique-slice / repeat factor;
- VRAM/throughput after active-only decoding and physics removal.

## Test embargo
Train, early stopping, checkpoint selection, architecture decisions, and v1.8.2/v2.0b GO/NO-GO use validation only. Test model inference requires explicit unlock after freeze.
