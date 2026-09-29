# SubsetSynth v2.0b

Source code for a target-conditioned spine MRI synthesis model spanning nine mappings among T1-weighted (T1W), T2-weighted (T2W), and fat-suppressed (FS) images: six single-source and three dual-source mappings. The shared model uses source-availability conditioning, a Transformer latent representation, and a target-conditioned decoder.

## Repository contents

- `subsetsynth/`: model, data handling, objective, training, and evaluation code
- `scripts/`: audit, training, evaluation, and export entry points
- `configs/base.yaml`: configuration template with local-path placeholders
- `configs/formal_run_redacted.yaml`: resolved configuration of the formal run, with machine paths replaced
- `tests/`: mapping and hierarchical-objective checks
- `docs/`: architecture and mapping descriptions

The source and resolved training configuration are provided here. Imaging data, split files, generated locks, clinical readings, and trained weights are not included. To run the pipeline, replace `/path/to/local/storage/` in the configuration and generate the required data-semantic and mapping-support locks from the local dataset.

The `run_04_validation_gate_TEMPLATE.sh` comparator requires a separate legacy model and is not a standalone main-model training step. See `PROVENANCE.md` for the relationship between this public copy, the source archive, and the formal checkpoint.
