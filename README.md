# Retrieval model and training

Training code and model definition for the GreenHyperSpectra cross-source trait experiments.

This repository contains the Transformer, the PROSPECT-PRO/4SAIL emulator, the loss, and the training scripts. It does not contain the downloader, the reflectance quality-control filter, manuscript files, or trained weights.

Put `labeled_all.parquet` at `data/raw_hf/labeled_all.parquet` before training. A local emulator checkpoint, when one already exists, is read from `data/rtm_library/`.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python scripts/run_v2_ladder.py
```

`scripts/run_real_experiments.py` is the broader experiment driver. `scripts/smoke_test.py` checks that the model builds.

```bash
pytest tests/test_model.py tests/test_rtm.py tests/test_spectra.py
```
