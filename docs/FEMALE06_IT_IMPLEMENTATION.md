# female_06 -> Italian voice: implementation guide

This repository now includes helper scripts to start from `female_06.wav` and launch training.

Added helpers:

- `script/append_female06_metadata` to append/update many `metadata.csv` entries from a file.
- `script/export_female06_it` to export checkpoint -> ONNX and copy matching config.

## 1) Prepare dataset

Run:

```sh
./script/prepare_female06_dataset \
  --wav-path female_06.wav \
  --dataset-dir training_data/female06_it \
  --transcript "REPLACE WITH EXACT SPOKEN TEXT"
```

This creates:

- `training_data/female06_it/audio/female_06.wav`
- `training_data/female06_it/metadata.csv`

## 2) Edit transcript

If you did not provide `--transcript`, metadata contains:

- `TODO: replace with exact transcript`

Replace it with the exact text spoken in the file.

## 3) Train (Italian phonemization)

Run:

```sh
./script/train_female06_it \
  --dataset-dir training_data/female06_it \
  --voice-name female06_it \
  --espeak-voice it \
  --batch-size 16 \
  --ckpt-path /path/to/finetune.ckpt
```

Use `--dry-run` to print command only.

## 4) Add many Italian files/transcripts

1. Copy your additional WAV files into `training_data/female06_it/audio`.
2. Fill `training_data/female06_it/entries_to_add.csv` with rows:

```csv
filename.wav|Exact transcript in Italian
```

3. Merge rows into metadata:

```sh
./script/append_female06_metadata \
  --dataset-dir training_data/female06_it \
  --entries-file training_data/female06_it/entries_to_add.csv \
  --fail-missing-audio
```

## 5) Export ONNX after training

```sh
./script/export_female06_it \
  --checkpoint /path/to/best.ckpt \
  --output-file training_data/female06_it/female06_it-medium.onnx \
  --config-path training_data/female06_it/female06_it.onnx.json
```

## 6) Important limitations

- A single WAV file is not enough for good quality.
- For usable Italian quality, add more Italian WAVs and append them to `metadata.csv`.
- Keep audio style consistent (same speaker, mic, room, gain).

## 7) Next command examples

Prepare again and overwrite metadata:

```sh
./script/prepare_female06_dataset --force --transcript "..."
```

Training without running (check args):

```sh
./script/train_female06_it --dry-run
```

Merge metadata without strict audio check:

```sh
./script/append_female06_metadata
```

Export command without running:

```sh
./script/export_female06_it --checkpoint /path/to/best.ckpt --dry-run
```
