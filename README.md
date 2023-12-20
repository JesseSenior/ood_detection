# OOD Detection

## Install

```bash
pip install -e .
python -m ood_detection --mode download --val-ood-keys "0" --test-ood-keys "1"
```

## Train

```bash
python -m ood_detection --mode train --val-ood-keys "0" --test-ood-keys "1"
```

## Test

```bash
python -m ood_detection --mode eval --val-ood-keys "0" --test-ood-keys "1"
```
