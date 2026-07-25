# CarvusTrain CLI Reference Guide

CarvusTrain includes a full command-line tool named `carvustrain`.

## Commands Overview

| Command | Description |
| :--- | :--- |
| `carvustrain init` | Initialize a new workspace project |
| `carvustrain train` | Train a model from dataset or config file |
| `carvustrain predict` | Generate prediction for a prompt |
| `carvustrain chat` | Interactive multi-turn terminal chat |
| `carvustrain export` | Export model to ONNX, GGUF, BIN, JSON, CT |
| `carvustrain evaluate` | Evaluate accuracy and loss on a dataset |
| `carvustrain benchmark` | Measure throughput (tokens/sec) and latency |
| `carvustrain convert` | Convert dataset format |
| `carvustrain version` | Print version and license info |
| `carvustrain doctor` | System diagnostic health check |
| `carvustrain config` | View and validate config files |
| `carvustrain install` | Display installation instructions for optional packages |
| `carvustrain update` | Check framework update status |

## Examples

### Train a Model
```bash
carvustrain train --data dataset.txt --name Carvus --epochs 10 --batch-size 32 --output model.ct
```

### Interactive Chat Mode
```bash
carvustrain chat --model model.ct
```

### Export to ONNX / GGUF
```bash
carvustrain export --model model.ct --output model.onnx --format onnx
carvustrain export --model model.ct --output model.gguf --format gguf
```
