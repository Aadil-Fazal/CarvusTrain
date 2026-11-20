# The CarvusTrain 4 Handbook

<img src="../logo.png" width="96" alt="CarvusTrain logo">

**From a first CPU run to a reproducible coding-model experiment**

CarvusTrain gives each coding model a local workspace for its base model, examples,
configuration, memory and checkpoints. This handbook follows a workspace from its
first CPU test through a repeatable fine-tuning run.

Start with chapters 1 and 2 to verify your installation. Chapters 3–8 cover
building and evaluating your own model. Chapters 9–13 cover automation,
storage, a worked experiment, troubleshooting, and command reference.

## Contents

1. [Install and open the terminal](#1-install-and-open-the-terminal)
2. [Build your first test model](#2-build-your-first-test-model)
3. [Plan a coding workspace](#3-plan-a-coding-workspace)
4. [Write teaching examples](#4-write-teaching-examples)
5. [Configure the brain manifests](#5-configure-the-brain-manifests)
6. [Train and manage checkpoints](#6-train-and-manage-checkpoints)
7. [Ask questions and evaluate results](#7-ask-questions-and-evaluate-results)
8. [Use CT Basic Memory](#8-use-ct-basic-memory)
9. [Automate a run in Python](#9-automate-a-run-in-python)
10. [Understand the architecture and storage](#10-understand-the-architecture-and-storage)
11. [Run a complete experiment](#11-run-a-complete-experiment)
12. [Troubleshoot a workspace](#12-troubleshoot-a-workspace)
13. [Command and menu reference](#13-command-and-menu-reference)

## 1. Install and open the terminal

Use Python 3.10 or newer. From the repository root, run
`python -m pip install .` and then `carvustrain` in PowerShell, a Linux
terminal or a macOS terminal. For a dedicated environment on Windows, run
`python -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1`
before installing. On Linux or macOS, the activation command is
`source .venv/bin/activate`.

The core installation lets you create workspaces and validate data. The coding
runtime supplies PyTorch, Transformers, safetensors and psutil. Install it from
**Runtime setup** in the menu or use `python -m pip install ".[coding]"`.
Restart the menu after installation so subsequent actions import the newly
installed modules.

At startup, the terminal draws the CarvusTrain logo beside the title where
space permits. Cyan identifies workspace actions and checkpoint progress,
magenta identifies model actions, and green, yellow, and red distinguish completed,
cancelled, and failed training. The same states are always shown in words.
Set `NO_COLOR` to disable the palette. If the terminal encoding cannot
represent block characters, the logo uses ASCII. Narrow terminals stack the menu.
Use `carvustrain ui --color always` to explicitly enable colors in an
ANSI-capable terminal, or `carvustrain ui --color never` for a plain session.
The logo draws row by row on startup, and the training bar advances as
checkpoints save. Set `CARVUSTRAIN_NO_MOTION=1` to draw the logo immediately
and report each saved step on a separate line.

In a redirected or non-interactive session, use explicit subcommands for
repeatable behavior. A bare non-interactive invocation scaffolds or describes
the default workspace.

## 2. Build your first test model

Open the menu and select **Build test model**. Press Enter to accept the
default folder, `CarvusTrain-test`, or enter a new folder. The action generates
a tokenizer and tiny GPT-2 architecture locally, initializes weights with a
fixed seed and runs one real optimizer update on CPU. No pretrained weights are
downloaded. Inspect `base/` for the original weights and
`jobs/JOB_ID/step-1/` for the saved weights.

The training example teaches a Python slicing idiom, and the vocabulary is
small enough to make this run useful as a setup check. Generated replies from
this test model may contain repeated or irrelevant tokens. Use it to verify
training, checkpoint storage and loading. The menu keeps the test workspace
available for inspection. The automated test uses a disposable workspace and
cleans it up after verification.

The result screen reports the job ID, status and checkpoint path. Inspect
the same job independently from the workspace directory.

```powershell
cd CarvusTrain-test
carvustrain coding status
carvustrain coding status JOB_ID
```

Replace `JOB_ID` with the full ID in the result. Expect `completed`, one
saved step, and a checkpoint ending in `step-1`.

To run the same workflow from a script, use
`carvustrain coding test-model CarvusTrain-test` with a new directory.
The command accepts `--steps` and `--learning-rate` for experiments with
the toy dataset and prints the job result as JSON.

Select **Ask model** to see the base and available checkpoints. A checkpoint
entry shows how many training steps were saved. Press Enter to select the
latest saved checkpoint when one is available.

## 3. Plan a coding workspace

Decide what you want the model to learn before choosing a base. A narrowly
scoped experiment could teach the naming conventions of one internal library
or recommend specific standard-library idioms. A pretrained coding model
usually offers a better starting point for those experiments than random
weights.

Choose **Create workspace** and enter a display name and a new folder. Select
an already available decoder-only base model. The directory needs
`config.json`, compatible tokenizer files and unquantized safetensors
weights. You can defer choosing a model and use **Model settings** later.
The model itself defines the parameter count. The context setting controls the
token budget, with an upper bound from the model architecture.

Start with CPU and a small model. For CUDA, select a system with a supported
NVIDIA GPU, a compatible driver and a CUDA-enabled PyTorch build. Within the
same environment, `python -c "import torch; print(torch.cuda.is_available())"`
checks detection. CarvusTrain estimates memory and checkpoint disk needs
before fine-tuning. Lowering the context limit can reduce activation memory.
Choosing a smaller model reduces both model memory and checkpoint size.

## 4. Write teaching examples

Both `.cl` files use UTF-8 JSON Lines. Put one JSON object on each nonblank
line with exactly two non-empty strings, `prompt` and `completion`. Use
`\n` within a JSON string to represent a newline.

```jsonl
{"prompt":"Join a directory and filename with pathlib.","completion":"Use Path(directory) / filename."}
{"prompt":"Read a UTF-8 file with pathlib.","completion":"Use Path(filename).read_text(encoding=\"utf-8\")."}
```

Use `data.cl` for the basic examples and `train.cl` for supplemental examples.
The optional extra mode appends the supplemental rows. The Learning Brain
cycles through the selected rows in order, so a short run may see only the
first few rows. Keep a separate review set of prompts that never enters either
file. Include altered names, variations on wording and edge cases in the
review set.

Validation checks the structure of every selected row. Backend tokenization
checks whether each example fits the model context at training time.
If a training example is too long, shorten it or adjust the context limit.
For repeatable results, record the dataset version along with the base and
training settings.

## 5. Configure the brain manifests

A coding workspace contains three JSON manifests with the `.brain` extension.

| File | What it controls |
| --- | --- |
| `main.brain` | Routes to the bundled learning and coding components and sets the default memory policy |
| `learn.brain` | Selects the local training mode and names the basic and additional datasets |
| `coding.brain` | Selects the bundled `model.py` coding backend |

Treat these manifests as data. Their component selectors accept known
implementations. Dataset filenames resolve inside the workspace, and an
incompatible selector or missing selected dataset produces a validation error.
Older workspaces with `module_call.json` can continue using that configuration.

The separate `project.json` stores the model name, base path, device and context
budget. Relative base paths resolve from the workspace. Refer to the
[coding workspace guide](coding_workspace.md) for complete configuration examples.

## 6. Train and manage checkpoints

Start with a bounded run in **Train model**. Enter the count of optimizer steps,
a learning rate and whether the job should use the supplemental dataset.
Training uses completion-only causal loss, so the target tokens are drawn
from the completion and prompt tokens are masked from the loss. The backend
uses batch size one and full-weight FP32 SGD. CarvusTrain saves a full model
and tokenizer after each completed step. The bar advances as checkpoints are
saved.

One terminal owns the foreground training process. From a second terminal in
the workspace, run `carvustrain coding status` to inspect progress and
`carvustrain coding cancel JOB_ID` to request cancellation. Cancellation is
checked between steps. Ctrl+C also cancels a foreground run. Resume with
`carvustrain coding resume JOB_ID` when the project, base and dataset inputs
are unchanged. Resume uses the saved weights and original job configuration.
SGD in this backend uses no momentum. A resumed job can draw different random
training samples from model-internal operations such as dropout.

Preserve checkpoints you need, especially before comparing experiments. A full
checkpoint per step can consume substantial disk space. Keep job state,
configuration and model files together when backing up a workspace.

## 7. Ask questions and evaluate results

From **Ask model**, select a model and enter a prompt. Ask for a fenced Python
block if you want the syntax checker to identify code. The answer screen labels
the model and source. CLI callers get a JSON record with the answer, extracted
code, matched notes and any validation result or error.

Evaluate the base and fine-tuned checkpoint against the same review prompts.
For each answer, compare whether it follows the requested interface,
preserves important behavior and handles edge cases. A successful syntax
check tells you that Python compiled the extracted block. Semantic correctness
requires further inspection and appropriate tests in your own environment.
CarvusTrain's validator compiles source in a timed subprocess and leaves the
generated bytecode unexecuted.

## 8. Use CT Basic Memory

Memory lets you add a workspace-specific note immediately. For example,
add `Use pathlib for all local file paths in this project.` under **Memory**.
A related prompt can retrieve that note and send it to the model as context.
Search uses word overlap, so use the same identifiers and terms developers
are likely to put in questions.

Add, search, get and delete operate on the workspace database. Requests and
answers are saved only when you explicitly add them as notes. Notes do not
update model weights. Use `--no-memory` on a CLI ask to compare output without
retrieval, or set the memory option in `main.brain` to false for the workspace.

## 9. Automate a run in Python

```python
from pathlib import Path

from carvustrain.coding.main_brain import MainBrain
from carvustrain.coding.schemas import CodingRequest, TrainingConfig

brain = MainBrain(Path("Test-model"))
print(f"Examples available: {len(brain.learning.validate(extra=True))}")
job = brain.learning.train(TrainingConfig(steps=2, extra=True))
if job["state"] != "completed":
    raise RuntimeError(job["error"] or job["state"])

reply = brain.ask(CodingRequest("Show a pathlib example.", checkpoint=job["id"]))
if reply.error:
    raise RuntimeError(reply.error)
print(reply.answer)
```

For a script that manages several models, create one `MainBrain` per
workspace. Handle failed training jobs before asking their checkpoints.
The existing library interfaces for datasets, tokenizers and models are
described in the [API reference](api_reference.md).

## 10. Understand the architecture and storage

The Main Brain loads model configuration, selects the base or checkpoint,
and passes optional SQLite references to the coding model. It can then pass
fenced Python source to a syntax checker. The Learning Brain reads the
configured datasets, updates the model and records completed steps. It
serializes training jobs within the same workspace while allowing status and
cancellation requests from another process.

```text
Test-model/
  project.json          Identity, model path and hardware selection
  main.brain            Routing and retrieval settings
  learn.brain           Dataset selection
  coding.brain          Model implementation selection
  data.cl              Basic examples
  train.cl             Supplemental examples
  run_carvustrain.py    Workspace-local CLI entry point
  workspace.sqlite3    Explicit memory and job metadata
  jobs/
    JOB_ID/
      training.jsonl   Recorded loss per step
      step-1/         Model weights and tokenizer
```

The model weights, memory and dataset each have a separate lifecycle. Backing
up only the checkpoint preserves the weights and tokenizer. Backing up the
workspace also preserves training configuration, job metadata and notes.

## 11. Run a complete experiment

This exercise uses a compatible local base to teach a filesystem convention.
The question under test is whether a short fine-tuning run makes the model
use `pathlib` consistently while preserving correct file handling.

### Establish a baseline

Replace the base path below with your local model directory. Use a new
workspace so the experiment has its own notes, checkpoints, and logs.

```powershell
carvustrain init-coding Paths-lab --base "C:/models/local-base"
cd Paths-lab
carvustrain coding info
carvustrain coding ask "Write a Python function that reads a UTF-8 file using pathlib. Return a fenced Python block." --no-memory --validate-code
```

Save the response and diagnostics for comparison. Keep this prompt out of
the training data. Prepare other review prompts that cover a missing file,
a path with spaces, and a caller that supplies a `Path` object.

### Train a small run

Replace `data.cl` and `train.cl` with your teaching examples. Put at least
one row in each file and use related examples such as joining a directory
and filename or checking whether a path exists. Chapter 4 shows the JSONL
format. Choose a step count that covers the rows you want the model to see.

```powershell
carvustrain coding validate --extra
carvustrain coding train --steps 2 --learning-rate 0.0001 --extra
carvustrain coding status
```

Record the full job ID and confirm the result is `completed`. Open the
job's `training.jsonl` to inspect the reported loss and use the checkpoint
path from `status` to locate the saved weights.

### Compare and record

```powershell
carvustrain coding ask "Write a Python function that reads a UTF-8 file using pathlib. Return a fenced Python block." --checkpoint JOB_ID --no-memory --validate-code
```

Run each held-out prompt against the base and the checkpoint. Record the
base model, dataset revision, step count, learning rate, device, context limit,
and job ID alongside your results.

| Review question | What to look for |
| --- | --- |
| Reads UTF-8 | The answer sets the encoding and returns the requested content |
| Handles a missing file | The answer follows the caller's requested error policy |
| Accepts paths with spaces | The answer passes a path object directly to the file API |
| Accepts a `Path` argument | The answer preserves the requested function interface |

Check syntax, then read the code for behavior. Repeat the comparison with
the same inputs when changing one training setting. Run a separate comparison
with memory enabled if you want to measure the contribution of retrieved notes.

## 12. Troubleshoot a workspace

| Symptom | Check |
| --- | --- |
| Launcher raises `ModuleNotFoundError` | Reinstall this checkout into the Python environment that owns the launcher. On PowerShell, compare `Get-Command python` and `Get-Command carvustrain`. |
| Menu is missing | Open the command directly in an interactive terminal with input and output attached. |
| Base model is missing | Use **Model settings** to select a local folder with config, tokenizer and safetensors files. |
| CUDA is unavailable | Check the PyTorch build and driver. CPU remains available for compatible small models. |
| Memory or disk estimate exceeds available resources | Use a smaller base model, fewer steps, or a shorter context as appropriate. |
| No checkpoint is listed | Verify at least one step finished and that you opened the correct workspace. |
| Generated code fails validation | Read the returned diagnostics and review the extracted source. |
| Logo appears as hashes or colors are absent | Check your terminal encoding, ANSI support, and whether `NO_COLOR` is set. |
| Resume rejects the job | Confirm the job is cancelled or failed and the base, dataset, and project settings match the original run. |

Use `carvustrain version` to confirm the installed version and
`carvustrain coding --help` for scriptable options. See the
[developer guide](developer_guide.md) for repository conventions and tests.

## 13. Command and menu reference

Run `carvustrain` or `carvustrain ui` to open the dashboard in an interactive terminal. For
automation, place `--workspace PATH` immediately after `coding` when the
workspace is outside your current directory.

| Menu | Command or equivalent |
| --- | --- |
| `1` Create workspace | `carvustrain init-coding DIRECTORY --name NAME --base PATH` |
| `2` Open workspace | `carvustrain coding --workspace PATH info` shows project details |
| `3` Model settings | Settings are stored in `project.json` |
| `4` Validate data | `carvustrain coding validate [--extra]` |
| `5` Train model | `carvustrain coding train --steps N --learning-rate RATE [--extra]` |
| `6` Build test model | `carvustrain coding test-model DIRECTORY` |
| `7` Ask model | `carvustrain coding ask "QUESTION" [--checkpoint JOB_ID] [--validate-code] [--no-memory]` |
| `8` Memory | `carvustrain coding memory add\|search\|get\|delete` with the required text or note ID |
| `9` Runtime setup | `python -m pip install ".[coding]"` from the checkout |
| `0` Exit | Close the interactive menu |

Use `carvustrain coding status [JOB_ID]` to inspect a run and
`carvustrain coding cancel JOB_ID` or `carvustrain coding resume JOB_ID`
to manage it. The workspace-local `run_carvustrain.py` accepts the same
workspace commands and resolves the workspace from its own location.

See the [coding workspace guide](coding_workspace.md) for configuration
examples, the [API reference](api_reference.md) for the framework's Python
interfaces, and the [developer guide](developer_guide.md) for repository work.
