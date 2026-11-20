# Local coding-model workspaces

Install this checkout from the repository root. The core package can scaffold a workspace without installing the ML runtime.

```powershell
python -m pip install .
# Install the optional runtime when ready to train or ask questions.
python -m pip install ".[coding]"
```

A plain `pip install carvustrain` selects a published release. This checkout includes the updated entry point for a future release.

From PowerShell or another interactive terminal, run `carvustrain` to open the menu.
Select **Create workspace**, then enter the model name, workspace folder,
local base-model path, device, and context token limit. **Runtime setup** offers an explicit install of the optional runtime.
**Build test model** creates a tiny model locally and trains it on CPU without
fetching pretrained weights. To start a job, choose **Train model** and enter the step limit,
learning rate, and whether to include `train.cl`. The progress bar advances as
checkpoints are saved, and the result includes the job ID and latest checkpoint.
**Ask model** lets you pick a base model or a saved checkpoint by number.
**Memory** has numbered controls for adding, searching, viewing, and deleting notes.
CPU is the default. CUDA requires an available NVIDIA GPU
and a CUDA-enabled PyTorch installation in the same Python environment.

The base model determines the parameter count, which CarvusTrain measures at
load time. For CPU training, use a small model and a short job. In non-interactive use, the bare command still scaffolds or
reports the `.carvustrain/` workspace. You can also set `base` in
`.carvustrain/project.json` or create a named workspace with the base selected up front.

```powershell
carvustrain init-coding Test-model --base "C:/models/my-small-coding-model"
cd Test-model
python run_carvustrain.py validate --extra
carvustrain coding memory add "Use pathlib for portable filesystem paths."
carvustrain coding train --steps 2 --extra
carvustrain coding status
carvustrain coding ask "Show a Python pathlib example in a fenced Python block." --checkpoint JOB_ID --validate-code
```

Replace `JOB_ID` with the training job's `id`. Without `--checkpoint`, the answer comes from the base model. The generated launcher supports the same subcommands as `carvustrain coding` and resolves files relative to its own location. Use `carvustrain coding --workspace PATH` to select a different folder. See `carvustrain coding --help` for available operations.

## Configuration and datasets

`project.json` is the workspace registry entry. It contains the model name, type, local base path, device and token budget. Relative base paths resolve from the workspace. The supported type is `coding`. The device defaults to `cpu` and can be set to `cuda` when available. The runtime records the actual parameter count alongside completed steps and returns it with answers.

`main.brain` selects the learning and coding manifests and controls memory retrieval. `coding.brain` selects the bundled model implementation. Both are JSON data. Older workspaces with `module_call.json` continue to work.

`learn.brain` connects `basic_training: "local"`, `basic_data: "data.cl"`, `more_training: "train.cl"`, `call_code: "model.py"` and `learn: "learn_brain"`. The last two select the bundled Python modules. Dataset paths must resolve inside the workspace. Training with `--extra` appends the optional dataset to the basic examples. Steps cycle through the selected rows in order.

Both `.cl` files are UTF-8 JSON Lines. Each nonblank line is one JSON object with exactly two nonempty strings. Escape embedded newlines as `\n`.

```json
{"prompt":"How do I reverse a string in Python?","completion":"Use text[::-1]."}
```

## Training and memory

Use a small, already-downloaded, unquantized decoder-only causal model that Transformers supports locally, with config, tokenizer and safetensors weights. The backend uses a question/answer text format and completion-only loss. It does not apply model-specific chat templates. The demo-sized dataset is for checking setup. Quality depends on the base model and training examples. No model downloads are initiated. The 3.1 billion parameter notebook entry is an example.

Training uses full-weight FP32 SGD, one example per step and at most 1,000 steps per job. Memory and checkpoint disk checks are estimates. Start with a small model and a short job. Every completed step saves a full model and tokenizer under `jobs/JOB_ID/step-N`, so disk use grows with step count. The job database stores configuration and progress, and `jobs/JOB_ID/training.jsonl` records step loss.

Training runs in the invoking terminal. Use `carvustrain coding status` and `carvustrain coding cancel JOB_ID` from another terminal in the same workspace. Cancellation takes effect between steps. Ctrl+C also cancels. Run `carvustrain coding resume JOB_ID` to continue a cancelled or failed run with its saved configuration and latest completed step. Resume requires unchanged inputs and does not guarantee a bit-for-bit continuation of random training operations. After an abrupt process exit, the next train or resume command recognizes the abandoned job and releases it for recovery.

CT Basic Memory stores only explicit `memory add` entries in SQLite. Use `memory search`, `memory get`, or `memory delete` to manage them. Matching notes are supplied to inference by default. Use `--no-memory` to disable retrieval for a request.

`--validate-code` compiles the first fenced Python block in an isolated-mode subprocess with a timeout and input size limit. It returns compile diagnostics and exit status. Generated bytecode is never executed. Python isolated mode is not an OS sandbox.

## Python API

```python
from pathlib import Path
from carvustrain.coding.main_brain import MainBrain
from carvustrain.coding.schemas import CodingRequest, TrainingConfig

brain = MainBrain(Path("Test-model"))
job = brain.learning.train(TrainingConfig(steps=2))
if job["state"] == "completed":
    response = brain.ask(CodingRequest("Explain pathlib.", checkpoint=job["id"]))
    print(response.answer or response.error)
else:
    print(job["state"], job["error"])
```



## Repair an existing Windows launcher

If the launcher raises `ModuleNotFoundError: No module named 'carvustrain.cli'`, install
this checkout into the interpreter that owns the launcher. In PowerShell, compare
`(Get-Command carvustrain).Source` and `(Get-Command python).Source`. When both commands
are from the same environment, run the following from the repository root.

```powershell
python -m pip install --force-reinstall --no-deps .
carvustrain version
```
