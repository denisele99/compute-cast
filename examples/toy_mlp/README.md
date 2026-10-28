# Small PyTorch example

A synthetic regression task: `Linear(100,64) → ReLU → Linear(64,1)`, **6,529
trainable parameters**, 64 examples, batch size 8, 2 epochs, FP32 AdamW, one GPU,
no sharding, checkpointing or gradient accumulation. No datasets or model weights
need downloading. Existing examples outside this directory are unchanged.

From the repository root, using an environment with PyTorch installed:

```bash
python examples/toy_mlp/train.py --device cpu
```

Install PyTorch using the command for your CPU/CUDA platform from
https://pytorch.org/get-started/locally/. PyTorch is required by this example,
not by the planner's metadata-only API. Importing either script does not train.

## Test the planner upload

Upload these four files together:

- `model.py`
- `train.py`
- `model_config.json`
- `train_config.json` **or** `train_config.yaml` (equivalent values)

Review the extracted fields. Expected values: 64 examples, batch size 8, 2 epochs,
6,529 parameters, FP32, AdamW and one active GPU. This is **16 microbatches and
16 optimizer updates**. Leave runtime, measured VRAM and pricing empty unless you
have matching real measurements/hardware. Parameter count alone does not predict
full activation memory, so unknown peak/runtime/cost is expected—not a failure.
Synthetic CPU training loss is not a GPU benchmark.

The script reads the JSON training config when executed. The YAML version is an
upload/parser example: keep it synchronized if editing the JSON.

## Local CUDA benchmark

Create a sample from the same configs:

```bash
python examples/toy_mlp/train.py --write-sample /tmp/toy_mlp_sample.pt
```

Use a writable sample path on Windows (for example `toy_mlp_sample.pt`) and
substitute that same path below. In the planner, select custom hardware matching
your CUDA GPU, use FP32/single GPU, upload/review the four files, and download
`analysis_config.json` from the benchmark section. Keep its settings equal to
these example configs. Then, from the repository root:

```bash
python backend/scripts/run_benchmark.py --model examples/toy_mlp/model.py --train examples/toy_mlp/train.py --data_sample /tmp/toy_mlp_sample.pt --config analysis_config.json --output benchmark.json
```

Import `benchmark.json` into the same analysis and rerun the estimate. The adapter
includes batch transfer, forward/loss/backward and AdamW work; it does not include
initial model/data setup. The tiny network is for checking the integration, not
comparing GPU performance. GPU runs can cost money on rented machines; this
example never provisions cloud hardware.
