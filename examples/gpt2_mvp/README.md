# GPT-2 MVP examples

Selectable from the workload page ("Load example") so testers do not need their own files.

| Set | Files | Purpose |
| --- | --- | --- |
| `gpt2_supported` | config.json, training_args.json, data_config.json, train.py | GPT-2 small (openai-community/gpt2 config fields), supported v1 settings |
| `gpt2_conflicts` | config.json, training_args_unsupported.json, data_config.json | Requests FP16, gradient checkpointing and LoRA to demonstrate conflict handling |

`gpt2_estimator_input.json` is the analytical prototype's CLI input (`python -m app.services.estimators.estimate_gpt2 ...`, run from `backend/`); its hardware block is a synthetic illustrative GPU.
No file here is executed by the app.
