
import json

#EXAMPLE

with open("../../data/hardware_catalog.json") as f:
    catalog = json.load(f)

gpu_id = "nvidia_a100_80gb_pcie"
gpu = catalog["gpus"][gpu_id]

bandwidth = gpu["memory"]["bandwidth_gb_per_second"]
bf16_tflops = gpu["peak_compute_tflops"]["bf16_tensor_dense"]
