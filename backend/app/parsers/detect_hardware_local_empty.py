"""Copyable commands that let users inspect their OWN hardware. Nothing here is executed.

The backend never runs these commands; it only returns them so users can paste the output
values into the custom hardware form.
"""

LOCAL_HARDWARE_COMMANDS = {
    'linux': (
        'nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version '
        '--format=csv && nproc && free -g && df -h . && '
        "nvidia-smi topo -m"
    ),
    'gpu_only': 'nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv',
    'notes': [
        'memory.total is per GPU in MiB; divide by 1024 for GiB.',
        'nvidia-smi topo -m shows NVLink/PCIe links between GPUs (affects GPU–GPU bandwidth).',
        'Bandwidths are not reported by these commands; use vendor specs or a measured benchmark.',
    ],
}
