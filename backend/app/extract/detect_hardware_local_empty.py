
//Optional, allows user to query GPU information from the system. This is useful for estimating available VRAM and other GPU properties.

#nvidia-smi \
#  --query-gpu=index,name,memory.total,memory.free,driver_version \
#  --format=csv