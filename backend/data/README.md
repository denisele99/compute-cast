hardware_catalog.json
// Verified hardware specifications. This file is used to provide accurate specifications for GPUs in the system.

// {
//   "gpu_id": "example_gpu",
//   "name": "Example GPU",
//   "vram_gb": 80,
//   "source_url": "https://example.com/specifications",
//   "last_verified": "2026-10-27"
// }


instance_types.json
This is currently optional if I want to take into account CPU, ram, SSD capacity, etc. Especially if I want to implement something like ZeRO-infinity that uses CPU and NXME
// Catalog of cloud instance configurations.
// Each entry defines the provider, GPU ID/count, vCPUs, CPU RAM,
// and included storage for an instance type.
// GPU IDs reference hardware.json; provider prices belong in pricing.json.
// Used to compare resource capacity across deployment options.


fetched data 10/27/26
nebius_spot_prices
