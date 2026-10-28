export type Training = {
  examples: number;
  batch_size: number;
  epochs: number;
  sequence_length: number;
  gradient_accumulation_steps: number;
  script_gpu_count: number;
  strategy: string;
  seconds_per_batch: number | null;
  measured_peak_vram_gib: number | null;
  precision: string;
  optimizer: string;
  assume_data_ready: boolean;
};
export type ModelInfo = {
  total_params: number | null;
  activation_gib: number | null;
  auxiliary_gib: number | null;
  flops_per_microbatch: number | null;
  hbm_bytes_per_microbatch: number | null;
  config: Record<string, unknown>;
};
export type Hardware = {
  option_id: string;
  name: string;
  vram_gib: number;
  memory_limit_gib: number | null;
  available_gpus: number;
  price_per_gpu_hour: number | null;
  cpu_ram_gib: number | null;
  cpu_cores: number | null;
  nvme_gib: number | null;
  ssd_gib: number | null;
  cpu_specs: string;
  compute_tflops: number | null;
  fp32_tflops: number | null;
  memory_bandwidth_gbps: number | null;
  gpu_link_gbps: number | null;
};
export type Analysis = {
  framework: "pytorch";
  training: Training;
  model: ModelInfo;
  hardware: Hardware;
};
export type Offer = {
  id: string;
  platform: string;
  name: string;
  region: string;
  vram_gib: number | null;
  memory_gb: number | null;
  price_per_gpu_hour: number;
  effective_time: string;
  fits?: boolean | null;
  note?: string;
};
export type Upload = { name: string; content: string };
export type Parsed = {
  values: { training?: Partial<Training>; model?: Partial<ModelInfo> };
  uncertain: Record<string, string>;
  errors: Record<string, string>;
  metadata: Record<string, unknown>;
  review_required: boolean;
};
export type Result = {
  mode: string;
  config: Analysis;
  normalized: Record<string, unknown>;
  training_steps: number;
  optimizer_updates: number;
  training_hours: number | null;
  timing_source: string;
  runtime_details: {
    status: string;
    missing_information?: string[];
    scenarios?: Record<string, { training_hours: number | null }>;
  } | null;
  gpu_hours: number | null;
  training_cost_usd: number | null;
  gpu: {
    peak_vram_gib: number | null;
    source: string;
    capacity_gib: number;
    limit_gib: number;
    utilization_percent: number | null;
    fits: boolean | null;
  };
  cpu: { required_gib: number | null; capacity_gib: number | null };
  assumptions: string[];
  comparison: Offer[];
  memory_details: { errors?: string[]; warnings?: string[] } | null;
  benchmark: {
    cloud_available: boolean;
    estimated_gpu_cost_usd: number | null;
    assumed_seconds: number;
    note: string;
  };
};
export class APIError extends Error {
  fields: Record<string, string>;
  constructor(message: string, fields: Record<string, string> = {}) {
    super(message);
    this.fields = fields;
  }
}
export const apiBase =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "";
export async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(
    `${apiBase}${path}`,
    body === undefined
      ? undefined
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const data = await response.json();
  if (!response.ok) {
    const fields: Record<string, string> = {};
    if (Array.isArray(data.detail))
      for (const item of data.detail)
        fields[item.loc.filter((v: string) => v !== "body").join(".")] =
          item.msg;
    throw new APIError(
      typeof data.detail === "string"
        ? data.detail
        : "Review the highlighted inputs.",
      fields,
    );
  }
  return data as T;
}
export const benchmarkURL = `${apiBase}/api/benchmark/script`;
export function initialConfig(): Analysis {
  return {
    framework: "pytorch",
    training: {
      examples: 100000,
      batch_size: 16,
      epochs: 10,
      sequence_length: 512,
      gradient_accumulation_steps: 1,
      script_gpu_count: 1,
      strategy: "single",
      seconds_per_batch: null,
      measured_peak_vram_gib: null,
      precision: "fp32",
      optimizer: "AdamW",
      assume_data_ready: false,
    },
    model: {
      total_params: null,
      activation_gib: null,
      auxiliary_gib: null,
      flops_per_microbatch: null,
      hbm_bytes_per_microbatch: null,
      config: {},
    },
    hardware: {
      option_id: "custom",
      name: "Custom GPU",
      vram_gib: 24,
      memory_limit_gib: null,
      available_gpus: 1,
      price_per_gpu_hour: null,
      cpu_ram_gib: null,
      cpu_cores: null,
      nvme_gib: null,
      ssd_gib: null,
      cpu_specs: "",
      compute_tflops: null,
      fp32_tflops: null,
      memory_bandwidth_gbps: null,
      gpu_link_gbps: null,
    },
  };
}
