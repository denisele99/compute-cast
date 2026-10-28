import { ChangeEvent, ReactNode, useEffect, useRef, useState } from "react";
import {
  Analysis,
  APIError,
  apiBase,
  benchmarkURL,
  initialConfig,
  Offer,
  Parsed,
  request,
  Result,
  Upload,
} from "./workflowApi";
import "./planner.css";

type Snapshot = {
  config: Analysis;
  uploads: Upload[];
  metadata: Record<string, unknown>;
};
const clone = <T,>(value: T): T => JSON.parse(JSON.stringify(value)) as T;
const number = (value: number | null, suffix = "") =>
  value === null
    ? "Unknown"
    : `${value.toLocaleString(undefined, { maximumFractionDigits: 4 })}${suffix}`;
function download(name: string, value: unknown) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}
function validation(config: Analysis): Record<string, string> {
  const errors: Record<string, string> = {};
  const ranges: Record<string, [number, number, boolean]> = {
    "training.examples": [1, 1e12, true],
    "training.batch_size": [1, 1e6, true],
    "training.epochs": [1, 100000, true],
    "training.sequence_length": [1, 1e7, true],
    "training.gradient_accumulation_steps": [1, 100000, true],
    "training.script_gpu_count": [1, 1024, true],
    "hardware.available_gpus": [1, 1024, true],
    "hardware.vram_gib": [0.000001, 10000, false],
    "hardware.memory_limit_gib": [0.000001, 10000, false],
    "hardware.price_per_gpu_hour": [0, 1e6, false],
    "hardware.cpu_ram_gib": [0.000001, 1e6, false],
    "hardware.cpu_cores": [1, 100000, true],
    "hardware.nvme_gib": [0, 1e9, false],
    "hardware.ssd_gib": [0, 1e9, false],
    "model.flops_per_microbatch": [1e-12, 1e30, false],
    "model.hbm_bytes_per_microbatch": [1e-12, 1e30, false],
    "hardware.compute_tflops": [1e-12, 1e9, false],
    "hardware.fp32_tflops": [1e-12, 1e9, false],
    "hardware.memory_bandwidth_gbps": [1e-12, 1e9, false],
    "hardware.gpu_link_gbps": [1e-12, 1e9, false],
    "model.total_params": [1, 1e13, true],
    "model.activation_gib": [0, 100000, false],
    "model.auxiliary_gib": [0, 100000, false],
    "training.seconds_per_batch": [0.000000001, 1e6, false],
    "training.measured_peak_vram_gib": [0.000000001, 100000, false],
  };
  const required = [
    "training.examples",
    "training.batch_size",
    "training.epochs",
    "training.sequence_length",
    "training.gradient_accumulation_steps",
    "training.script_gpu_count",
    "hardware.available_gpus",
    "hardware.vram_gib",
  ];
  for (const [path, [min, max, integer]] of Object.entries(ranges)) {
    const [group, key] = path.split(".");
    const value = (
      config[group as "training"] as unknown as Record<string, unknown>
    )[key];
    if (value === null) {
      if (required.includes(path))
        errors[path] = "Required: enter a value or keep the displayed default.";
      continue;
    }
    if (
      typeof value !== "number" ||
      !Number.isFinite(value) ||
      value < min ||
      value > max ||
      (integer && !Number.isInteger(value))
    )
      errors[path] =
        `Enter ${integer ? "a whole number" : "a number"} from ${min} to ${max}.`;
  }
  if (config.training.batch_size > config.training.examples)
    errors["training.batch_size"] =
      "Batch size cannot exceed training examples.";
  if (
    config.hardware.memory_limit_gib !== null &&
    config.hardware.memory_limit_gib > config.hardware.vram_gib
  )
    errors["hardware.memory_limit_gib"] =
      "Limit cannot exceed total GPU capacity.";
  if (config.training.script_gpu_count > config.hardware.available_gpus)
    errors["training.script_gpu_count"] =
      "Script GPU count cannot exceed available GPUs.";
  if (
    config.training.script_gpu_count > 1 &&
    config.training.strategy === "single"
  )
    errors["training.strategy"] =
      "Choose DDP or an explicit sharding strategy for multiple training GPUs.";
  if (
    config.training.script_gpu_count === 1 &&
    config.training.strategy !== "single"
  )
    errors["training.strategy"] =
      "Select single GPU or increase GPUs used by the script.";
  return errors;
}

export default function App() {
  const [config, setConfig] = useState<Analysis>(initialConfig);
  const [step, setStep] = useState<"workload" | "hardware" | "results">(
    "workload",
  );
  const [mode, setMode] = useState<"prediction" | "benchmark">("prediction");
  const [offers, setOffers] = useState<Offer[]>([]);
  const [query, setQuery] = useState("");
  const [catalogNote, setCatalogNote] = useState("");
  const [catalogError, setCatalogError] = useState("");
  const [files, setFiles] = useState<Upload[]>([]);
  const [metadata, setMetadata] = useState<Record<string, unknown>>({});
  const [uncertain, setUncertain] = useState<Record<string, string>>({});
  const [uploadErrors, setUploadErrors] = useState<Record<string, string>>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [serverErrors, setServerErrors] = useState<Record<string, string>>({});
  const [reviewed, setReviewed] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [original, setOriginal] = useState<Snapshot | null>(null);
  const [benchmarkConfig, setBenchmarkConfig] = useState<Analysis | null>(null);
  const [recommendationReady, setRecommendationReady] = useState(false);
  const [recommendationConsent, setRecommendationConsent] = useState(false);
  const [recommendationPrompt, setRecommendationPrompt] = useState(
    "Suggest minimal PyTorch training changes to improve memory fit while preserving training semantics. Explain tradeoffs and validation.",
  );
  const [proposal, setProposal] = useState<{
    review: {
      explanation: string;
      missing_information: string[];
      changed_files: { path: string; content: string; reason: string }[];
    };
    validation: unknown;
  } | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const generation = useRef(0);
  const errors = { ...validation(config), ...serverErrors };
  const hasUploads = files.length > 0;
  const stale =
    result && JSON.stringify(result.config) !== JSON.stringify(config);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${apiBase}/api/hardware?q=${encodeURIComponent(query)}&limit=10`, {
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok)
          throw new Error(
            "Could not load the GPU catalogue. Use custom hardware or retry.",
          );
        const data = await response.json();
        setOffers(data.items);
        setCatalogNote(data.note);
        setCatalogError("");
      })
      .catch((error) => {
        if (error.name !== "AbortError") setCatalogError(String(error.message));
      });
    return () => controller.abort();
  }, [query]);

  useEffect(() => {
    if (step !== "results") return;
    let active = true;
    request<{ configured: boolean }>("/api/recommendations/status")
      .then((data) => {
        if (active) setRecommendationReady(data.configured);
      })
      .catch(() => {
        if (active) setRecommendationReady(false);
      });
    return () => {
      active = false;
    };
  }, [step]);

  async function recommend() {
    const run = generation.current;
    setBusy(true);
    setMessage("");
    setProposal(null);
    try {
      const data = await request<NonNullable<typeof proposal>>(
        "/api/recommendations",
        {
          prompt: recommendationPrompt,
          files: Object.fromEntries(
            files.map((file) => [file.name, file.content]),
          ),
          context: {
            original: original
              ? { config: original.config, metadata: original.metadata }
              : null,
            current: config,
            normalized: result?.normalized,
            last_completed_result: result
              ? {
                  config: result.config,
                  gpu: result.gpu,
                  training_hours: result.training_hours,
                  assumptions: result.assumptions,
                }
              : null,
          },
          consent_to_paid_request: recommendationConsent,
        },
      );
      if (run === generation.current) setProposal(data);
    } catch (error) {
      if (run === generation.current) handleError(error);
    } finally {
      if (run === generation.current) setBusy(false);
    }
  }
  async function applyProposal() {
    if (!proposal) return;
    const run = generation.current;
    const changed = proposal.review.changed_files.map((file) => ({
      name: file.path,
      content: file.content,
    }));
    const combined = [
      ...files.filter(
        (file) => !changed.some((change) => change.name === file.name),
      ),
      ...changed,
    ];
    setBusy(true);
    try {
      const parsed = await request<Parsed>("/api/analysis/parse", {
        files: combined,
      });
      if (run !== generation.current) return;
      setFiles(combined);
      setMetadata(parsed.metadata);
      setUncertain(parsed.uncertain);
      setUploadErrors(parsed.errors);
      setReviewed(false);
      setServerErrors({});
      setSubmitted(false);
      setTouched({});
      setConfig((previous) => ({
        ...previous,
        training: {
          ...previous.training,
          ...parsed.values.training,
          seconds_per_batch: null,
          measured_peak_vram_gib: null,
        },
        model: {
          ...previous.model,
          ...parsed.values.model,
          config: parsed.metadata,
        },
      }));
      setStep("workload");
      setProposal(null);
      setMessage(
        "Proposed source applied to this page cache only. Review extracted values and rerun estimation; generated code has not been executed.",
      );
    } catch (error) {
      if (run === generation.current) handleError(error);
    } finally {
      if (run === generation.current) setBusy(false);
    }
  }

  function update(path: string, value: unknown) {
    const [group, key] = path.split(".");
    setConfig((previous) => {
      const next = {
        ...previous,
        [group]: { ...previous[group as "training"], [key]: value },
      };
      if (
        ![
          "training.seconds_per_batch",
          "training.measured_peak_vram_gib",
          "hardware.price_per_gpu_hour",
          "hardware.memory_limit_gib",
        ].includes(path)
      ) {
        next.training = {
          ...next.training,
          seconds_per_batch: null,
          measured_peak_vram_gib: null,
        };
      }
      return next;
    });
    setBenchmarkConfig(null);
    setTouched((previous) => ({ ...previous, [path]: true }));
    setServerErrors({});
    setMessage("");
  }
  function numeric(
    path: string,
    label: string,
    optional = false,
    hint = "",
    required = false,
  ) {
    const [group, key] = path.split(".");
    const value = (
      config[group as "training"] as unknown as Record<string, unknown>
    )[key];
    const showError = submitted || touched[path] ? errors[path] : undefined;
    const warning = uncertain[path] && !touched[path];
    return (
      <label
        className={`input-label ${showError || warning ? "attention" : ""}`}
        key={path}
        htmlFor={path}
      >
        {label}
        {required && <span className="required"> *</span>}
        {optional && <span className="muted"> · optional</span>}
        <input
          id={path}
          aria-invalid={!!showError}
          aria-describedby={`${path}-help`}
          type="number"
          step="any"
          value={value === null ? "" : String(value)}
          onChange={(event) =>
            update(
              path,
              event.target.value === "" ? null : Number(event.target.value),
            )
          }
        />
        <span
          id={`${path}-help`}
          className={showError || warning ? "field-error" : "muted"}
        >
          {showError ||
            (warning
              ? `* ${uncertain[path]}`
              : hint ||
                (!touched[path] && value !== null
                  ? `Default: ${value}`
                  : optional && value === null
                    ? "Not supplied; remains unknown."
                    : ""))}
        </span>
      </label>
    );
  }
  function select(
    path: string,
    label: string,
    options: string[],
    hint: string,
  ) {
    const [group, key] = path.split(".");
    const value = (
      config[group as "training"] as unknown as Record<string, unknown>
    )[key];
    return (
      <label className="input-label" htmlFor={path}>
        {label}
        <select
          id={path}
          value={String(value)}
          onChange={(e) => update(path, e.target.value)}
        >
          {options.map((item) => (
            <option key={item}>{item}</option>
          ))}
        </select>
        <span
          className={
            errors[path] && (submitted || touched[path])
              ? "field-error"
              : "muted"
          }
        >
          {((submitted || touched[path]) && errors[path]) || hint}
        </span>
      </label>
    );
  }
  async function upload(event: ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (!selected.length) return;
    setMessage("");
    setBusy(true);
    const run = ++generation.current;
    try {
      if (selected.some((file) => file.size > 250000))
        throw new Error("Each upload must be at most 250 KB.");
      const incoming = await Promise.all(
        selected.map(async (file) => ({
          name: file.name,
          content: await file.text(),
        })),
      );
      const combined = [
        ...files.filter(
          (file) => !incoming.some((item) => item.name === file.name),
        ),
        ...incoming,
      ];
      if (
        combined.length > 8 ||
        combined.reduce((sum, file) => sum + new Blob([file.content]).size, 0) >
          1_000_000
      )
        throw new Error("Use at most 8 files and 1 MB combined.");
      const parsed = await request<Parsed>("/api/analysis/parse", {
        files: combined,
      });
      if (run !== generation.current) return;
      if (run !== generation.current) return;
      setFiles(combined);
      setMetadata(parsed.metadata);
      setUncertain(parsed.uncertain);
      setUploadErrors(parsed.errors);
      setReviewed(false);
      setConfig((previous) => ({
        ...previous,
        training: {
          ...previous.training,
          ...parsed.values.training,
          seconds_per_batch: parsed.values.training?.seconds_per_batch ?? null,
          measured_peak_vram_gib: null,
        },
        model: {
          ...previous.model,
          ...parsed.values.model,
          config: parsed.metadata,
        },
      }));
      setTouched({});
      setServerErrors({});
      setMessage(
        "Review all autofilled fields. Red * marks missing or uncertain extraction; optional values may stay empty. Defaults are shown in grey.",
      );
    } catch (error) {
      if (run === generation.current)
        setMessage(String((error as Error).message));
    } finally {
      if (run === generation.current) setBusy(false);
    }
  }
  function chooseOffer(id: string) {
    if (id === "custom") {
      update("hardware.option_id", id);
      update("hardware.name", "Custom GPU");
      return;
    }
    const offer = offers.find((item) => item.id === id);
    if (!offer) return;
    setConfig((previous) => ({
      ...previous,
      training: {
        ...previous.training,
        seconds_per_batch: null,
        measured_peak_vram_gib: null,
      },
      hardware: {
        ...previous.hardware,
        compute_tflops: null,
        fp32_tflops: null,
        memory_bandwidth_gbps: null,
        gpu_link_gbps: null,
        option_id: id,
        name: `${offer.name} · ${offer.region}`,
        vram_gib: offer.vram_gib ?? (null as unknown as number),
        memory_limit_gib: null,
        price_per_gpu_hour: offer.price_per_gpu_hour,
      },
    }));
    setTouched((previous) => ({
      ...previous,
      "hardware.vram_gib": true,
      "hardware.price_per_gpu_hour": true,
    }));
    setServerErrors({});
    setMessage(
      offer.vram_gib === null
        ? "This platform is missing memory specifications in the saved catalogue. Enter and verify its per-GPU memory before continuing."
        : "Memory and price filled from the saved catalogue. Verify the selected region and spot-price date.",
    );
  }
  async function next() {
    const run = generation.current;
    setSubmitted(true);
    setMessage("");
    if (hasUploads && (!reviewed || Object.keys(uploadErrors).length)) {
      setMessage(
        "Resolve upload errors and confirm that you reviewed autofilled fields.",
      );
      return;
    }
    if (
      Object.keys(errors).some(
        (key) =>
          (key.startsWith("training") &&
            !["training.script_gpu_count", "training.strategy"].includes(
              key,
            )) ||
          key.startsWith("model"),
      )
    ) {
      setMessage("Review the workload errors before continuing.");
      return;
    }
    // Validate this step's edited workload; GPU/deployment consistency is validated on the hardware step and full estimate.
    setBusy(true);
    try {
      await request("/api/analysis/normalize", {
        ...config,
        training: {
          ...config.training,
          script_gpu_count: 1,
          strategy: "single",
        },
        hardware: initialConfig().hardware,
      });
      if (run !== generation.current) return;
      setStep("hardware");
      setSubmitted(false);
    } catch (error) {
      if (run === generation.current) handleError(error);
    } finally {
      if (run === generation.current) setBusy(false);
    }
  }
  function handleError(error: unknown) {
    if (error instanceof APIError) setServerErrors(error.fields);
    setMessage((error as Error).message);
  }
  async function estimate() {
    setSubmitted(true);
    setMessage("");
    if (
      Object.keys(errors).length ||
      (hasUploads && (!reviewed || Object.keys(uploadErrors).length))
    ) {
      setMessage(
        "Resolve required inputs and review uploaded values before estimating.",
      );
      return;
    }
    setBusy(true);
    const run = generation.current;
    try {
      const data = await request<Result>("/api/analysis/estimate", config);
      if (run !== generation.current) return;
      setOriginal(
        (previous) => previous ?? clone({ config, uploads: files, metadata }),
      );
      setResult(data);
      setStep("results");
      setSubmitted(false);
    } catch (error) {
      if (run === generation.current) handleError(error);
    } finally {
      if (run === generation.current) setBusy(false);
    }
  }
  async function importBenchmark(event: ChangeEvent<HTMLInputElement>) {
    const run = generation.current;
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    try {
      if (file.size > 1_000_000)
        throw new Error("Benchmark report must be at most 1 MB.");
      const report = JSON.parse(await file.text());
      const imported = await request<{
        config: Analysis;
        gpu_name: string;
        note: string;
      }>("/api/benchmark/import", { config, report });
      if (run !== generation.current) return;
      setConfig(imported.config);
      setBenchmarkConfig(clone(imported.config));
      setMessage(
        `Imported measurements from ${imported.gpu_name}. ${imported.note} Review the measurements, then run the estimate again.`,
      );
      setTouched((previous) => ({
        ...previous,
        "training.seconds_per_batch": true,
        "training.measured_peak_vram_gib": true,
      }));
    } catch (error) {
      if (run === generation.current) handleError(error);
    }
  }
  function reset() {
    generation.current++;
    setConfig(initialConfig());
    setFiles([]);
    setMetadata({});
    setUncertain({});
    setUploadErrors({});
    setTouched({});
    setServerErrors({});
    setReviewed(false);
    setSubmitted(false);
    setBusy(false);
    setOriginal(null);
    setResult(null);
    setBenchmarkConfig(null);
    setStep("workload");
    setMode("prediction");
    setMessage("");
    setProposal(null);
    setRecommendationConsent(false);
  }
  const difference = original
    ? Object.entries({
        ...Object.fromEntries(
          Object.entries(config.training).map(([key, value]) => [
            `training.${key}`,
            value,
          ]),
        ),
        ...Object.fromEntries(
          Object.entries(config.hardware).map(([key, value]) => [
            `hardware.${key}`,
            value,
          ]),
        ),
        ...Object.fromEntries(
          Object.entries(config.model).map(([key, value]) => [
            `model.${key}`,
            value,
          ]),
        ),
      }).filter(([path, value]) => {
        const [group, key] = path.split(".");
        return (
          JSON.stringify(value) !==
          JSON.stringify(
            (
              original.config[group as "training"] as unknown as Record<
                string,
                unknown
              >
            )[key],
          )
        );
      })
    : [];
  return (
    <main className="planner">
      <header className="planner-header">
        <div>
          <p className="eyebrow">COMPUTE CAST / GPU PLANNER</p>
          <h1>Plan your training run.</h1>
          <p>
            PyTorch scripts only. Upload Python and YAML/JSON configs, or enter
            the same inputs manually.
          </p>
        </div>
        <button className="secondary" onClick={reset}>
          New analysis
        </button>
      </header>
      <nav aria-label="Analysis steps">
        {(["workload", "hardware", "results"] as const).map((item, index) => (
          <span key={item} aria-current={step === item ? "step" : undefined}>
            {index + 1}. {item}
          </span>
        ))}
      </nav>
      {message && (
        <div className="notice" role="status">
          {message}
        </div>
      )}
      {Object.entries(serverErrors).map(([key, value]) => (
        <p role="alert" className="field-error" key={key}>
          {key || "Inputs"}: {value}
        </p>
      ))}
      {step === "workload" && (
        <>
          <section>
            <h2>Model & training inputs</h2>
            <p>
              Uploads are held in this page’s memory for recommendations and
              exports. The server parses each request without caching files.
              Refreshing, closing the page, or starting a new analysis clears
              this cache.
            </p>
            <label className="upload">
              Upload model/training Python scripts or JSON/YAML configs
              <input
                ref={fileInput}
                aria-label="Upload model and training files"
                type="file"
                accept=".py,.json,.yaml,.yml"
                multiple
                disabled={busy}
                onChange={upload}
              />
            </label>
            <p className="muted">
              Up to 8 files · 250 KB each · 1 MB total. Scripts are inspected
              with Python AST, never imported or executed by the upload API.
              Dynamic model shapes and parameter counts may require manual
              input.
            </p>
            {files.map((file) => (
              <div className="file" key={file.name}>
                <span>{file.name}</span>
                <button
                  className="secondary"
                  disabled={busy}
                  onClick={() => {
                    setFiles((previous) =>
                      previous.filter((item) => item.name !== file.name),
                    );
                    setMetadata((previous) => {
                      const next = { ...previous };
                      delete next[file.name];
                      return next;
                    });
                    setConfig((previous) => {
                      const next = { ...previous.model.config };
                      delete next[file.name];
                      return {
                        ...previous,
                        model: { ...previous.model, config: next },
                      };
                    });
                    setProposal(null);
                    setReviewed(false);
                    setMessage(
                      "File removed from cache. Previously extracted fields remain editable; review them before continuing.",
                    );
                    setUploadErrors((previous) => {
                      const next = { ...previous };
                      delete next[file.name];
                      return next;
                    });
                  }}
                >
                  Remove
                </button>
                {uploadErrors[file.name] && (
                  <p className="field-error">{uploadErrors[file.name]}</p>
                )}
              </div>
            ))}
            {hasUploads && (
              <label className="check">
                <input
                  type="checkbox"
                  checked={reviewed}
                  onChange={(event) => setReviewed(event.target.checked)}
                />
                I have verified the autofilled fields and reviewed any defaults
                or missing values.
              </label>
            )}
            <div className="form-grid">
              {numeric(
                "training.examples",
                "Training examples",
                false,
                "",
                true,
              )}
              {numeric(
                "training.batch_size",
                "Batch size per GPU",
                false,
                "",
                true,
              )}
              {numeric("training.epochs", "Epochs", false, "", true)}
              {numeric(
                "model.total_params",
                "Total model parameters",
                true,
                "Needed for a model-based memory estimate; unknown is allowed.",
              )}
            </div>
            <details>
              <summary>Advanced model / training settings</summary>
              <div className="form-grid">
                {numeric("training.sequence_length", "Sequence length")}
                {numeric(
                  "training.gradient_accumulation_steps",
                  "Gradient accumulation steps",
                )}
                {select(
                  "training.precision",
                  "Compute precision",
                  ["fp32", "fp16", "bf16"],
                  "Default FP32. Stored parameters and gradients are assumed FP32.",
                )}
                {select(
                  "training.optimizer",
                  "Optimizer",
                  ["AdamW", "Adam", "SGD"],
                  "Default AdamW; FP32 optimizer state. SGD assumes no momentum.",
                )}
                {numeric(
                  "model.activation_gib",
                  "Saved activation storage per rank (GiB)",
                  true,
                  "Optional planning override, not inferred from parameter count.",
                )}
                {numeric(
                  "model.auxiliary_gib",
                  "Auxiliary tensor storage per rank (GiB)",
                  true,
                  "Provide together with activation storage for a peak estimate.",
                )}
                {numeric(
                  "model.flops_per_microbatch",
                  "Forward/backward FLOPs per microbatch",
                  true,
                  "For metadata-only runtime planning; use operation counts for this exact per-rank batch.",
                )}
                {numeric(
                  "model.hbm_bytes_per_microbatch",
                  "HBM traffic bytes per microbatch",
                  true,
                  "Traffic, not resident VRAM. Required alongside FLOPs.",
                )}
              </div>
              <label className="check">
                <input
                  type="checkbox"
                  checked={config.training.assume_data_ready}
                  onChange={(event) =>
                    update("training.assume_data_ready", event.target.checked)
                  }
                />
                Assume data is already on the GPU / loading is completely hidden
                (optional runtime planning assumption).
              </label>
              <p className="muted">
                Assumptions: dense fully trainable parameters, zero model
                buffers, no extra master copy, no checkpointing, one balanced
                node. Unsupported models may remain incomplete. Original configs
                are retained in the general export; only supported fields enter
                the normalized estimator config.
              </p>
            </details>
          </section>
          <button disabled={busy} onClick={next}>
            {busy ? "Validating…" : "Review hardware →"}
          </button>
        </>
      )}
      {step === "hardware" && (
        <>
          <section>
            <h2>Hardware</h2>
            <label className="input-label">
              Search Nebius platforms / regions
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="H200, H100, eu-north1…"
              />
            </label>
            {catalogError && (
              <p className="field-error" role="alert">
                {catalogError}
              </p>
            )}
            <label className="input-label" htmlFor="gpu-platform">
              GPU / platform
              <select
                id="gpu-platform"
                aria-label="GPU / platform"
                value={config.hardware.option_id}
                onChange={(event) => chooseOffer(event.target.value)}
              >
                <option value="custom">Custom hardware</option>
                {!offers.some(
                  (offer) => offer.id === config.hardware.option_id,
                ) &&
                  config.hardware.option_id !== "custom" && (
                    <option value={config.hardware.option_id}>
                      {config.hardware.name} (selected)
                    </option>
                  )}
                {offers.map((offer) => (
                  <option key={offer.id} value={offer.id}>
                    {offer.name} · {offer.region} ·{" "}
                    {offer.memory_gb === null
                      ? "Memory unknown"
                      : `${offer.memory_gb} GB`}{" "}
                    · ${offer.price_per_gpu_hour}/GPU-hour
                  </option>
                ))}
              </select>
            </label>
            <p className="muted">
              {catalogNote} Showing up to 10 matching offers. Regional variants
              are separate offers.
            </p>
            <div className="form-grid">
              {numeric(
                "hardware.vram_gib",
                "Memory per GPU (GiB)",
                false,
                config.hardware.option_id !== "custom"
                  ? "Catalogue GB converted to GiB; verify capacity."
                  : "Default custom GPU: 24 GiB. Verify your actual device.",
                true,
              )}
              {numeric(
                "hardware.price_per_gpu_hour",
                "Price (USD / GPU-hour)",
                true,
                "Autofilled for catalogue offers; editable. Empty price leaves cost unknown.",
              )}
              {numeric(
                "hardware.memory_limit_gib",
                "Usable memory limit per GPU (GiB)",
                true,
                `Default: 90% of capacity (${number(config.hardware.vram_gib === null ? null : config.hardware.vram_gib * 0.9)} GiB).`,
              )}
              {numeric(
                "hardware.available_gpus",
                "Available identical GPUs",
                false,
                "Default 1. Availability does not imply the script uses every GPU.",
                true,
              )}
              {(config.hardware.available_gpus > 1 ||
                config.training.script_gpu_count > 1) &&
                numeric(
                  "training.script_gpu_count",
                  "How many GPUs are used in your current training script?",
                  false,
                  "Default 1, or a reviewed value extracted from training info.",
                  true,
                )}
              {select(
                "training.strategy",
                "Distribution / sharding",
                ["single", "ddp", "zero1", "zero2", "zero3", "fsdp_full_shard"],
                "Default single GPU. DDP replicates states; ZeRO/FSDP explicitly shard states. This plans resources; it does not rewrite your training script.",
              )}
            </div>
            <details>
              <summary>Advanced hardware settings · optional</summary>
              <div className="form-grid">
                {numeric(
                  "hardware.cpu_ram_gib",
                  "Available CPU RAM (GiB)",
                  true,
                )}
                {numeric("hardware.cpu_cores", "CPU cores", true)}
                {numeric("hardware.nvme_gib", "NVMe capacity (GiB)", true)}
                {numeric("hardware.ssd_gib", "SSD capacity (GiB)", true)}
                {numeric(
                  "hardware.compute_tflops",
                  "GPU dense compute peak at selected precision (TFLOPS)",
                  true,
                  "Required for metadata-only runtime. Catalogue snapshots do not contain throughput.",
                )}
                {numeric(
                  "hardware.fp32_tflops",
                  "GPU FP32 peak for optimizer (TFLOPS)",
                  true,
                )}
                {numeric(
                  "hardware.memory_bandwidth_gbps",
                  "GPU memory bandwidth (GB/s)",
                  true,
                )}
                {numeric(
                  "hardware.gpu_link_gbps",
                  "Effective GPU–GPU bandwidth (GB/s)",
                  true,
                  "Needed for multi-GPU runtime planning.",
                )}
                <label className="input-label">
                  CPU specifications
                  <input
                    value={config.hardware.cpu_specs}
                    maxLength={500}
                    onChange={(event) =>
                      update("hardware.cpu_specs", event.target.value)
                    }
                  />
                </label>
              </div>
              <p className="muted">
                Retained for review; these capacities alone cannot predict CPU
                RAM demand, data loading, or storage throughput. CPU/NVMe
                offload is not enabled by filling these fields.
              </p>
            </details>
          </section>
          <section>
            <h2>Prediction or benchmark</h2>
            <Mode mode={mode} setMode={setMode} />
            <p>
              Prediction is immediate and uses metadata only. Benchmarking is
              more accurate for the measured workload, requires GPU access, and
              may incur cloud charges.
            </p>
            <div className="form-grid">
              {numeric(
                "training.seconds_per_batch",
                "Matching measured seconds per microbatch",
                true,
                "Includes forward, loss, backward, optimizer/data work for the current GPU count. Empty uses explicit runtime metadata if complete; otherwise time remains unknown.",
              )}
              {numeric(
                "training.measured_peak_vram_gib",
                "Matching measured peak VRAM per GPU (GiB)",
                true,
                "Empty uses the memory planner if sufficient metadata is supplied.",
              )}
            </div>
            <p className="muted">
              Changing hardware, batch size or sharding invalidates earlier
              measurements. Re-enter only measurements matching the current
              setup.
            </p>
          </section>
          <div className="actions">
            <button
              className="secondary"
              onClick={() => {
                setStep("workload");
                setSubmitted(false);
              }}
            >
              ← Workload
            </button>
            <button disabled={busy} onClick={estimate}>
              {busy ? "Estimating…" : "Run estimate"}
            </button>
          </div>
        </>
      )}
      {step === "results" && result && (
        <>
          <section>
            <div className="actions">
              <h2>Results</h2>
              <Mode mode={mode} setMode={setMode} />
            </div>
            {stale && (
              <p className="notice">
                Inputs have changed. These results describe the last completed
                run; run the estimate again to update them.
              </p>
            )}
            <p className="muted">
              {benchmarkConfig &&
              JSON.stringify(config) === JSON.stringify(benchmarkConfig)
                ? "Imported local benchmark measurements"
                : "Prediction / user-supplied metadata"}{" "}
              · {result.config.hardware.name} ·{" "}
              {result.config.training.script_gpu_count} active GPU(s)
            </p>
            <div className="result-grid">
              <Metric
                title="Training runtime"
                value={number(result.training_hours, " h")}
              />
              <Metric
                title="GPU time"
                value={number(result.gpu_hours, " GPU-h")}
              />
              <Metric
                title="Estimated GPU cost"
                value={
                  result.training_cost_usd === null
                    ? "Unknown"
                    : `$${result.training_cost_usd.toFixed(4)}`
                }
              />
            </div>
            <p className="muted">Timing source: {result.timing_source}</p>
            {result.runtime_details?.scenarios && (
              <p className="muted">
                Planning scenarios:{" "}
                {Object.entries(result.runtime_details.scenarios)
                  .map(
                    ([name, data]) =>
                      `${name}: ${number(data.training_hours, " h")}`,
                  )
                  .join(" · ")}
                . These are assumptions, not confidence bounds.
              </p>
            )}
            <h3>Does the model fit?</h3>
            <p className={result.gpu.fits === false ? "field-error" : ""}>
              {result.gpu.fits === null
                ? "Unknown — more memory metadata or a matching measurement is needed."
                : result.gpu.fits
                  ? "Within the selected memory limit; test on the actual GPU before a full run."
                  : "Exceeds the selected per-GPU memory limit."}
            </p>
            <label>
              VRAM · {number(result.gpu.peak_vram_gib, " GiB")} /{" "}
              {number(result.gpu.capacity_gib, " GiB")} · {result.gpu.source}
              <progress
                aria-label="VRAM utilization"
                max={100}
                value={result.gpu.utilization_percent ?? 0}
              />
              {result.gpu.utilization_percent === null
                ? "Usage unknown"
                : `${result.gpu.utilization_percent.toFixed(1)}% of physical capacity`}{" "}
              · usable limit {number(result.gpu.limit_gib, " GiB")}
            </label>
            <p>
              CPU RAM: required <b>unknown</b> / available{" "}
              {number(result.cpu.capacity_gib, " GiB")}. No CPU usage percentage
              is invented.
            </p>
            <p>
              {result.training_steps.toLocaleString()} microbatches per rank ·{" "}
              {result.optimizer_updates.toLocaleString()} optimizer updates.
            </p>
            <details>
              <summary>How we calculated this</summary>
              <p>
                Microbatches per rank = ceil(ceil(examples / active GPUs) /
                per-GPU batch size) × epochs. Runtime = microbatches × matching
                measured seconds per microbatch. Cost = hours × active GPUs ×
                price per GPU-hour. Accumulation changes optimizer update count,
                not the number of forward/backward microbatches.
              </p>
              <ul>
                {result.assumptions.map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
              {result.memory_details?.errors?.map((error) => (
                <p className="field-error" key={error}>
                  {error}
                </p>
              ))}
              {result.memory_details?.warnings?.map((warning) => (
                <p key={warning}>{warning}</p>
              ))}
              <button
                className="secondary"
                onClick={() =>
                  download(
                    "normalized_estimator_config.json",
                    result.normalized,
                  )
                }
              >
                Download normalized estimator config
              </button>
            </details>
          </section>
          {mode === "benchmark" && (
            <section>
              <h2>Benchmark this workload</h2>
              <p>
                Requires a GPU and trusted training code. Cloud execution can
                incur additional charges; inspect the prediction and cost
                assumptions first.
              </p>
              <p>
                Illustrative 60-second GPU-only benchmark cost:{" "}
                <b>
                  {result.benchmark.estimated_gpu_cost_usd === null
                    ? "Unknown — enter a GPU-hour price"
                    : `$${result.benchmark.estimated_gpu_cost_usd.toFixed(4)}`}
                </b>
                . {result.benchmark.note}
              </p>
              <h3>Run on Nebius</h3>
              <button
                disabled
                title="Cloud job provisioning and billing authorization are not implemented"
              >
                Run benchmark on Nebius — unavailable
              </button>
              <p className="muted">
                Cloud provisioning, billing consent, job isolation and
                credentials still need integration. No paid service is called.
              </p>
              <h3>Run locally</h3>
              <ol>
                <li>
                  Use a CUDA GPU with matching hardware and install your
                  project’s PyTorch/dependencies locally.
                </li>
                <li>
                  Keep your model script, training adapter, data sample and all
                  project imports available. The training file must expose{" "}
                  <code>build_benchmark(model_path, data_path)</code>, returning
                  a no-argument callable for one complete training microbatch.
                  It must match the exported config.
                </li>
                <li>
                  Download the runner and current config. Inspect your scripts:
                  the local runner executes your training adapter. It supports
                  one GPU; distributed benchmarking requires a distributed
                  runner.
                </li>
                <li>
                  Run the command, review the report, then import it below. Only
                  matching workload/configuration reports are accepted.
                </li>
              </ol>
              <div className="actions">
                <a className="button" href={benchmarkURL} download>
                  Download benchmark script
                </a>
                <button
                  className="secondary"
                  onClick={() => download("analysis_config.json", config)}
                >
                  Download current config
                </button>
              </div>
              <pre>
                python run_benchmark.py --model model_path --train training_path
                --data_sample data --config analysis_config.json --output
                benchmark.json
              </pre>
              <label className="upload">
                Import completed benchmark report
                <input
                  aria-label="Import benchmark report"
                  type="file"
                  accept=".json"
                  onChange={importBenchmark}
                />
              </label>
            </section>
          )}
          <section>
            <h2>Hardware comparison</h2>
            <p className="muted">
              Fits uses 90% of each card’s capacity. Measured memory is
              configuration-specific; comparisons are indicative and require a
              new benchmark on the selected hardware. Runtime and cost are not
              transferable between GPUs.
            </p>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Platform / region</th>
                    <th>Memory</th>
                    <th>USD/GPU-hour</th>
                    <th>Fit</th>
                  </tr>
                </thead>
                <tbody>
                  {result.comparison.map((offer) => (
                    <tr key={offer.id}>
                      <td>
                        {offer.name}
                        <br />
                        <small>{offer.region}</small>
                      </td>
                      <td>{number(offer.vram_gib, " GiB")}</td>
                      <td>${offer.price_per_gpu_hour}</td>
                      <td>
                        {offer.fits === null
                          ? "Unknown"
                          : offer.fits
                            ? "Within limit"
                            : "Exceeds limit"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
          <section>
            <h2>Code recommendations · optional</h2>
            <p>
              Use the cached model/training files and original/current metadata
              to request a proposed script update. Generated files require
              review and testing. The provider request limit is 250 KB including
              metadata.
            </p>
            {!recommendationReady && (
              <p className="muted">
                The model provider is not configured. Configure NEBIUS_API_KEY
                and the OpenAI SDK on the backend to enable this optional
                feature. Estimation and local benchmarks work independently.
              </p>
            )}
            {!files.length && (
              <p className="muted">
                Upload source files on the workload step to enable code
                recommendations.
              </p>
            )}
            <label className="input-label">
              Recommendation goal
              <textarea
                rows={3}
                value={recommendationPrompt}
                maxLength={10000}
                onChange={(event) =>
                  setRecommendationPrompt(event.target.value)
                }
              />
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={recommendationConsent}
                onChange={(event) =>
                  setRecommendationConsent(event.target.checked)
                }
              />
              I approve sending cached source files and analysis metadata to the
              configured Nebius model provider. Paid API charges may apply.
            </label>
            <button
              disabled={
                busy ||
                !recommendationReady ||
                !recommendationConsent ||
                !files.length ||
                !recommendationPrompt.trim()
              }
              onClick={recommend}
            >
              {busy ? "Working…" : "Generate code recommendations (paid API)"}
            </button>
            {proposal && (
              <div>
                <h3>Review proposed changes</h3>
                <p>{proposal.review.explanation}</p>
                {proposal.review.missing_information.map((item) => (
                  <p key={item}>{item}</p>
                ))}
                {proposal.review.changed_files.map((file) => (
                  <details key={file.path}>
                    <summary>{file.path}</summary>
                    <p>{file.reason}</p>
                    <pre>{file.content}</pre>
                  </details>
                ))}
                <div className="actions">
                  <button
                    className="secondary"
                    onClick={() =>
                      download("proposed_code_review.json", proposal)
                    }
                  >
                    Download proposed files & review
                  </button>
                  <button
                    disabled={busy || !proposal.review.changed_files.length}
                    onClick={applyProposal}
                  >
                    Apply proposed files to page cache
                  </button>
                </div>
                <p className="muted">
                  Review full file contents before applying. This updates cached
                  files only; it does not write to your repository, execute
                  code, or validate model behavior.
                </p>
              </div>
            )}
          </section>
          <section>
            <h2>Original run & current inputs</h2>
            <p>
              The first successful run stays unchanged until refresh or New
              analysis. Uploaded source remains available in page memory for a
              recommendation/export bundle.
            </p>
            {difference.length ? (
              <ul>
                {difference.map(([path, value]) => {
                  const [group, key] = path.split(".");
                  return (
                    <li key={path}>
                      {path}: original{" "}
                      <code>
                        {JSON.stringify(
                          (
                            original!.config[
                              group as "training"
                            ] as unknown as Record<string, unknown>
                          )[key],
                        )}
                      </code>{" "}
                      → current <code>{JSON.stringify(value)}</code>
                    </li>
                  );
                })}
              </ul>
            ) : (
              <p className="muted">No changes from the original run.</p>
            )}
            <button
              className="secondary"
              onClick={() =>
                download("analysis_review_bundle.json", {
                  original,
                  current: { config, uploads: files, metadata },
                  last_result: result,
                })
              }
            >
              Download review / recommendation bundle
            </button>
            <p className="muted">
              Includes uploaded source. Downloading this bundle does not call a
              model provider.
            </p>
          </section>
          <div className="actions">
            <button className="secondary" onClick={() => setStep("hardware")}>
              ← Edit inputs
            </button>
            <button disabled={busy} onClick={estimate}>
              {busy ? "Estimating…" : "Run estimate again"}
            </button>
          </div>
        </>
      )}
      <footer>
        Planning estimates are not guarantees. Optional inputs may remain empty;
        unknown requirements stay unknown. No script execution or GPU allocation
        occurs on this server.
      </footer>
    </main>
  );
}
function Mode({
  mode,
  setMode,
}: {
  mode: "prediction" | "benchmark";
  setMode: (mode: "prediction" | "benchmark") => void;
}) {
  return (
    <div className="mode" role="group" aria-label="Result mode">
      <button
        aria-pressed={mode === "prediction"}
        onClick={() => setMode("prediction")}
      >
        Prediction
      </button>
      <button
        aria-pressed={mode === "benchmark"}
        onClick={() => setMode("benchmark")}
      >
        Benchmark
      </button>
    </div>
  );
}
function Metric({ title, value }: { title: string; value: ReactNode }) {
  return (
    <article>
      <p className="muted">{title}</p>
      <strong>{value}</strong>
    </article>
  );
}
