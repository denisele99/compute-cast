import { ChangeEvent } from 'react';
import { Upload } from 'lucide-react';
import { Accordion, Badge, NumField, SelectField, Section, Toggle, Unavailable } from '../components/ui';
import { Setup } from '../lib/scenarios';
import { ParsedWorkload } from '../lib/workloadFile';

export function WorkloadPage({ setup, set, extracted, fileName, fileMessage, onFile, hasCollector, onNext, error }: {
  setup: Setup; set: <K extends keyof Setup>(k: K, v: Setup[K]) => void; extracted: ParsedWorkload | null; fileName: string; fileMessage: string;
  onFile: (e: ChangeEvent<HTMLInputElement>) => void; hasCollector: boolean; onNext: () => void; error: string;
}): JSX.Element {
  const n = (k: keyof Setup) => (v: number | '') => set(k, v as never);
  return <form className="space-y-6" onSubmit={(e) => { e.preventDefault(); onNext(); }} noValidate>
    <Section number="01" title="Model" hint="Peak-memory estimates need the tensor inventory that collect_workload.py records.">
      <div className="grid gap-4 md:grid-cols-2">
        <label className="flex cursor-pointer items-center gap-4 border border-dashed border-violet bg-violet/5 p-5 focus-within:ring-2 focus-within:ring-violet"><Upload size={20} /><span className="text-sm"><strong className="block">Upload collect_workload.py output (.json)</strong><span className="text-xs text-[#a7a0bb]">{fileName || 'Read in your browser. Python files are never executed.'}</span></span><input type="file" accept=".json,application/json" className="sr-only" onChange={onFile} /></label>
        <div className="space-y-2"><Unavailable>Nebius Token Factory model list: needs a catalog endpoint (proposed GET /api/models).</Unavailable><Unavailable>Hugging Face model lookup: needs POST /api/workloads/parse. Upload model.py or config.json will use the same endpoint.</Unavailable></div>
      </div>
      {fileMessage && <p className="mt-4 text-sm text-[#d8d3e3]" aria-live="polite">{fileMessage}</p>}
      {extracted && <div className="mt-4 border border-line bg-[#11111a] p-4 text-xs"><p className="eyebrow mb-3">EXTRACTED FROM FILE</p><dl className="grid gap-2 sm:grid-cols-2">{extracted.summary.map((r) => <div key={r.label} className="flex justify-between gap-3"><dt className="capitalize text-[#8e899b]">{r.label}</dt><dd>{r.value}</dd></div>)}</dl>
        {extracted.missing.length > 0 && <p className="mt-3 text-[#e6d28a]">Not in the file, please enter: {extracted.missing.join(', ')}.</p>}{extracted.warnings.map((w) => <p key={w} className="mt-2 text-[#e6d28a]">{w}</p>)}</div>}
      {!hasCollector && <p className="mt-4 text-sm text-[#e6d28a]">Without the collector file, VRAM cannot be estimated. Runtime and cost can still be computed from a benchmark.</p>}
    </Section>
    <Section number="02" title="Training setup" hint="Names match the collector's training section.">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <NumField label="Batch size per GPU" value={setup.batch_size_per_gpu} min={1} onChange={n('batch_size_per_gpu')} />
        <NumField label="Epochs" value={setup.epochs} min={1} onChange={n('epochs')} />
        <NumField label="Gradient accumulation" value={setup.gradient_accumulation_steps} min={1} suffix="steps" onChange={n('gradient_accumulation_steps')} />
        <SelectField label="Precision" value={setup.precision} options={['fp32', 'fp16', 'bf16']} onChange={(v) => set('precision', v)} />
        <NumField label="Measured seconds per batch" value={setup.seconds_per_batch} min={0} step={0.001} suffix="s" optional onChange={n('seconds_per_batch')} help="From a real benchmark. Needed for runtime and cost." />
        <SelectField label="Optimizer" value={setup.optimizer} options={['AdamW', 'Adam', 'SGD']} onChange={(v) => set('optimizer', v)} />
      </div>
      <div className="mt-4"><Unavailable>Uploading a training script or config to autofill these fields needs POST /api/workloads/parse.</Unavailable></div>
    </Section>
    <Section number="03" title="Dataset">
      <div className="grid gap-4 sm:grid-cols-3">
        <NumField label="Training examples" value={setup.num_examples} min={1} onChange={n('num_examples')} />
        <SelectField label="Input type" value={setup.input_type} options={['text / tokens', 'images', 'tabular']} onChange={(v) => set('input_type', v)} />
        <NumField label="Sequence length" value={setup.sequence_length} min={1} suffix="tokens" optional onChange={n('sequence_length')} help="Informational. The estimator reads input shapes from the collector file." />
      </div>
      <div className="mt-4 space-y-4"><Accordion title="Advanced settings" hint="memory optimization">
        <div className="grid gap-4 sm:grid-cols-2"><Toggle label="Gradient checkpointing" hint="Recompute activations to save VRAM" checked={setup.gradient_checkpointing} onChange={(v) => set('gradient_checkpointing', v)} />
          <SelectField label="Sharding strategy" value={setup.distributed_strategy} options={['single']} onChange={(v) => set('distributed_strategy', v)} /></div>
        <p className="mt-3 text-xs text-[#8e899b]">Sharding and CPU/NVMe offload need an explicit deployment config (backend: deployment). Only single-GPU is exposed until that translation exists.</p></Accordion>
        <Unavailable>Uploading a dataset sample to autofill needs a parsing endpoint.</Unavailable></div>
    </Section>
    {error && <p role="alert" className="border border-[#8a3040] bg-[#2a0f16] p-4 text-sm text-[#ffb3be]">{error}</p>}
    <div className="flex justify-end"><button className="bg-white px-6 py-3 text-sm font-semibold text-black transition hover:bg-violet hover:text-white">Continue to hardware</button></div>
  </form>;
}
