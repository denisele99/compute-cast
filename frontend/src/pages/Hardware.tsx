import { useEffect, useState } from 'react';
import { HardwareOption, HardwareSearch, api } from '../api';
import { Field, FieldGrid, FormCtx } from '../components/FieldGrid';
import { Accordion, Alert, Badge, CopyBlock, Section } from '../components/ui';

const unitLabel = (u: string | null | undefined) => (u === 'gpu_hour' ? 'per GPU-hour' : u === 'instance_hour' ? 'per instance-hour' : 'unit unknown');

export function priceText(o: HardwareOption): string {
  if (!o.price || o.price.value === null) return 'No price in catalogue';
  return `$${o.price.value.toFixed(2)} ${unitLabel(o.price.unit)}${o.price.preemptible ? ' (preemptible)' : ''}`;
}

export function HardwarePage({ ctx, onSelect, onCustom, onBack, onRun, busy, blockers }: {
  ctx: FormCtx; onSelect: (o: HardwareOption) => Promise<void>; onCustom: () => void; onBack: () => void; onRun: () => void; busy: boolean; blockers: string[];
}): JSX.Element {
  const [q, setQ] = useState(''); const [offset, setOffset] = useState(0);
  const [result, setResult] = useState<HardwareSearch | null>(null); const [error, setError] = useState(''); const [loading, setLoading] = useState(false);
  const [command, setCommand] = useState<{ linux: string; gpu_only: string; notes: string[] } | null>(null);
  const selected = ctx.values['hardware.selection_id'] as string | null; const source = ctx.values['hardware.source'];
  useEffect(() => {
    const t = setTimeout(() => { setLoading(true); setError('');
      api.hardwareSearch(q, offset).then(setResult).catch((e) => setError(e instanceof Error ? e.message : String(e))).finally(() => setLoading(false)); }, 200);
    return () => clearTimeout(t);
  }, [q, offset]);
  useEffect(() => { api.localCommand().then(setCommand).catch(() => setCommand(null)); }, []);
  const avail = Number(ctx.values['hardware.gpu_count_available'] ?? 1);
  const spec = (k: string) => ctx.schema.fields.find((f) => f.key === k)!;
  const selectedRow = result?.items.find((o) => o.id === selected);
  return <div className="space-y-6">
    <Section number="04" title="Hardware" hint="General GPU specs come from the hardware catalogue; Nebius availability and preemptible prices come from separate Nebius data files.">
      <div className="grid gap-4 lg:grid-cols-[1fr_auto]">
        <input className="field" aria-label="Search GPUs and Nebius platforms" placeholder="Search: H100, H200 us-central1, L40S, A100…" value={q} onChange={(e) => { setQ(e.target.value); setOffset(0); }} />
        <button type="button" onClick={onCustom} className={`border px-4 py-2 text-sm ${source === 'custom' ? 'border-violet bg-violet/15' : 'border-line'}`}>Enter custom hardware</button>
      </div>
      {loading && <p className="mt-3 text-sm text-[#a7a0bb]" aria-live="polite">Loading catalogue…</p>}
      {error && <div className="mt-3"><Alert kind="error">{error}</Alert></div>}
      {result && <>
        <ul className="mt-4 grid gap-2 sm:grid-cols-2" aria-label="Hardware search results">{result.items.map((o) => <li key={o.id}>
          <button type="button" onClick={() => void onSelect(o)} aria-pressed={selected === o.id} className={`w-full border p-4 text-left ${selected === o.id ? 'border-violet bg-violet/10' : 'border-line hover:border-white/40'}`}>
            <strong className="block text-sm">{o.label}{o.price?.region ? ` · ${o.price.region}` : ''}</strong>
            <span className="mt-2 flex flex-wrap gap-2 text-xs text-[#c4bfcd]"><span>{o.vram_gib !== null ? `${o.vram_gib} GiB VRAM / GPU` : 'VRAM unknown'}</span><span>· {priceText(o)}</span>
              <Badge kind="muted">{o.provider === 'nebius' ? 'Nebius' : 'general catalogue'}</Badge>{!o.specs && <Badge kind="warn">no catalogue specs</Badge>}</span>
            {o.issues.length > 0 && <span className="mt-2 block text-[11px] text-[#e6d28a]">{o.issues.join(' ')}</span>}
          </button></li>)}</ul>
        {result.items.length === 0 && <p className="mt-4 text-sm text-[#e6d28a]">No catalogue match for “{q}”. Enter custom hardware instead.</p>}
        <div className="mt-3 flex items-center justify-between text-xs text-[#8e899b]"><span>Showing {result.total ? offset + 1 : 0}–{Math.min(offset + 10, result.total)} of {result.total} · price snapshot {result.price_snapshot ?? 'unknown'}</span>
          <span className="flex gap-2"><button type="button" className="border border-line px-2 py-1 disabled:opacity-40" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 10))}>Previous</button>
            <button type="button" className="border border-line px-2 py-1 disabled:opacity-40" disabled={offset + 10 >= result.total} onClick={() => setOffset(offset + 10)}>Next</button></span></div>
        {result.notes.map((n) => <p key={n} className="mt-1 text-xs text-[#8e899b]">{n}</p>)}
      </>}
      {selectedRow?.presets.length ? <p className="mt-3 text-xs text-[#a7a0bb]">Presets for {selectedRow.label}: {selectedRow.presets.map((p) => `${p.name} (${p.gpu_count} GPU, ${p.ram_gib} GiB RAM)`).join(', ')}</p> : null}

      <h3 className="mb-3 mt-8 text-sm font-semibold">GPU and price</h3>
      <FieldGrid ctx={ctx} section="hardware" only={['hardware.gpu_name', 'hardware.total_vram_gib', 'hardware.usable_vram_gib', 'hardware.price_per_hour', 'hardware.price_unit', 'hardware.gpus_per_instance']} />
      <p className="mt-2 text-xs text-[#8e899b]">Price unit matters: per GPU-hour prices are multiplied by GPUs allocated; per instance-hour prices by instances. Cost uses elapsed hours, so GPU count is applied once.</p>

      <h3 className="mb-3 mt-8 text-sm font-semibold">GPUs and distribution</h3>
      <div className="grid gap-x-4 gap-y-5 sm:grid-cols-2 lg:grid-cols-3"><Field ctx={ctx} spec={spec('hardware.gpu_count_available')} /><Field ctx={ctx} spec={spec('training.gpu_count_used')} /><Field ctx={ctx} spec={spec('training.distributed_strategy')} /></div>
      {avail > 1 && <div className="mt-3"><Alert kind="info">{avail} identical GPUs are available. How many does your current training configuration use? Allocated GPUs are billed even when idle. Multiple GPUs do not imply sharding: choose DDP, ZeRO or FSDP explicitly.</Alert></div>}

      <div className="mt-6"><Accordion title="Advanced hardware settings" hint="TFLOPS, bandwidths, CPU RAM, NVMe, sharding details"><FieldGrid ctx={ctx} section="hardware" advanced /></Accordion></div>

      <div className="mt-6"><Accordion title="How do I find my local hardware specs?" hint="copyable command; nothing runs on your machine from here">
        {command ? <div className="space-y-2"><CopyBlock text={command.linux} label="hardware inspection command" /><CopyBlock text={command.gpu_only} label="GPU-only command" />{command.notes.map((n) => <p key={n} className="text-xs text-[#8e899b]">{n}</p>)}</div> : <p className="text-xs">Command unavailable (backend unreachable).</p>}</Accordion></div>
    </Section>
    {blockers.length > 0 && <Alert kind="error">Resolve before estimating: {blockers.join('; ')}</Alert>}
    <div className="flex justify-between"><button type="button" onClick={onBack} className="border border-line px-5 py-3 text-sm">Back</button>
      <button type="button" onClick={onRun} disabled={busy || blockers.length > 0} className="bg-white px-6 py-3 text-sm font-semibold text-black disabled:opacity-50">{busy ? 'Calculating…' : 'Run prediction'}</button></div>
  </div>;
}
