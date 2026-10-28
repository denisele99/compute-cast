import { useMemo, useState } from 'react';
import { Badge, NumField, Section, TextField, Unavailable } from '../components/ui';
import { HardwareOption } from '../api';

export type Hw = { mode: 'catalog' | 'custom'; optionId: string; name: string; vram: number | ''; limit: number | ''; count: number | ''; price: number | ''; cpuRam: number | ''; nvme: number | '' };
export const initialHw: Hw = { mode: 'catalog', optionId: '', name: 'Custom GPU', vram: 24, limit: '', count: 1, price: 0, cpuRam: '', nvme: '' };
export const optionName = (o: HardwareOption): string => `${o.label}${o.region ? ` · ${o.region}` : ''}`;

export function HardwarePage({ hw, setHw, options, note, loading, loadError, onBack, onRun, busy, error }: {
  hw: Hw; setHw: (h: Hw) => void; options: HardwareOption[]; note: string; loading: boolean; loadError: string; onBack: () => void; onRun: () => void; busy: boolean; error: string;
}): JSX.Element {
  const [q, setQ] = useState('');
  const shown = useMemo(() => options.filter((o) => optionName(o).toLowerCase().includes(q.toLowerCase())), [options, q]);
  const pick = (o: HardwareOption) => setHw({ ...hw, optionId: o.id, name: optionName(o), vram: o.vram_gib ?? '', limit: '', price: o.hourly_price_usd ?? 0 });
  const f = <K extends keyof Hw>(k: K) => (v: Hw[K]) => setHw({ ...hw, [k]: v });
  return <div className="space-y-6"><Section number="04" title="Hardware" hint="VRAM is per GPU. CPU RAM and storage are per node.">
    <div className="mb-6 flex gap-2" role="tablist">{(['catalog', 'custom'] as const).map((m) => <button key={m} role="tab" aria-selected={hw.mode === m} onClick={() => f('mode')(m)} className={`border px-4 py-2 text-sm ${hw.mode === m ? 'border-violet bg-violet/15' : 'border-line text-[#a7a0bb]'}`}>{m === 'catalog' ? 'Cloud GPU list' : 'Custom hardware'}</button>)}</div>
    {hw.mode === 'catalog' ? <>
      <TextField label="Search GPUs" value={q} onChange={setQ} placeholder="H100, B200, A100…" />
      {loading && <p className="mt-4 text-sm text-[#a7a0bb]" aria-live="polite">Loading hardware…</p>}
      {loadError && <p role="alert" className="mt-4 text-sm text-[#ffb3be]">{loadError}</p>}
      <ul className="mt-4 grid gap-2 sm:grid-cols-2">{shown.map((o) => <li key={o.id}><button onClick={() => pick(o)} aria-pressed={hw.optionId === o.id} className={`w-full border p-4 text-left ${hw.optionId === o.id ? 'border-violet bg-violet/10' : 'border-line hover:border-white/40'}`}>
        <strong className="block text-sm">{optionName(o)}</strong><span className="mt-2 flex flex-wrap gap-2 text-xs text-[#c4bfcd]"><span>{o.vram_gib ? `${o.vram_gib} GB VRAM` : 'VRAM unavailable'}</span><span>{o.hourly_price_usd !== null ? `$${o.hourly_price_usd}/GPU-hr (preemptible)` : 'no price source'}</span><Badge kind="muted">{o.provider}</Badge></span></button></li>)}</ul>
      {note && <p className="mt-3 text-xs text-[#8e899b]">{note}</p>}<div className="mt-3"><Unavailable>AWS and other providers can be added once their catalogs exist.</Unavailable></div></> : <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      <TextField label="GPU model" value={hw.name} onChange={f('name')} /><NumField label="VRAM per GPU" value={hw.vram} min={1} suffix="GB" onChange={f('vram')} /></div>}
    <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      <NumField label="Memory limit per GPU" value={hw.limit} min={1} suffix="GB" optional onChange={f('limit')} help="Empty uses the full VRAM." />
      <NumField label="Number of GPUs" value={hw.count} min={1} onChange={f('count')} help="Peak VRAM is per GPU; multi-GPU speed-up is not modeled." />
      <NumField label="Price per GPU-hour" value={hw.price} min={0} step={0.01} suffix="USD" optional onChange={f('price')} help="Default 0. Token Factory per-token pricing comes later." />
      <NumField label="CPU RAM per node" value={hw.cpuRam} min={1} suffix="GB" optional onChange={f('cpuRam')} /><NumField label="NVMe storage per node" value={hw.nvme} min={1} suffix="GB" optional onChange={f('nvme')} />
    </div>
    <div className="mt-6"><Unavailable>“How do I find my hardware info?” (nvidia-smi helper and config upload) is planned. For now, enter values manually.</Unavailable></div>
  </Section>
  {error && <p role="alert" className="border border-[#8a3040] bg-[#2a0f16] p-4 text-sm text-[#ffb3be]">{error}</p>}
  <div className="flex justify-between"><button onClick={onBack} className="border border-line px-5 py-3 text-sm">Back</button><button onClick={onRun} disabled={busy} className="bg-white px-6 py-3 text-sm font-semibold text-black disabled:opacity-50">{busy ? 'Calculating…' : 'Estimate'}</button></div></div>;
}
