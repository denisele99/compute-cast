import { useEffect, useState } from 'react';
import { HardwareOption, HardwareSearch, api } from '../api';
import { AssumptionsPanel, FieldGrid, FormCtx } from '../components/FieldGrid';
import { Accordion, Alert, Badge, CopyBlock, Section } from '../components/ui';

const unitLabel = (u: string | null | undefined) => (u === 'gpu_hour' ? '/GPU-hour' : u === 'instance_hour' ? '/instance-hour' : ' (unit unknown)');
export function priceText(o: HardwareOption): string {
  if (!o.price || o.price.value === null) return 'no price';
  return `$${o.price.value.toFixed(2)}${unitLabel(o.price.unit)}${o.price.preemptible ? ' spot' : ''}`;
}

export function HardwarePage({ ctx, onSelect, onCustom, onBack, onRun, busy, blockers }: {
  ctx: FormCtx; onSelect: (o: HardwareOption) => Promise<void>; onCustom: () => void; onBack: () => void; onRun: () => void; busy: boolean; blockers: string[];
}): JSX.Element {
  const [q, setQ] = useState(''); const [offset, setOffset] = useState(0);
  const [result, setResult] = useState<HardwareSearch | null>(null); const [error, setError] = useState(''); const [loading, setLoading] = useState(false);
  const [command, setCommand] = useState<{ linux: string; gpu_only: string; notes: string[] } | null>(null);
  const selected = ctx.values['hardware.gpu_id'] as string | null; const custom = ctx.values['hardware.source'] === 'custom';
  useEffect(() => {
    const t = setTimeout(() => { setLoading(true); setError('');
      api.hardwareSearch(q, offset).then(setResult).catch((e) => setError(e instanceof Error ? e.message : String(e))).finally(() => setLoading(false)); }, 200);
    return () => clearTimeout(t);
  }, [q, offset]);
  useEffect(() => { api.localCommand().then(setCommand).catch(() => setCommand(null)); }, []);
  return <div className="space-y-3">
    <Section number="04" title="Hardware" hint="Specs: general GPU catalogue. Availability and price: Nebius data files (kept separate). One GPU in v1.">
      <div className="flex gap-2">
        <input className="field-sm" aria-label="Search GPUs and Nebius platforms" placeholder="Search: H100, H200 us-central1, L40S, A100…" value={q} onChange={(e) => { setQ(e.target.value); setOffset(0); }} />
        <button type="button" onClick={onCustom} className={`whitespace-nowrap border px-3 text-xs ${custom ? 'border-violet bg-violet/15' : 'border-line'}`}>Custom GPU</button>
      </div>
      {loading && <p className="mt-2 text-xs text-[#a7a0bb]" aria-live="polite">Loading catalogue…</p>}
      {error && <div className="mt-2"><Alert kind="error">{error}</Alert></div>}
      {result && <>
        <ul className="mt-2 grid gap-1 sm:grid-cols-2" aria-label="Hardware search results">{result.items.map((o) => <li key={o.id}>
          <button type="button" onClick={() => void onSelect(o)} aria-pressed={selected === o.id} className={`flex w-full flex-wrap items-center gap-x-2 border px-2 py-1.5 text-left text-xs ${selected === o.id ? 'border-violet bg-violet/10' : 'border-line hover:border-white/40'}`}>
            <strong className="mr-auto">{o.label}{o.price?.region ? ` · ${o.price.region}` : ''}</strong>
            <span>{o.vram_gib !== null ? `${o.vram_gib} GiB` : 'VRAM ?'}</span><span className="text-[#c4bfcd]">{priceText(o)}</span>
            {!o.specs && <Badge kind="warn">no specs</Badge>}{o.provider === 'catalog' && <Badge kind="muted">catalogue</Badge>}
            {o.issues.length > 0 && <span className="w-full truncate text-[10px] text-[#e6d28a]" title={o.issues.join(' ')}>{o.issues[0]}</span>}
          </button></li>)}</ul>
        {result.items.length === 0 && <p className="mt-2 text-xs text-[#e6d28a]">No catalogue match for “{q}”. Use Custom GPU.</p>}
        <div className="mt-1 flex items-center justify-between text-[11px] text-[#8e899b]"><span>{result.total ? offset + 1 : 0}–{Math.min(offset + 10, result.total)} of {result.total} · prices {result.price_snapshot ?? 'unknown date'} (preemptible)</span>
          <span className="flex gap-1"><button type="button" className="border border-line px-2 disabled:opacity-40" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 10))}>Prev</button>
            <button type="button" className="border border-line px-2 disabled:opacity-40" disabled={offset + 10 >= result.total} onClick={() => setOffset(offset + 10)}>Next</button></span></div>
      </>}
      <div className="mt-3"><FieldGrid ctx={ctx} section="hardware" /></div>
      <p className="mt-2 text-[11px] text-[#8e899b]">Cost = elapsed hours × 1 GPU × price per GPU-hour (a 1-GPU instance-hour is the same). Usable limit defaults to 100%: the estimate already adds a runtime baseline and margin.</p>
      <div className="mt-3 space-y-2"><AssumptionsPanel schema={ctx.schema} sections={['hardware']} />
        <Accordion title="Find my local GPU specs" hint="copyable command; nothing runs from here">
          {command ? <div className="space-y-1"><CopyBlock text={command.linux} label="hardware inspection command" /><CopyBlock text={command.gpu_only} label="GPU-only command" />{command.notes.map((n) => <p key={n} className="text-[11px] text-[#8e899b]">{n}</p>)}</div> : <p className="text-xs">Unavailable (backend unreachable).</p>}</Accordion></div>
    </Section>
    {blockers.length > 0 && <Alert kind="error"><strong>{blockers.length} item(s) to resolve:</strong> {blockers.slice(0, 6).join('; ')}{blockers.length > 6 ? '; …' : ''}</Alert>}
    <div className="flex justify-between"><button type="button" onClick={onBack} className="border border-line px-4 py-2 text-sm">Back</button>
      <button type="button" onClick={onRun} disabled={busy || blockers.length > 0} className="bg-white px-5 py-2 text-sm font-semibold text-black disabled:opacity-50">{busy ? 'Calculating…' : 'Run estimate'}</button></div>
  </div>;
}
