import { KeyboardEvent, useEffect, useId, useRef, useState } from 'react';
import { Cpu, List, Search, SlidersHorizontal, Wrench } from 'lucide-react';
import { HardwareOption, HardwareSearch, OptionSummary, api } from '../api';
import { AssumptionsPanel, FieldGrid, FormCtx } from '../components/FieldGrid';
import { Accordion, Alert, Badge, CopyBlock, Section } from '../components/ui';

const unitLabel = (u: string | null | undefined) => (u === 'gpu_hour' ? '/GPU-hour' : u === 'instance_hour' ? '/instance-hour' : ' (unit unknown)');
export function priceText(o: { price: HardwareOption['price'] }): string {
  if (!o.price || o.price.value === null) return 'no price';
  return `$${o.price.value.toFixed(2)}${unitLabel(o.price.unit)}${o.price.preemptible ? ' spot' : ''}`;
}
const optionLabel = (o: { label: string; price: HardwareOption['price']; vram_gib: number | null }) =>
  `${o.label}${o.price?.region ? ` · ${o.price.region}` : ''} · ${o.vram_gib !== null ? `${o.vram_gib} GiB` : 'VRAM ?'} · ${priceText(o)}`;

/** Type-ahead search over Nebius AI Cloud platforms and the GPU catalogue, with a dropdown of matches. */
function HardwareCombobox({ selected, onSelect }: { selected: string | null; onSelect: (o: HardwareOption) => void }): JSX.Element {
  const [q, setQ] = useState(''); const [open, setOpen] = useState(false); const [active, setActive] = useState(0);
  const [result, setResult] = useState<HardwareSearch | null>(null); const [error, setError] = useState(''); const [loading, setLoading] = useState(false);
  const listId = useId(); const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const t = setTimeout(() => { setLoading(true); setError('');
      api.hardwareSearch(q).then((r) => { setResult(r); setActive(0); }).catch((e) => setError(e instanceof Error ? e.message : String(e))).finally(() => setLoading(false)); }, 150);
    return () => clearTimeout(t);
  }, [q, open]);
  useEffect(() => { const close = (e: MouseEvent) => { if (!box.current?.contains(e.target as Node)) setOpen(false); }; document.addEventListener('mousedown', close); return () => document.removeEventListener('mousedown', close); }, []);
  const items = result?.items ?? [];
  const choose = (o: HardwareOption) => { onSelect(o); setQ(''); setOpen(false); };
  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setOpen(true); setActive((a) => Math.min(a + 1, items.length - 1)); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive((a) => Math.max(a - 1, 0)); }
    else if (e.key === 'Enter' && open && items[active]) { e.preventDefault(); choose(items[active]); }
    else if (e.key === 'Escape') setOpen(false);
  };
  return <div ref={box} className="relative">
    <label className="mb-1 block text-xs font-medium text-[#c4bfcd]" htmlFor={`${listId}-input`}>Search Nebius AI Cloud and GPU catalogue</label>
    <div className="flex items-center gap-2 border border-line bg-black/40 px-2"><Search size={14} className="text-[#8e899b]" aria-hidden />
      <input id={`${listId}-input`} role="combobox" aria-expanded={open} aria-controls={listId} aria-autocomplete="list" autoComplete="off"
        aria-activedescendant={open && items[active] ? `${listId}-${active}` : undefined} className="w-full bg-transparent py-1.5 text-sm outline-none"
        placeholder="Type to search: H100, H200 us-central1, L40S, B200…" value={q} onFocus={() => setOpen(true)} onChange={(e) => { setQ(e.target.value); setOpen(true); }} onKeyDown={onKey} /></div>
    {open && <ul id={listId} role="listbox" aria-label="Hardware search results" className="absolute z-30 mt-1 max-h-80 w-full overflow-y-auto border border-line bg-[#15151f] shadow-glow">
      {loading && <li className="px-3 py-2 text-xs text-[#a7a0bb]">Searching…</li>}
      {error && <li className="px-3 py-2 text-xs text-[#ff9aa8]">{error}</li>}
      {!loading && items.length === 0 && <li className="px-3 py-2 text-xs text-[#e6d28a]">No match for “{q}”. Use Custom GPU.</li>}
      {items.map((o, i) => <li key={o.id} id={`${listId}-${i}`} role="option" aria-selected={i === active} onMouseEnter={() => setActive(i)} onMouseDown={(e) => { e.preventDefault(); choose(o); }}
        className={`flex cursor-pointer flex-wrap items-center gap-x-2 px-3 py-1.5 text-xs ${i === active ? 'bg-violet/20' : ''} ${selected === o.id ? 'text-white' : ''}`}>
        <strong className="mr-auto">{o.label}{o.price?.region ? ` · ${o.price.region}` : ''}</strong>
        <span>{o.vram_gib !== null ? `${o.vram_gib} GiB` : 'VRAM ?'}</span><span className="text-[#c4bfcd]">{priceText(o)}</span>
        <Badge kind="muted">{o.provider === 'nebius' ? 'Nebius' : 'catalogue'}</Badge>{!o.specs && <Badge kind="warn">no specs</Badge>}
        {o.issues.length > 0 && <span className="w-full truncate text-[10px] text-[#e6d28a]" title={o.issues.join(' ')}>{o.issues[0]}</span>}</li>)}
      {result && result.total > items.length && <li className="px-3 py-1.5 text-[10px] text-[#8e899b]">Showing {items.length} of {result.total}. Type more to narrow, or use the full list below.</li>}
    </ul>}
  </div>;
}

export function HardwarePage({ ctx, onSelect, onCustom, onBack, onRun, busy, blockers }: {
  ctx: FormCtx; onSelect: (o: HardwareOption) => Promise<void>; onCustom: () => void; onBack: () => void; onRun: () => void; busy: boolean; blockers: string[];
}): JSX.Element {
  const [all, setAll] = useState<OptionSummary[]>([]); const [snapshot, setSnapshot] = useState<string | null>(null);
  const [command, setCommand] = useState<{ linux: string; gpu_only: string; notes: string[] } | null>(null);
  const selected = ctx.values['hardware.gpu_id'] as string | null; const custom = ctx.values['hardware.source'] === 'custom';
  useEffect(() => { api.hardwareAll().then((r) => { setAll(r.items); setSnapshot(r.price_snapshot); }).catch(() => setAll([])); }, []);
  useEffect(() => { api.localCommand().then(setCommand).catch(() => setCommand(null)); }, []);
  const pick = (id: string) => { if (!id) return; void api.hardwareOption(id).then((r) => onSelect(r.option)); };
  const current = all.find((o) => o.id === selected);
  return <div className="space-y-3">
    <Section icon={Cpu} title="Hardware" hint="Pick the GPU you plan to train on. We fill in its memory, speed and price for you; you can override any value.">
      <div className="grid gap-3 lg:grid-cols-[1.3fr_1fr]">
        <HardwareCombobox selected={selected} onSelect={(o) => void onSelect(o)} />
        <div><label className="mb-1 flex items-center gap-1.5 text-xs font-medium text-[#c4bfcd]" htmlFor="preloaded-gpus"><List size={13} aria-hidden /> Or choose from the pre-loaded list</label>
          <select id="preloaded-gpus" className="field-sm w-full" value={current ? current.id : ''} onChange={(e) => pick(e.target.value)}>
            <option value="">{all.length ? `Select one of ${all.length} GPUs…` : 'Loading…'}</option>
            <optgroup label="Nebius AI Cloud (platform · region)">{all.filter((o) => o.provider === 'nebius').map((o) => <option key={o.id} value={o.id}>{optionLabel(o)}</option>)}</optgroup>
            <optgroup label="General GPU catalogue (no provider price)">{all.filter((o) => o.provider === 'catalog').map((o) => <option key={o.id} value={o.id}>{optionLabel(o)}</option>)}</optgroup>
          </select></div>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-[#8e899b]">
        {current ? <span>Selected: <strong className="text-white">{current.label}</strong>{current.price?.region ? ` · ${current.price.region}` : ''}</span> : custom ? <span>Custom GPU: enter its values below.</span> : <span>No GPU selected yet.</span>}
        <button type="button" onClick={onCustom} className={`ml-auto flex items-center gap-1 border px-2 py-1 text-xs ${custom ? 'border-violet bg-violet/15 text-white' : 'border-line'}`}><Wrench size={12} aria-hidden /> Custom GPU</button></div>
      <p className="mt-1 text-[10px] text-[#8e899b]">Nebius prices are preemptible (spot) snapshots{snapshot ? ` from ${snapshot.slice(0, 10)}` : ''}. Missing Nebius specs are filled from the general GPU catalogue.</p>

      <div className="mt-3"><FieldGrid ctx={ctx} section="hardware" /></div>
      <p className="mt-2 text-[11px] text-[#8e899b]">Price starts at $0 until you pick a priced Nebius GPU or type your own. Usable limit defaults to 100%: the estimate already adds a runtime baseline and margin.</p>
      <div className="mt-3 space-y-2">
        <Accordion title="Advanced hardware settings (optional)" hint="peak speed and bandwidth; filled from the catalogue, needed only for the runtime estimate">
          <div className="flex items-start gap-2"><SlidersHorizontal size={14} className="mt-1 text-[#8e899b]" aria-hidden /><div className="flex-1"><FieldGrid ctx={ctx} section="hardware" advanced /></div></div>
          <p className="mt-2 text-[11px] text-[#8e899b]">Leave these empty if you don't know them: memory still gets estimated; only training time and cost need them.</p></Accordion>
        <AssumptionsPanel schema={ctx.schema} sections={['hardware']} />
        <Accordion title="Find my local GPU specs" hint="copyable command; nothing runs from here">
          {command ? <div className="space-y-1"><CopyBlock text={command.linux} label="hardware inspection command" /><CopyBlock text={command.gpu_only} label="GPU-only command" />{command.notes.map((n) => <p key={n} className="text-[11px] text-[#8e899b]">{n}</p>)}</div> : <p className="text-xs">Unavailable (backend unreachable).</p>}</Accordion></div>
    </Section>
    {blockers.length > 0 && <Alert kind="error"><strong>{blockers.length} item(s) to resolve:</strong> {blockers.slice(0, 6).join('; ')}{blockers.length > 6 ? '; …' : ''}</Alert>}
    <div className="flex justify-between"><button type="button" onClick={onBack} className="border border-line px-4 py-2 text-sm">Back</button>
      <button type="button" onClick={onRun} disabled={busy || blockers.length > 0} className="bg-white px-5 py-2 text-sm font-semibold text-black disabled:opacity-50">{busy ? 'Calculating…' : 'Run estimate'}</button></div>
  </div>;
}
