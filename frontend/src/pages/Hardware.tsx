import { useEffect, useState } from 'react';
import { Cloud, Cpu, Database, Wrench } from 'lucide-react';
import { HardwareOption, OptionSummary, api } from '../api';
import { FieldGrid, FormCtx } from '../components/FieldGrid';
import { Accordion, Alert, CopyBlock, Section } from '../components/ui';

type Source = 'nebius' | 'catalog' | 'custom';
const unitLabel = (u: string | null | undefined) => (u === 'gpu_hour' ? '/GPU-hour' : u === 'instance_hour' ? '/instance-hour' : '');
export function priceText(o: { price: HardwareOption['price'] }): string {
  if (!o.price || o.price.value === null) return 'no price';
  return `$${o.price.value.toFixed(2)}${unitLabel(o.price.unit)}${o.price.preemptible ? ' spot' : ''}`;
}
const optionLabel = (o: OptionSummary) => [o.label, o.price?.region, o.vram_gib !== null ? `${o.vram_gib} GiB` : null, o.provider === 'nebius' ? priceText(o) : null].filter(Boolean).join(' · ');
const SOURCES: [Source, string, typeof Cloud][] = [['nebius', 'Nebius AI Cloud', Cloud], ['catalog', 'GPU catalogue', Database], ['custom', 'Custom GPU (optional)', Wrench]];
const SPEC_KEYS = ['hardware.gpu_name', 'hardware.vram_gib', 'hardware.usable_vram_gib', 'hardware.bf16_supported', 'hardware.bf16_dense_tflops',
  'hardware.fp32_tflops', 'hardware.memory_bandwidth_gbps', 'hardware.price_per_hour', 'hardware.price_unit'];

export function HardwarePage({ ctx, onSelect, onCustom, onBack, onRun, busy, blockers }: {
  ctx: FormCtx; onSelect: (o: HardwareOption) => Promise<void>; onCustom: () => void; onBack: () => void; onRun: () => void; busy: boolean; blockers: string[];
}): JSX.Element {
  const [groups, setGroups] = useState<{ nebius: OptionSummary[]; catalog: OptionSummary[] }>({ nebius: [], catalog: [] });
  const [command, setCommand] = useState<{ linux: string; gpu_only: string; notes: string[] } | null>(null);
  const selected = ctx.values['hardware.gpu_id'] as string | null;
  const current: Source = ctx.values['hardware.source'] === 'custom' ? 'custom' : selected?.startsWith('catalog:') ? 'catalog' : 'nebius';
  const [source, setSource] = useState<Source>(current);
  useEffect(() => { api.hardwareAll().then((r) => setGroups({ nebius: r.nebius, catalog: r.catalog })).catch(() => undefined); }, []);
  useEffect(() => { if (source === 'custom') api.localCommand().then(setCommand).catch(() => setCommand(null)); }, [source]);
  const pick = (id: string) => { if (id) void api.hardwareOption(id).then((r) => onSelect(r.option)); };
  const choose = (s: Source) => { setSource(s); if (s === 'custom') onCustom(); };
  const list = source === 'catalog' ? groups.catalog : groups.nebius;
  const base = [...groups.nebius, ...groups.catalog].find((o) => o.id === selected);
  const edited = SPEC_KEYS.filter((k) => ctx.form[k]?.source === 'manual').length;
  return <div className="space-y-3">
    <Section icon={Cpu} title="Hardware" hint="Pick a GPU to fill in its specs, then change any value you like. Switch GPUs at any time, or skip the list and type your own values.">
      <div className="flex flex-wrap gap-1" role="radiogroup" aria-label="GPU source">{SOURCES.map(([id, label, Icon]) =>
        <button key={id} type="button" role="radio" aria-checked={source === id} onClick={() => choose(id)}
          className={`flex items-center gap-1.5 border px-3 py-1.5 text-xs transition ${source === id ? 'border-violet bg-violet/15 text-white' : 'border-line text-[#a7a0bb] hover:text-white'}`}><Icon size={13} aria-hidden />{label}</button>)}</div>
      {source !== 'custom' && <select aria-label={source === 'nebius' ? 'Nebius AI Cloud GPU' : 'Catalogue GPU'} className="field-sm mt-3 w-full max-w-xl"
        value={list.some((o) => o.id === selected) ? selected ?? '' : ''} onChange={(e) => pick(e.target.value)}>
        <option value="">Select a GPU…</option>
        {list.map((o) => <option key={o.id} value={o.id}>{optionLabel(o)}</option>)}
      </select>}
      <p className="mt-2 text-[11px] text-[#a7a0bb]" aria-live="polite">{base
        ? <>Values from <strong className="text-white">{base.label}</strong>{edited > 0 && <> · {edited} edited by you <button type="button" onClick={() => selected && pick(selected)} className="ml-1 underline">restore {base.label} specs</button></>}. Every field below stays editable; picking another GPU replaces them.</>
        : source === 'custom' ? 'Enter your GPU\'s specs below.' : 'No GPU picked yet: choose one above or type the values below.'}</p>
      <div className="mt-3"><FieldGrid ctx={ctx} section="hardware" /><div className="mt-2"><FieldGrid ctx={ctx} section="hardware" advanced /></div></div>
      {source === 'custom' && <div className="mt-3"><Accordion title="Find my local GPU specs">
        {command ? <div className="space-y-1"><CopyBlock text={command.linux} label="hardware inspection command" /><CopyBlock text={command.gpu_only} label="GPU-only command" /></div> : <p className="text-xs">Unavailable.</p>}</Accordion></div>}
    </Section>
    {blockers.length > 0 && <Alert kind="error"><strong>{blockers.length} item(s) to resolve:</strong> {blockers.slice(0, 6).join('; ')}{blockers.length > 6 ? '; …' : ''}</Alert>}
    <div className="flex justify-between"><button type="button" onClick={onBack} className="border border-line px-4 py-2 text-sm">Back</button>
      <button type="button" onClick={onRun} disabled={busy || blockers.length > 0} className="bg-white px-5 py-2 text-sm font-semibold text-black disabled:opacity-50">{busy ? 'Calculating…' : 'Run estimate'}</button></div>
  </div>;
}
