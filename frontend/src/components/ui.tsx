import { ReactNode, useEffect, useId, useRef, useState } from 'react';
import { AlertCircle, ArrowDown, ArrowUp, Check, ChevronDown, Copy, Info, Lock, type LucideIcon } from 'lucide-react';
import type { FieldSpec, Json } from '../api';
import { FieldState, SOURCE_LABEL } from '../lib/form';

export function Badge({ kind, children }: { kind: 'good' | 'warn' | 'bad' | 'muted'; children: ReactNode }): JSX.Element {
  const tone = { good: 'border-[#2f7a5c] text-[#7ee0b0]', warn: 'border-[#8a7430] text-[#e6d28a]', bad: 'border-[#8a3040] text-[#ff9aa8]', muted: 'border-line text-[#a7a0bb]' }[kind];
  return <span className={`inline-flex items-center gap-1.5 border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${tone}`}>{children}</span>;
}

/** Panel with an icon, a title and a one-line plain-language description of what the section is for. */
export function Section({ number, icon: Icon, title, hint, children, id, actions }: { number?: string; icon?: LucideIcon; title: string; hint?: ReactNode; children: ReactNode; id?: string; actions?: ReactNode }): JSX.Element {
  return <section id={id} className="panel p-4 sm:p-5"><div className="mb-3 flex flex-wrap items-start justify-between gap-2"><div className="flex items-start gap-2.5">
    {Icon ? <span className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center border border-line text-[#8e7cea]" aria-hidden><Icon size={15} /></span> : number && <span className="eyebrow mt-1 text-[#8e7cea]">{number}</span>}
    <div><h2 className="text-base font-semibold">{title}</h2>{hint && <p className="mt-0.5 text-xs text-[#a7a0bb]">{hint}</p>}</div></div>{actions}</div>{children}</section>;
}

/** Signed change vs the original run: arrow shows the direction; green = better, red = worse for that metric. */
export function Delta({ now, before, lowerIsBetter = true, format }: { now: number | null | undefined; before: number | null | undefined; lowerIsBetter?: boolean; format: (n: number) => string }): JSX.Element | null {
  if (now === null || now === undefined || before === null || before === undefined || !Number.isFinite(now) || !Number.isFinite(before)) return null;
  const d = now - before;
  if (Math.abs(d) < 1e-9 * Math.max(1, Math.abs(before))) return <span className="ml-1.5 text-[11px] text-[#8e899b]">no change</span>;
  const better = lowerIsBetter ? d < 0 : d > 0; const Arrow = d > 0 ? ArrowUp : ArrowDown;
  const pct = before !== 0 ? ` (${d > 0 ? '+' : ''}${((d / Math.abs(before)) * 100).toFixed(0)}%)` : '';
  return <span className={`ml-1.5 inline-flex items-center gap-0.5 text-[11px] font-semibold ${better ? 'text-[#7ee0b0]' : 'text-[#ff9aa8]'}`} title="Change from the original estimate">
    <Arrow size={11} aria-hidden />{d > 0 ? '+' : '−'}{format(Math.abs(d))}{pct}<span className="sr-only"> vs original</span></span>;
}

export function Accordion({ title, hint, children, defaultOpen = false }: { title: string; hint?: string; children: ReactNode; defaultOpen?: boolean }): JSX.Element {
  const [open, setOpen] = useState(defaultOpen);
  const id = useId();
  return <div className="border border-line"><button type="button" aria-expanded={open} aria-controls={id} onClick={() => setOpen(!open)} className="flex w-full items-center justify-between px-3 py-2 text-left">
    <span><strong className="text-xs">{title}</strong>{hint && <span className="ml-2 text-[11px] text-[#8e899b]">{hint}</span>}</span><ChevronDown size={14} className={open ? 'rotate-180' : ''} /></button>
    {open && <div id={id} className="border-t border-line p-3">{children}</div>}</div>;
}

export function Unavailable({ children, label = 'Not available' }: { children: ReactNode; label?: string }): JSX.Element {
  return <div className="border border-dashed border-line p-3 text-xs leading-5 text-[#a7a0bb]"><Badge kind="muted">{label}</Badge> <span className="ml-2">{children}</span></div>;
}

export function Alert({ kind, children }: { kind: 'error' | 'warn' | 'info'; children: ReactNode }): JSX.Element {
  const tone = { error: 'border-[#8a3040] bg-[#2a0f16] text-[#ffb3be]', warn: 'border-[#8a7430] bg-[#211b09] text-[#f0dc98]', info: 'border-line bg-[#11111a] text-[#d8d3e3]' }[kind];
  return <div role={kind === 'error' ? 'alert' : 'status'} className={`border px-3 py-2 text-xs ${tone}`}>{children}</div>;
}

export function CopyBlock({ text, label }: { text: string; label: string }): JSX.Element {
  const [copied, setCopied] = useState(false);
  return <div className="flex items-start gap-2 border border-line bg-black/40 p-2"><code className="flex-1 whitespace-pre-wrap break-all font-mono text-[11px] text-[#d8d3e3]">{text}</code>
    <button type="button" aria-label={`Copy ${label}`} onClick={() => { void navigator.clipboard?.writeText(text).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); }); }} className="border border-line p-1.5">{copied ? <Check size={12} /> : <Copy size={12} />}</button></div>;
}

/** Icon that reveals a message popover on hover, focus or click (messages are not printed under the box). */
export function IconTip({ tone, label, children }: { tone: 'error' | 'warn' | 'info' | 'lock'; label: string; children: ReactNode }): JSX.Element {
  const [open, setOpen] = useState(false); const [hover, setHover] = useState(false);
  const ref = useRef<HTMLSpanElement>(null); const id = useId();
  useEffect(() => { if (!open) return; const close = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false); }; document.addEventListener('mousedown', close); return () => document.removeEventListener('mousedown', close); }, [open]);
  const Icon = tone === 'info' ? Info : tone === 'lock' ? Lock : AlertCircle;
  const color = { error: 'text-[#ff7088]', warn: 'text-[#e6c25a]', info: 'text-[#8e899b]', lock: 'text-[#8e899b]' }[tone];
  return <span ref={ref} className="relative inline-flex" onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}>
    <button type="button" aria-label={label} aria-expanded={open || hover} aria-describedby={open || hover ? id : undefined} onClick={() => setOpen(!open)} className={`p-0.5 ${color}`}><Icon size={14} /></button>
    {(open || hover) && <span id={id} role="tooltip" className="absolute right-0 top-6 z-20 w-64 border border-line bg-[#15151f] p-2 text-left text-[11px] font-normal leading-4 text-[#d8d3e3] shadow-glow">{children}</span>}
  </span>;
}

export type FieldMessages = { errors: string[]; warnings: string[] };
const display = (v: Json | undefined): string => (v === null || v === undefined ? '—' : typeof v === 'boolean' ? (v ? 'yes' : 'no') : String(v));

export function SchemaField({ spec, state, messages, requiredFor, lock, onChange, onAccept, onDismiss }: {
  spec: FieldSpec; state?: FieldState; messages: FieldMessages; requiredFor: string[]; lock: string | null;
  onChange: (v: Json) => void; onAccept: () => void; onDismiss: () => void;
}): JSX.Element {
  const id = `field-${spec.key}`;
  const value = state?.value ?? null;
  const empty = value === null || value === '';
  const blockingMissing = empty && requiredFor.includes('memory');
  const flagged = !!state && !state.reviewed && (state.status === 'uncertain' || state.status === 'unresolved');
  const problems = [...messages.errors, ...(blockingMissing ? ['Required for the memory estimate.'] : [])];
  const red = problems.length > 0 || flagged;
  const cls = `field-sm ${red ? '!border-[#d0455c]' : state?.pending ? '!border-[#c9a83a]' : ''} ${lock ? 'opacity-70' : ''}`;
  let input: JSX.Element;
  if (spec.type === 'select' || spec.type === 'bool') {
    const opts = spec.type === 'bool' ? ['true', 'false'] : (spec.options ?? []);
    input = <select id={id} className={cls} value={spec.type === 'bool' ? (value === null ? '' : String(value)) : String(value ?? '')} disabled={!!lock} aria-invalid={red}
      onChange={(e) => onChange(e.target.value === '' ? null : spec.type === 'bool' ? e.target.value === 'true' : e.target.value)}>
      <option value="">—</option>{opts.map((o) => <option key={o} value={o}>{spec.type === 'bool' ? (o === 'true' ? 'Yes' : 'No') : o.replace(/_/g, ' ')}</option>)}</select>;
  } else {
    input = <input id={id} className={cls} type={spec.type === 'text' ? 'text' : 'number'} inputMode={spec.type === 'text' ? undefined : 'decimal'} step={spec.type === 'int' ? 1 : 'any'}
      min={spec.minimum ?? undefined} value={value === null ? '' : String(value)} disabled={!!lock} aria-invalid={red}
      onChange={(e) => { const raw = e.target.value; onChange(raw === '' ? null : spec.type === 'text' ? raw : Number(raw)); }} />;
  }
  const tone = red ? 'error' : state?.pending || messages.warnings.length ? 'warn' : lock ? 'lock' : 'info';
  const source = state && state.source !== 'manual' ? `From ${SOURCE_LABEL[state.source]}${state.detail ? ` · ${state.detail}` : ''}.` : state ? 'Entered manually.' : null;
  const showTip = red || state?.pending || messages.warnings.length > 0 || lock || source || spec.help;
  return <div data-field={spec.key} className="min-w-0">
    <div className="mb-1 flex items-center justify-between gap-1">
      <label htmlFor={id} className="truncate text-[11px] font-medium text-[#c4bfcd]" title={spec.label}>{spec.label}{spec.unit && <span className="text-[#8e899b]"> ({spec.unit})</span>}
        {requiredFor.length > 0 && <span className={requiredFor.includes('memory') ? 'text-[#ff7088]' : 'text-[#e6d28a]'} title={requiredFor.includes('memory') ? 'required' : `needed for ${requiredFor.join(' & ')}`}> *</span>}</label>
      {showTip && <IconTip tone={tone} label={`Details for ${spec.label}`}>
        {problems.map((m) => <span key={m} className="block text-[#ff9aa8]">{m}</span>)}
        {flagged && <span className="block text-[#ff9aa8]">{state?.status === 'unresolved' ? 'Could not be resolved statically' : 'Uncertain'}{state?.message ? `: ${state.message}` : ''}. Please review.</span>}
        {messages.warnings.filter((m) => m !== state?.message).map((m) => <span key={m} className="block text-[#e6d28a]">{m}</span>)}
        {state?.pending && <span className="block text-[#f0dc98]">Your value is kept. {SOURCE_LABEL[state.pending.source]} found <strong>{display(state.pending.value)}</strong>{state.pending.detail ? ` (${state.pending.detail})` : ''}.
          <button type="button" onClick={onAccept} className="ml-1 underline">Use extracted</button><button type="button" onClick={onDismiss} className="ml-1 underline">Keep mine</button></span>}
        {lock && <span className="block">{lock}</span>}
        {source && <span className="block text-[#a7a0bb]">{source}</span>}
        {spec.help && <span className="block text-[#8e899b]">{spec.help}</span>}
        {requiredFor.length > 0 && !requiredFor.includes('memory') && <span className="block text-[#8e899b]">Needed for the {requiredFor.join(' and ')} estimate.</span>}
      </IconTip>}
    </div>
    {input}
  </div>;
}

/** Locked v1 assumption shown as a one-option dropdown. */
export function FixedSelect({ label, value, hint }: { label: string; value: string; hint?: string | null }): JSX.Element {
  const id = useId();
  return <div className="min-w-0"><label htmlFor={id} className="mb-1 block truncate text-[11px] text-[#a7a0bb]" title={hint ? `Later: ${hint}` : undefined}>{label}</label>
    <select id={id} className="field-sm opacity-80" value={value} onChange={() => undefined} aria-describedby={hint ? `${id}-h` : undefined}><option value={value}>{value}</option></select>
    {hint && <span id={`${id}-h`} className="sr-only">Locked in v1. Later: {hint}</span>}</div>;
}

export function MemoryBar({ used, usable, total, label }: { used: number; usable: number; total: number | null; label: string }): JSX.Element {
  const scale = Math.max(total ?? usable, used);
  const pct = (v: number) => Math.min(100, (v / scale) * 100);
  const ofUsable = (used / usable) * 100; const over = used > usable;
  return <div>
    <div className="mb-1 flex flex-wrap items-baseline justify-between gap-2"><span className={`text-3xl font-semibold ${over ? 'text-[#ff9aa8]' : 'text-white'}`}>{ofUsable.toFixed(1)}%</span>
      <span className="text-xs text-[#c4bfcd]">{label} ÷ usable VRAM limit per GPU ({usable.toFixed(2)} GiB{total ? ` of ${total.toFixed(2)} GiB total` : ''})</span></div>
    <div role="img" aria-label={`${used.toFixed(2)} GiB is ${ofUsable.toFixed(1)} percent of ${usable.toFixed(2)} GiB usable VRAM`} className="relative h-5 border border-line bg-[#0c0c13]">
      <div className={`h-full ${over ? 'bg-[#d0455c]' : ofUsable > 90 ? 'bg-[#c9a83a]' : 'bg-violet'}`} style={{ width: `${pct(used)}%` }} />
      <div className="absolute inset-y-0 w-0.5 bg-[#e6d28a]" style={{ left: `${pct(usable)}%` }} title="usable limit" />
    </div>
    <div className="mt-1 flex justify-between text-[10px] text-[#8e899b]"><span>0 GiB</span><span>▲ usable limit {usable.toFixed(1)} GiB</span><span>{total ? `total ${total.toFixed(1)} GiB` : ''}</span></div>
  </div>;
}
