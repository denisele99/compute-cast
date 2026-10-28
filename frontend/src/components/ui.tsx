import { ReactNode, useId, useState } from 'react';
import { Check, ChevronDown, Copy } from 'lucide-react';
import type { FieldSpec, Json } from '../api';
import { FieldState, SOURCE_LABEL } from '../lib/form';

export function Badge({ kind, children }: { kind: 'good' | 'warn' | 'bad' | 'muted'; children: ReactNode }): JSX.Element {
  const tone = { good: 'border-[#2f7a5c] text-[#7ee0b0]', warn: 'border-[#8a7430] text-[#e6d28a]', bad: 'border-[#8a3040] text-[#ff9aa8]', muted: 'border-line text-[#a7a0bb]' }[kind];
  return <span className={`inline-flex items-center gap-2 border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wider ${tone}`}>{children}</span>;
}

export function Section({ number, title, hint, children, id }: { number: string; title: string; hint?: ReactNode; children: ReactNode; id?: string }): JSX.Element {
  return <section id={id} className="panel p-5 sm:p-8"><div className="mb-6 flex items-baseline gap-4"><span className="eyebrow text-[#8e7cea]">{number}</span><div><h2 className="text-xl font-semibold">{title}</h2>{hint && <div className="mt-1 text-sm text-[#a7a0bb]">{hint}</div>}</div></div>{children}</section>;
}

export function TextField({ label, value, onChange, placeholder, disabled }: { label: string; value: string; onChange: (v: string) => void; placeholder?: string; disabled?: boolean }): JSX.Element {
  const id = useId();
  return <div><label htmlFor={id} className="mb-2 block text-xs font-medium text-[#c4bfcd]">{label}</label><input id={id} className="field disabled:opacity-50" value={value} placeholder={placeholder} disabled={disabled} onChange={(e) => onChange(e.target.value)} /></div>;
}

export function Accordion({ title, hint, children, defaultOpen = false }: { title: string; hint?: string; children: ReactNode; defaultOpen?: boolean }): JSX.Element {
  const [open, setOpen] = useState(defaultOpen);
  const id = useId();
  return <div className="border border-line"><button type="button" aria-expanded={open} aria-controls={id} onClick={() => setOpen(!open)} className="flex w-full items-center justify-between p-4 text-left"><span><strong className="text-sm">{title}</strong>{hint && <span className="ml-3 text-xs text-[#8e899b]">{hint}</span>}</span><ChevronDown size={16} className={open ? 'rotate-180' : ''} /></button>{open && <div id={id} className="border-t border-line p-4">{children}</div>}</div>;
}

export function Unavailable({ children, label = 'Not available' }: { children: ReactNode; label?: string }): JSX.Element {
  return <div className="border border-dashed border-line p-3 text-xs leading-5 text-[#a7a0bb]"><Badge kind="muted">{label}</Badge> <span className="ml-2">{children}</span></div>;
}

export function Alert({ kind, children }: { kind: 'error' | 'warn' | 'info'; children: ReactNode }): JSX.Element {
  const tone = { error: 'border-[#8a3040] bg-[#2a0f16] text-[#ffb3be]', warn: 'border-[#8a7430] bg-[#211b09] text-[#f0dc98]', info: 'border-line bg-[#11111a] text-[#d8d3e3]' }[kind];
  return <div role={kind === 'error' ? 'alert' : 'status'} className={`border p-4 text-sm ${tone}`}>{children}</div>;
}

export function CopyBlock({ text, label }: { text: string; label: string }): JSX.Element {
  const [copied, setCopied] = useState(false);
  return <div className="flex items-start gap-2 border border-line bg-black/40 p-3"><code className="flex-1 whitespace-pre-wrap break-all font-mono text-xs text-[#d8d3e3]">{text}</code>
    <button type="button" aria-label={`Copy ${label}`} onClick={() => { void navigator.clipboard?.writeText(text).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); }); }} className="border border-line p-2 text-xs">{copied ? <Check size={14} /> : <Copy size={14} />}</button></div>;
}

export type FieldMessages = { errors: string[]; warnings: string[]; unsupported: string[] };

const display = (v: Json | undefined): string => (v === null || v === undefined ? '—' : typeof v === 'boolean' ? (v ? 'yes' : 'no') : String(v));

export function SchemaField({ spec, state, messages, requiredFor, onChange, onAccept, onDismiss, onReview, optionLabels }: {
  spec: FieldSpec; state?: FieldState; messages: FieldMessages; requiredFor: string[]; onChange: (v: Json) => void;
  onAccept: () => void; onDismiss: () => void; onReview: () => void; optionLabels?: Record<string, string>;
}): JSX.Element {
  const id = `field-${spec.key}`;
  const value = state?.value ?? null;
  const empty = value === null || value === '';
  const blockingMissing = empty && requiredFor.includes('memory');
  const flagged = !!state && !state.reviewed && (state.status === 'uncertain' || state.status === 'unresolved');
  const red = blockingMissing || messages.errors.length > 0 || messages.unsupported.length > 0 || flagged;
  const cls = `field disabled:opacity-50 ${red ? '!border-[#d0455c] ring-1 ring-[#d0455c]/40' : ''}`;
  const describedBy = `${id}-msg`;
  let input: JSX.Element;
  if (spec.type === 'select' || spec.type === 'bool') {
    const opts = spec.type === 'bool' ? ['true', 'false'] : (spec.options ?? []);
    const current = spec.type === 'bool' ? (value === null ? '' : String(value)) : String(value ?? '');
    input = <select id={id} className={cls} value={current} aria-invalid={red} aria-describedby={describedBy}
      onChange={(e) => onChange(e.target.value === '' ? null : spec.type === 'bool' ? e.target.value === 'true' : e.target.value)}>
      <option value="">{spec.type === 'bool' ? 'Choose…' : 'Select…'}</option>
      {opts.map((o) => <option key={o} value={o}>{spec.type === 'bool' ? (o === 'true' ? 'Yes' : 'No') : optionLabels?.[o] ?? o}</option>)}</select>;
  } else {
    input = <div className="relative"><input id={id} className={`${cls} ${spec.unit ? 'pr-16' : ''}`} type={spec.type === 'text' ? 'text' : 'number'} inputMode={spec.type === 'text' ? undefined : 'decimal'}
      step={spec.type === 'int' ? 1 : 'any'} min={spec.minimum ?? undefined} value={value === null ? '' : String(value)} aria-invalid={red} aria-describedby={describedBy}
      onChange={(e) => { const raw = e.target.value; onChange(raw === '' ? null : spec.type === 'text' ? raw : Number(raw)); }} />
      {spec.unit && <span className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-xs text-[#8e899b]">{spec.unit}</span>}</div>;
  }
  const req = requiredFor.includes('memory') ? 'required' : requiredFor.length ? `needed for ${requiredFor.join(' & ')}` : '';
  return <div data-field={spec.key}>
    <label htmlFor={id} className="mb-2 flex flex-wrap items-baseline gap-x-2 text-xs font-medium text-[#c4bfcd]">
      <span>{spec.label}{requiredFor.length > 0 && <span className={requiredFor.includes('memory') ? 'text-[#ff7088]' : 'text-[#e6d28a]'} title={req}> *</span>}</span>
      {requiredFor.length > 0 && !requiredFor.includes('memory') && <span className="text-[10px] text-[#8e899b]">({req})</span>}
    </label>
    {input}
    <div id={describedBy} className="mt-1 space-y-1 text-xs">
      {state && state.source !== 'manual' && <p className="text-[#8e899b]">From {SOURCE_LABEL[state.source]}{state.detail ? ` · ${state.detail}` : ''}
        {!state.reviewed && state.value !== null && <button type="button" onClick={onReview} className="ml-2 underline decoration-dotted">mark reviewed</button>}
        {state.reviewed && <span className="ml-2 text-[#7ee0b0]">reviewed</span>}</p>}
      {flagged && <p className="text-[#ff9aa8]">{state?.status === 'unresolved' ? 'Could not be resolved statically' : 'Uncertain'}{state?.message ? `: ${state.message}` : ''}. Please review.</p>}
      {blockingMissing && !flagged && <p className="text-[#ff9aa8]">Required for the memory estimate.</p>}
      {messages.errors.map((m) => <p key={m} className="text-[#ff9aa8]">{m}</p>)}
      {messages.unsupported.map((m) => <p key={m} className="text-[#ff9aa8]">Unsupported: {m}</p>)}
      {messages.warnings.filter((m) => !(flagged && m === state?.message)).map((m) => <p key={m} className="text-[#e6d28a]">{m}</p>)}
      {state?.pending && <div className="border border-[#8a7430] p-2 text-[#f0dc98]">Your value is kept. {SOURCE_LABEL[state.pending.source]} found <strong>{display(state.pending.value)}</strong>{state.pending.detail ? ` (${state.pending.detail})` : ''}.
        <button type="button" onClick={onAccept} className="ml-2 underline">Use extracted</button><button type="button" onClick={onDismiss} className="ml-2 underline">Keep mine</button></div>}
      {spec.help && <p className="text-[#8e899b]">{spec.help}</p>}
    </div>
  </div>;
}

export function MemoryBar({ peak, usable, total }: { peak: number; usable: number; total: number | null }): JSX.Element {
  const scale = Math.max(total ?? usable, peak);
  const pct = (v: number) => Math.min(100, (v / scale) * 100);
  const ofUsable = (peak / usable) * 100;
  const over = peak > usable;
  return <div>
    <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2"><span className={`text-4xl font-semibold ${over ? 'text-[#ff9aa8]' : 'text-white'}`}>{ofUsable.toFixed(1)}%</span>
      <span className="text-sm text-[#c4bfcd]">of the usable VRAM limit per GPU ({usable.toFixed(2)} GiB{total ? ` of ${total.toFixed(2)} GiB total` : ''})</span></div>
    <div role="img" aria-label={`Peak ${peak.toFixed(2)} GiB is ${ofUsable.toFixed(1)} percent of ${usable.toFixed(2)} GiB usable VRAM`} className="relative h-6 border border-line bg-[#0c0c13]">
      <div className={`h-full ${over ? 'bg-[#d0455c]' : ofUsable > 90 ? 'bg-[#c9a83a]' : 'bg-violet'}`} style={{ width: `${pct(peak)}%` }} />
      <div className="absolute inset-y-0 w-0.5 bg-[#e6d28a]" style={{ left: `${pct(usable)}%` }} title="usable limit" />
    </div>
    <div className="mt-2 flex justify-between text-[11px] text-[#8e899b]"><span>0 GiB</span><span>▲ usable limit {usable.toFixed(1)} GiB</span><span>{total ? `total ${total.toFixed(1)} GiB` : ''}</span></div>
  </div>;
}
