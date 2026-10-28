import { ReactNode, useId, useState } from 'react';
import { ChevronDown } from 'lucide-react';

export function Badge({ kind, children }: { kind: 'good' | 'warn' | 'bad' | 'muted'; children: ReactNode }): JSX.Element {
  const tone = { good: 'border-[#2f7a5c] text-[#7ee0b0]', warn: 'border-[#8a7430] text-[#e6d28a]', bad: 'border-[#8a3040] text-[#ff9aa8]', muted: 'border-line text-[#a7a0bb]' }[kind];
  return <span className={`inline-flex items-center gap-2 border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wider ${tone}`}>{children}</span>;
}

export function Section({ number, title, hint, children }: { number: string; title: string; hint?: string; children: ReactNode }): JSX.Element {
  return <section className="panel p-5 sm:p-8"><div className="mb-6 flex items-baseline gap-4"><span className="eyebrow text-[#8e7cea]">{number}</span><div><h2 className="text-xl font-semibold">{title}</h2>{hint && <p className="mt-1 text-sm text-[#a7a0bb]">{hint}</p>}</div></div>{children}</section>;
}

export function NumField({ label, value, onChange, min = 0, step = 1, suffix, optional, disabled, help }: { label: string; value: number | ''; onChange: (v: number | '') => void; min?: number; step?: number; suffix?: string; optional?: boolean; disabled?: boolean; help?: string }): JSX.Element {
  const id = useId();
  return <div><label htmlFor={id} className="mb-2 block text-xs font-medium text-[#c4bfcd]">{label}{optional && <span className="ml-2 text-[#8e899b]">optional</span>}</label>
    <div className="relative"><input id={id} className="field pr-14 disabled:opacity-50" type="number" inputMode="decimal" min={min} step={step} value={value} disabled={disabled} aria-describedby={help ? `${id}-h` : undefined}
      onChange={(e) => onChange(e.target.value === '' ? '' : Number(e.target.value))} />{suffix && <span className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-xs text-[#8e899b]">{suffix}</span>}</div>
    {help && <p id={`${id}-h`} className="mt-1 text-xs text-[#8e899b]">{help}</p>}</div>;
}

export function TextField({ label, value, onChange, placeholder, disabled }: { label: string; value: string; onChange: (v: string) => void; placeholder?: string; disabled?: boolean }): JSX.Element {
  const id = useId();
  return <div><label htmlFor={id} className="mb-2 block text-xs font-medium text-[#c4bfcd]">{label}</label><input id={id} className="field disabled:opacity-50" value={value} placeholder={placeholder} disabled={disabled} onChange={(e) => onChange(e.target.value)} /></div>;
}

export function SelectField({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (v: string) => void }): JSX.Element {
  const id = useId();
  return <div><label htmlFor={id} className="mb-2 block text-xs font-medium text-[#c4bfcd]">{label}</label><select id={id} className="field" value={value} onChange={(e) => onChange(e.target.value)}>{options.map((o) => <option key={o}>{o}</option>)}</select></div>;
}

export function Toggle({ label, hint, checked, onChange }: { label: string; hint: string; checked: boolean; onChange: (v: boolean) => void }): JSX.Element {
  return <label className="flex cursor-pointer items-center justify-between gap-4 border border-line bg-[#0c0c13] p-4"><span><strong className="block text-sm">{label}</strong><span className="text-xs text-[#8e899b]">{hint}</span></span><input type="checkbox" className="h-5 w-5 accent-[#7258dd]" checked={checked} onChange={(e) => onChange(e.target.checked)} /></label>;
}

export function Accordion({ title, hint, children }: { title: string; hint?: string; children: ReactNode }): JSX.Element {
  const [open, setOpen] = useState(false);
  const id = useId();
  return <div className="border border-line"><button type="button" aria-expanded={open} aria-controls={id} onClick={() => setOpen(!open)} className="flex w-full items-center justify-between p-4 text-left"><span><strong className="text-sm">{title}</strong>{hint && <span className="ml-3 text-xs text-[#8e899b]">{hint}</span>}</span><ChevronDown size={16} className={open ? 'rotate-180' : ''} /></button>{open && <div id={id} className="border-t border-line p-4">{children}</div>}</div>;
}

export function Unavailable({ children }: { children: ReactNode }): JSX.Element {
  return <p className="border border-dashed border-line p-3 text-xs leading-5 text-[#a7a0bb]"><Badge kind="muted">Not available yet</Badge> <span className="ml-2">{children}</span></p>;
}

export function MemoryBar({ peak, usable, total }: { peak: number | null; usable: number | null; total: number | null }): JSX.Element {
  const cap = total ?? usable ?? 1; const pct = (v: number | null) => (v === null ? 0 : Math.min(100, (v / cap) * 100));
  return <div role="img" aria-label={`Peak ${peak?.toFixed(2)} of ${usable?.toFixed(2)} GiB usable`}><div className="relative h-4 border border-line bg-[#0c0c13]"><div className="h-full bg-violet" style={{ width: `${pct(peak)}%` }} />{usable !== null && <div className="absolute inset-y-0 w-px bg-[#e6d28a]" style={{ left: `${pct(usable)}%` }} />}</div>
    <div className="mt-2 flex justify-between text-[11px] text-[#8e899b]"><span>0</span><span>usable {usable?.toFixed(1)} GiB</span><span>total {total?.toFixed(1)} GiB</span></div></div>;
}
