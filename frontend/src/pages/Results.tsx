import { EstimateResponse, MemoryResult, mockMode } from '../api';
import { Accordion, Badge, MemoryBar, Section, Unavailable } from '../components/ui';
import { classifyFit } from '../lib/fit';

export type CompareRow = { id: string; label: string; vram: number | null; price: number | null; mem?: MemoryResult; error?: string };
const tone = { fits: 'good', tight: 'warn', exceeds: 'bad', unknown: 'muted' } as const;
const gib = (v: number | null | undefined) => (v === null || v === undefined ? '—' : `${v.toFixed(2)} GiB`);

export function ResultsPage({ mem, memError, runtime, runtimeNote, gpuCount, compare, onEdit, onRecs }: {
  mem: MemoryResult | null; memError: string; runtime: EstimateResponse | null; runtimeNote: string; gpuCount: number; compare: CompareRow[]; onEdit: () => void; onRecs: () => void;
}): JSX.Element {
  const fit = mem ? classifyFit(mem.gpu.peak_vram_gib, mem.gpu.usable_vram_gib) : null;
  return <div className="space-y-6"><Section number="05" title="Results" hint="Planning estimates, not measurements.">
    <div className="mb-4 flex flex-wrap gap-2">{mockMode ? <Badge kind="warn">Demo data</Badge> : <Badge kind="muted">Estimate · not benchmarked</Badge>}</div>
    {memError && <div role="alert" className="border border-[#8a3040] bg-[#2a0f16] p-4 text-sm text-[#ffb3be]"><strong>VRAM estimate unavailable.</strong> {memError}</div>}
    {fit && mem && <div className={`border p-5 ${fit.kind === 'exceeds' ? 'border-[#8a3040]' : fit.kind === 'tight' ? 'border-[#8a7430]' : 'border-line'}`}>
      <div className="flex flex-wrap items-center justify-between gap-3"><Badge kind={tone[fit.kind]}>{fit.label}</Badge><span className="text-xs text-[#a7a0bb]">{fit.utilization !== null && `${(fit.utilization * 100).toFixed(0)}% of usable VRAM`}</span></div>
      <p className="mt-3 text-sm">{fit.message}</p>
      <div className="mt-5"><MemoryBar peak={mem.gpu.peak_vram_gib} usable={mem.gpu.usable_vram_gib} total={mem.gpu.total_vram_gib} /></div>
      <dl className="mt-5 grid gap-4 sm:grid-cols-3 text-sm"><div><dt className="eyebrow">Peak VRAM / GPU</dt><dd className="mt-1 text-xl font-semibold">{gib(mem.gpu.peak_vram_gib)}</dd></div><div><dt className="eyebrow">Usable VRAM / GPU</dt><dd className="mt-1 text-xl font-semibold">{gib(mem.gpu.usable_vram_gib)}</dd></div><div><dt className="eyebrow">Backend fit</dt><dd className="mt-1 text-xl font-semibold">{mem.gpu.fits === null ? 'Unknown' : mem.gpu.fits ? 'Fits' : 'Does not fit'}</dd></div></dl>
      {(mem.cpu || mem.nvme) && <dl className="mt-5 grid gap-4 border-t border-line pt-4 text-sm sm:grid-cols-2">{mem.cpu && <div><dt className="eyebrow">CPU RAM per node</dt><dd>{gib(mem.cpu.required_ram_gib_per_node)} needed · {mem.cpu.available_gib_per_node === null ? 'available not entered' : `${gib(mem.cpu.available_gib_per_node)} available`}</dd></div>}{mem.nvme && <div><dt className="eyebrow">NVMe per node</dt><dd>{gib(mem.nvme.required_storage_gib_per_node)} needed · {mem.nvme.available_gib_per_node === null ? 'available not entered' : `${gib(mem.nvme.available_gib_per_node)} available`}</dd></div>}</dl>}
      {mem.warnings.map((w) => <p key={w} className="mt-3 text-xs text-[#e6d28a]">{w}</p>)}</div>}
    <div className="mt-6 grid gap-4 sm:grid-cols-2"><div className="border border-line p-5"><p className="eyebrow">Training runtime</p>{runtime ? <p className="mt-2 text-2xl font-semibold">{runtime.training_hours} h <span className="text-sm font-normal text-[#8e899b]">{runtime.training_steps.toLocaleString('en-US')} batches</span></p> : <p className="mt-2 text-sm text-[#a7a0bb]">{runtimeNote}</p>}</div>
      <div className="border border-line p-5"><p className="eyebrow">Estimated cost</p>{runtime ? <p className="mt-2 text-2xl font-semibold">${runtime.training_cost_usd.toLocaleString('en-US')}</p> : <p className="mt-2 text-sm text-[#a7a0bb]">{runtimeNote}</p>}{runtime && <p className="mt-1 text-xs text-[#8e899b]">Compute only, {gpuCount} GPU.</p>}</div></div>
  </Section>
  <Section number="06" title="Hardware comparison" hint="The same workload checked against each listed GPU by the backend.">
    {compare.length === 0 ? <Unavailable>Comparison needs a collector file and the hardware list.</Unavailable> : <div className="overflow-x-auto"><table className="w-full min-w-[560px] text-left text-sm"><thead className="text-xs text-[#8e899b]"><tr><th className="py-2">Configuration</th><th>VRAM</th><th>Fit</th><th>Price / GPU-hr</th></tr></thead><tbody>{compare.map((r) => { const f = r.mem ? classifyFit(r.mem.gpu.peak_vram_gib, r.mem.gpu.usable_vram_gib) : null; return <tr key={r.id} className="border-t border-line"><td className="py-3">{r.label}</td><td>{r.vram ? `${r.vram} GB` : '—'}</td><td>{f ? <Badge kind={tone[f.kind]}>{f.label}</Badge> : <span className="text-xs text-[#a7a0bb]">{r.error ?? 'No VRAM data'}</span>}</td><td>{r.price !== null ? `$${r.price}` : '—'}</td></tr>; })}</tbody></table></div>}
  </Section>
  <Section number="07" title="Benchmark" hint="A short measured run makes runtime and cost estimates much more reliable.">
    <div className="grid gap-3 sm:grid-cols-2"><Unavailable>Run on Nebius: needs POST /api/benchmarks (currently 501) and Nebius sign-in.</Unavailable><Unavailable>Generate a benchmark script for your own server: not built yet. It would need a dataset sample.</Unavailable></div>
  </Section>
  <Accordion title="How we calculated this"><ul className="list-disc space-y-2 pl-5 text-sm text-[#c4bfcd]"><li>Peak VRAM comes from the Python estimator: weights, gradients, optimizer states, activations, plus unsourced planning allowances for workspace, runtime and fragmentation. It is not a measured peak.</li><li>Fit compares peak to usable VRAM (limit minus other usage). Peaks within 10% of the limit are flagged.</li><li>Runtime = batches × your measured seconds per batch. Cost = runtime × price. Single GPU only.</li>{runtime?.assumptions.map((a) => <li key={a}>{a}</li>)}</ul></Accordion>
  <div className="flex justify-between"><button onClick={onEdit} className="border border-line px-5 py-3 text-sm">Edit inputs</button><button onClick={onRecs} className="bg-violet px-6 py-3 text-sm font-semibold text-white">Improve this plan</button></div></div>;
}
