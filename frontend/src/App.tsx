import { ChangeEvent, useEffect, useState } from 'react';
import { Menu } from 'lucide-react';
import { ArrowRight } from 'lucide-react';
import { EstimateResponse, HardwareOption, MemoryResult, estimateMemory, fetchHardware, requestEstimate } from './api';
import { HardwarePage, Hw, initialHw } from './pages/Hardware';
import { RecommendationsPage, ScenarioState } from './pages/Recommendations';
import { CompareRow, ResultsPage } from './pages/Results';
import { WorkloadPage } from './pages/Workload';
import { Scenario, Setup } from './lib/scenarios';
import { MAX_FILE_BYTES, ParsedWorkload, buildWorkload, parseCollector, parseWorkloadJson } from './lib/workloadFile';

type Step = 'workload' | 'hardware' | 'results' | 'recs';
const stepLabels: [Step, string][] = [['workload', 'Workload'], ['hardware', 'Hardware'], ['results', 'Results'], ['recs', 'Recommendations']];
const initialSetup: Setup = { batch_size_per_gpu: 16, epochs: 10, gradient_accumulation_steps: 1, precision: 'fp32', optimizer: 'AdamW', gradient_checkpointing: false, num_examples: 100000, sequence_length: '', input_type: 'text / tokens', distributed_strategy: 'single', seconds_per_batch: '' };
const message = (e: unknown): string => (e instanceof Error ? e.message : 'Unexpected error.');

function validate(s: Setup, hw: Hw): string[] {
  const errs: string[] = []; const whole = (name: string, v: number | '') => { if (v === '' || !Number.isInteger(v) || v < 1) errs.push(`${name} must be a whole number of at least 1.`); };
  whole('Batch size', s.batch_size_per_gpu); whole('Epochs', s.epochs); whole('Gradient accumulation', s.gradient_accumulation_steps); whole('Training examples', s.num_examples); whole('Number of GPUs', hw.count);
  if (hw.vram === '' || hw.vram <= 0) errs.push('Choose a GPU or enter VRAM per GPU.');
  if (hw.limit !== '' && (hw.limit <= 0 || (hw.vram !== '' && hw.limit > hw.vram))) errs.push('Memory limit must be above 0 and not exceed VRAM per GPU.');
  if (s.seconds_per_batch !== '' && s.seconds_per_batch <= 0) errs.push('Seconds per batch must be greater than 0.');
  if (hw.price !== '' && hw.price < 0) errs.push('Price cannot be negative.');
  return errs;
}

export default function App(): JSX.Element {
  const [screen, setScreen] = useState<'landing' | 'planner'>('landing');
  const [step, setStep] = useState<Step>('workload');
  const [setup, setSetup] = useState<Setup>(initialSetup);
  const [collector, setCollector] = useState<Record<string, unknown> | null>(null);
  const [extracted, setExtracted] = useState<ParsedWorkload | null>(null);
  const [fileName, setFileName] = useState(''); const [fileMessage, setFileMessage] = useState('');
  const [hw, setHw] = useState<Hw>(initialHw);
  const [options, setOptions] = useState<HardwareOption[]>([]); const [hwNote, setHwNote] = useState(''); const [hwLoading, setHwLoading] = useState(false); const [hwError, setHwError] = useState('');
  const [mem, setMem] = useState<MemoryResult | null>(null); const [memError, setMemError] = useState('');
  const [runtime, setRuntime] = useState<EstimateResponse | null>(null); const [runtimeNote, setRuntimeNote] = useState('');
  const [compare, setCompare] = useState<CompareRow[]>([]); const [scen, setScen] = useState<Record<string, ScenarioState>>({});
  const [busy, setBusy] = useState(false); const [error, setError] = useState('');
  const set = <K extends keyof Setup>(k: K, v: Setup[K]): void => { setSetup((s) => ({ ...s, [k]: v })); setError(''); };
  const capacity = Number(hw.limit || hw.vram || 0);
  const hardwareBody = () => ({ gpu: { vram_gib_per_gpu: capacity }, ...(hw.cpuRam !== '' ? { cpu: { ram_gib: Number(hw.cpuRam) } } : {}), ...(hw.nvme !== '' ? { storage: { capacity_gib: Number(hw.nvme) } } : {}) });

  useEffect(() => {
    if (step !== 'hardware' || options.length > 0 || hwLoading) return;
    setHwLoading(true); setHwError('');
    fetchHardware().then((r) => { setOptions(r.items); setHwNote(r.note); }).catch((e) => setHwError(message(e))).finally(() => setHwLoading(false));
  }, [step]); // eslint-disable-line react-hooks/exhaustive-deps

  function go(next: Step): void { setStep(next); setError(''); window.scrollTo({ top: 0, behavior: 'smooth' }); }

  function onFile(event: ChangeEvent<HTMLInputElement>): void {
    const file = event.target.files?.[0]; if (!file) return; setFileName(file.name); setExtracted(null);
    if (!file.name.toLowerCase().endsWith('.json')) { setFileMessage('Choose a JSON file. Python files are never read or executed.'); return; }
    if (file.size > MAX_FILE_BYTES) { setFileMessage('The file is larger than 5 MB.'); return; }
    const reader = new FileReader(); reader.onerror = () => setFileMessage('The file could not be read.');
    reader.onload = () => {
      const text = String(reader.result); const parsed = parseWorkloadJson(text); const raw = parseCollector(text);
      if (!parsed.ok || !raw) { setCollector(null); setFileMessage(parsed.ok ? 'The file has no "model" section from collect_workload.py.' : parsed.error); return; }
      const t = (raw.training ?? {}) as Record<string, unknown>; const f = parsed.data.fields;
      setCollector(raw); setExtracted(parsed.data); setFileMessage('Loaded. Review the extracted values and complete anything missing.');
      setSetup((s) => ({ ...s, num_examples: f.examples ?? s.num_examples, epochs: f.epochs ?? s.epochs, batch_size_per_gpu: f.batch_size ?? s.batch_size_per_gpu, sequence_length: f.sequence_length ?? s.sequence_length,
        gradient_accumulation_steps: typeof t.gradient_accumulation_steps === 'number' ? t.gradient_accumulation_steps : s.gradient_accumulation_steps, precision: typeof t.precision === 'string' ? t.precision : s.precision,
        optimizer: typeof t.optimizer === 'string' ? t.optimizer : s.optimizer, gradient_checkpointing: typeof t.gradient_checkpointing === 'boolean' ? t.gradient_checkpointing : s.gradient_checkpointing }));
    };
    reader.readAsText(file);
  }

  function next(): void { const errs = validate(setup, { ...hw, vram: hw.vram === '' ? 1 : hw.vram }); const own = errs.filter((e) => !e.includes('GPU')); if (own.length) { setError(own.join(' ')); return; } go('hardware'); }

  async function run(s: Setup = setup): Promise<void> {
    if (busy) return; const errs = validate(s, hw); if (errs.length) { setError(errs.join(' ')); return; }
    setBusy(true); setError(''); setMem(null); setMemError(''); setRuntime(null); setCompare([]); setScen({});
    try {
      if (collector) { try { setMem(await estimateMemory({ workload: buildWorkload(collector, s, Number(hw.count)), hardware: hardwareBody() })); } catch (e) { setMemError(message(e)); } }
      else setMemError('Upload the collect_workload.py JSON on the Workload step so the backend has the tensor inventory it needs.');
      if (s.seconds_per_batch === '') setRuntimeNote('Needs a measured seconds-per-batch benchmark. Enter one on the Workload step.');
      else if (Number(hw.count) > 1) setRuntimeNote('Multi-GPU runtime is not modeled yet.');
      else { try { setRuntime(await requestEstimate({ examples: s.num_examples, batch_size: s.batch_size_per_gpu, epochs: s.epochs, sequence_length: Number(s.sequence_length) || 1, seconds_per_batch: s.seconds_per_batch, hourly_price_usd: Number(hw.price || 0), measured_peak_vram_gb: null, target_vram_gb: capacity })); } catch (e) { setRuntimeNote(message(e)); } }
      go('results');
      if (collector) {
        const rows = options.filter((o) => o.vram_gib).slice(0, 10).map((o): CompareRow => ({ id: o.id, label: `${o.label}${o.region ? ` · ${o.region}` : ''}`, vram: o.vram_gib, price: o.hourly_price_usd }));
        const settled = await Promise.allSettled(rows.map((r) => estimateMemory({ workload: buildWorkload(collector, s, Number(hw.count)), hardware: { gpu: { vram_gib_per_gpu: r.vram as number } } })));
        setCompare(rows.map((r, i) => { const x = settled[i]; return x.status === 'fulfilled' ? { ...r, mem: x.value } : { ...r, error: message(x.reason) }; }));
      }
    } finally { setBusy(false); }
  }

  async function recalc(sc: Scenario): Promise<void> {
    const changed = sc.change(setup); if (!changed || !collector) return;
    setScen((p) => ({ ...p, [sc.id]: { status: 'loading' } }));
    try { const mem2 = await estimateMemory({ workload: buildWorkload(collector, changed, Number(hw.count)), hardware: hardwareBody() }); setScen((p) => ({ ...p, [sc.id]: { status: 'done', mem: mem2 } })); }
    catch (e) { setScen((p) => ({ ...p, [sc.id]: { status: 'error', error: message(e) } })); }
  }
  function apply(sc: Scenario): void { const changed = sc.change(setup); if (!changed) return; setSetup(changed); void run(changed); }

  if (screen === 'landing') return <LandingPage onStart={() => { setScreen('planner'); setStep('workload'); }} />;
  const reached = (s: Step) => s === 'workload' || s === 'hardware' || mem !== null || memError !== '' || runtime !== null || runtimeNote !== '';
  return <div className="min-h-screen bg-ink text-white"><TopBar onLogo={() => setScreen('landing')} onBack={() => setScreen('landing')} />
    <main className="mx-auto max-w-[1180px] px-5 py-10 sm:px-8 lg:px-10"><h1 className="text-3xl font-semibold tracking-tight">Plan your training run</h1><p className="mt-2 text-sm text-[#a7a0bb]">Estimate memory, runtime and cost before you commit compute.</p>
      <nav aria-label="Steps" className="my-8 flex flex-wrap gap-2">{stepLabels.map(([id, label], i) => <button key={id} disabled={!reached(id)} aria-current={step === id ? 'step' : undefined} onClick={() => go(id)} className={`border px-4 py-2 text-xs ${step === id ? 'border-violet bg-violet/15 text-white' : 'border-line text-[#a7a0bb] disabled:opacity-40'}`}>{i + 1}. {label}</button>)}</nav>
      {step === 'workload' && <WorkloadPage setup={setup} set={set} extracted={extracted} fileName={fileName} fileMessage={fileMessage} onFile={onFile} hasCollector={collector !== null} onNext={next} error={error} />}
      {step === 'hardware' && <HardwarePage hw={hw} setHw={setHw} options={options} note={hwNote} loading={hwLoading} loadError={hwError} onBack={() => go('workload')} onRun={() => void run()} busy={busy} error={error} />}
      {step === 'results' && <ResultsPage mem={mem} memError={memError} runtime={runtime} runtimeNote={runtimeNote} gpuCount={Number(hw.count)} compare={compare} onEdit={() => go('workload')} onRecs={() => go('recs')} />}
      {step === 'recs' && <RecommendationsPage setup={setup} baseline={mem} states={scen} canRun={collector !== null} onRun={(s) => void recalc(s)} onApply={apply} onBack={() => go('results')} />}
    </main></div>;
}

function TopBar({ onLogo, onBack }: { onLogo: () => void; onBack: () => void }): JSX.Element { return <header className="border-b border-white/10 bg-ink/80 backdrop-blur-xl"><div className="mx-auto flex max-w-[1180px] items-center justify-between px-5 py-5 sm:px-8 lg:px-10"><button onClick={onLogo} className="flex items-center gap-3" aria-label="Return to planner start"><span className="grid h-8 w-8 place-items-center bg-gradient-to-br from-violet to-cyan text-xs font-extrabold text-black">CC</span><span className="text-sm font-bold tracking-[0.14em]">COMPUTE <span className="text-violet">CAST</span></span></button><nav className="hidden items-center gap-8 text-xs text-[#a7a0bb] md:flex"><button onClick={onBack} className="transition hover:text-white">Overview</button><span>Estimator</span><span>Model review <sup className="text-violet">soon</sup></span></nav><button className="text-[#a7a0bb] md:hidden" aria-label="Open navigation"><Menu size={20} /></button></div></header>; }
function Landing({ onStart }: { onStart: () => void }): JSX.Element { return <section className="relative min-h-screen overflow-hidden bg-black"><div className="absolute inset-0" style={{ backgroundImage: 'radial-gradient(ellipse 42% 54% at 2% 30%, rgba(199,43,177,.82) 0%, rgba(80,19,105,.5) 38%, transparent 72%), radial-gradient(ellipse 64% 48% at 55% 84%, rgba(24,28,206,.82) 0%, rgba(22,7,123,.72) 42%, transparent 76%), radial-gradient(ellipse 50% 32% at 80% 108%, rgba(47,220,214,.78) 0%, rgba(8,79,148,.56) 34%, transparent 70%), radial-gradient(ellipse 35% 52% at 104% 48%, rgba(20,25,194,.9) 0%, rgba(10,5,72,.62) 58%, transparent 100%), linear-gradient(145deg,#09000e 0%,#00000a 43%,#02001d 70%,#02000b 100%)' }} /><div className="absolute -right-24 top-12 h-[42vh] w-[52vw] rounded-[45%] bg-black/75 blur-2xl" /><div className="relative mx-auto flex min-h-screen max-w-[1280px] flex-col px-6 pb-10 pt-8 sm:px-10 lg:px-16"><LandingNav onStart={onStart} /><div className="flex flex-1 items-center pb-20 pt-28 sm:pt-36"><div className="max-w-4xl"><h1 className="text-[clamp(3.5rem,10vw,8.8rem)] font-extrabold leading-[.88] tracking-[-.075em] text-[#eeeeee]">COMPUTE <span className="text-[#8e7cea]">CAST</span></h1><p className="mt-9 max-w-[610px] text-lg font-light leading-[1.28] text-[#e0dce8] sm:text-2xl">estimate GPU resources and recommend optimizations for model training</p><p className="mt-5 max-w-[520px] text-sm leading-6 text-[#bdb7cc]">Know whether your model fits <strong className="font-semibold text-white">before you rent the GPUs</strong>. Get VRAM, runtime and cost estimates, then the cheapest fixes.</p><div className="mt-10 flex flex-col items-start gap-4 sm:flex-row sm:items-center"><button onClick={onStart} className="group flex items-center gap-3 bg-white px-6 py-4 text-sm font-semibold text-black transition hover:bg-violet hover:text-white">Open estimator <ArrowRight size={17} className="transition group-hover:translate-x-1" /></button></div></div></div></div></section>; }
function LandingNav({ onStart }: { onStart: () => void }): JSX.Element { return <nav className="flex items-center justify-between"><div className="flex items-center gap-3"><span className="grid h-8 w-8 place-items-center border border-white/30 text-[10px] font-bold">CC</span><span className="text-xs font-bold tracking-[.18em]">COMPUTE CAST</span></div><button onClick={onStart} className="border border-white/30 px-4 py-2 text-xs text-white transition hover:border-white hover:bg-white hover:text-black">Launch tool</button></nav>; }
function HowItWorks({ onStart }: { onStart: () => void }): JSX.Element {
  const items = [{ n: '01', title: 'Upload your workload', body: 'Run collect_workload.py on your model, then drop in the JSON. We read parameters, precision and optimizer.', tag: 'collect_workload.json' },
    { n: '02', title: 'Choose your hardware', body: 'Pick a GPU from the list with VRAM and price shown, or enter your own machine.', tag: 'GPU · VRAM · $/hr' },
    { n: '03', title: 'See the fit, then optimize', body: 'Get peak VRAM against your limit, runtime and cost, plus three ways to improve the plan.', tag: 'Fits · Tight · Exceeds' }];
  return <main className="bg-black px-6 pb-24 pt-20 sm:px-10 lg:px-16"><div className="mx-auto max-w-[1180px]"><p className="eyebrow text-[#8e7cea]">HOW IT WORKS</p><h2 className="mt-3 max-w-2xl text-3xl font-semibold tracking-tight sm:text-4xl">Three steps from model to <span className="text-[#8e7cea]">compute plan</span></h2>
    <div className="mt-10 grid gap-4 md:grid-cols-3">{items.map((i) => <article key={i.n} className="border border-white/15 bg-white/[0.03] p-6"><span className="text-xs font-bold text-[#8e7cea]">{i.n}</span><h3 className="mt-4 text-lg font-semibold">{i.title}</h3><p className="mt-3 text-sm leading-6 text-[#bdb7cc]">{i.body}</p><p className="mt-5 inline-block border border-white/20 px-2 py-1 font-mono text-[11px] text-[#a7a0bb]">{i.tag}</p></article>)}</div>
    <div className="mt-10 border border-white/15 p-6" aria-label="Example result, illustrative"><p className="eyebrow">EXAMPLE RESULT (ILLUSTRATIVE)</p><div className="mt-4 h-4 border border-white/20"><div className="h-full w-[62%] bg-[#7258dd]" /></div><div className="mt-2 flex justify-between text-xs text-[#a7a0bb]"><span>Peak VRAM</span><span>Usable limit</span></div></div>
    <button onClick={onStart} className="mt-10 bg-white px-6 py-4 text-sm font-semibold text-black transition hover:bg-violet hover:text-white">Open estimator</button></div></main>;
}
function LandingPage({ onStart }: { onStart: () => void }): JSX.Element { return <><Landing onStart={onStart} /><HowItWorks onStart={onStart} /></>; }
