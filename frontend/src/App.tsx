import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowRight, Menu, RotateCcw } from 'lucide-react';
import { BenchmarkPlan, BenchmarkResult, CompareRow, ConfigBody, Extraction, HardwareOption, Json, Prediction, Schema, UploadedFile, Validation, api } from './api';
import { FormCtx } from './components/FieldGrid';
import { Alert } from './components/ui';
import { FormState, acceptPending, applyCatalogue, applyFindings, applySuggestions, dismissPending, initialForm, markReviewed, needsReview, setManual, toProvenance, toValues, visible } from './lib/form';
import { Scenario } from './lib/scenarios';
import { HardwarePage } from './pages/Hardware';
import { RecommendationsPage, ScenarioState } from './pages/Recommendations';
import { Async, ResultsPage } from './pages/Results';
import { WorkloadPage } from './pages/Workload';

type Step = 'workload' | 'hardware' | 'results' | 'recs';
const stepLabels: [Step, string][] = [['workload', 'Workload'], ['hardware', 'Hardware'], ['results', 'Results'], ['recs', 'Recommendations']];
const message = (e: unknown): string => (e instanceof Error ? e.message : String(e));
const CATALOGUE_KEYS = ['hardware.gpu_name', 'hardware.total_vram_gib', 'hardware.usable_vram_gib', 'hardware.catalog_gpu_id', 'hardware.peak_tflops_fp32', 'hardware.peak_tflops_bf16',
  'hardware.peak_tflops_fp16', 'hardware.memory_bandwidth_gb_s', 'hardware.price_per_hour', 'hardware.price_unit', 'hardware.price_region', 'hardware.price_note', 'hardware.gpus_per_instance'];

function Planner({ schema }: { schema: Schema }): JSX.Element {
  const [analysisId, setAnalysisId] = useState(''); const [sessionError, setSessionError] = useState('');
  const [step, setStep] = useState<Step>('workload');
  const [form, setForm] = useState<FormState>(() => initialForm(schema));
  const [validation, setValidation] = useState<Validation | null>(null);
  const [extraction, setExtraction] = useState<Extraction | null>(null); const [conflicts, setConflicts] = useState<string[]>([]);
  const [uploadBusy, setUploadBusy] = useState<string | null>(null); const [uploadError, setUploadError] = useState('');
  const [mode, setMode] = useState<'prediction' | 'benchmark'>('prediction');
  const [prediction, setPrediction] = useState<Async<Prediction>>({ status: 'idle' });
  const [compare, setCompare] = useState<Async<{ rows: CompareRow[]; note: string }>>({ status: 'idle' });
  const [plan, setPlan] = useState<Async<BenchmarkPlan>>({ status: 'idle' });
  const [benchmarks, setBenchmarks] = useState<Async<BenchmarkResult[]>>({ status: 'idle', data: [] });
  const [scen, setScen] = useState<Record<string, ScenarioState>>({});
  const values = useMemo(() => toValues(form), [form]);
  const body = useCallback((): ConfigBody => ({ values: toValues(form), provenance: toProvenance(form), metadata: { hardware_selection: form['hardware.selection_id']?.value ?? null } }), [form]);
  const labels = useMemo(() => Object.fromEntries(schema.fields.map((f) => [f.key, f.label])), [schema]);
  const section = useMemo(() => Object.fromEntries(schema.fields.map((f) => [f.key, f.section])), [schema]);

  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    // Create one analysis per mount/reset; one created after unmount (e.g. StrictMode re-run) is deleted at once.
    let cancelled = false;
    api.createAnalysis().then((r) => { if (cancelled) void api.resetAnalysis(r.analysis_id).catch(() => undefined); else { setAnalysisId(r.analysis_id); setSessionError(''); } })
      .catch((e) => { if (!cancelled) setSessionError(message(e)); });
    return () => { cancelled = true; };
  }, [nonce]);
  useEffect(() => {
    if (!analysisId) return;
    const close = () => { api.closeBeacon(analysisId); };
    window.addEventListener('pagehide', close);
    return () => window.removeEventListener('pagehide', close);
  }, [analysisId]);

  // Validate on every edit/upload (debounced); backend-derived defaults fill only empty/default fields.
  const seq = useRef(0);
  useEffect(() => {
    if (!analysisId) return;
    const id = ++seq.current;
    const t = setTimeout(() => {
      api.validate(analysisId, { values: toValues(form), provenance: toProvenance(form) }).then((v) => {
        if (id !== seq.current) return;
        setValidation(v);
        setForm((f) => applySuggestions(f, v.suggestions));
      }).catch((e) => { if (id === seq.current) setSessionError(message(e)); });
    }, 250);
    return () => clearTimeout(t);
  }, [form, analysisId]);

  async function reset(): Promise<void> {
    if (analysisId) { try { await api.resetAnalysis(analysisId); } catch { /* expiry sweeping is the backstop */ } }
    setForm(initialForm(schema)); setValidation(null); setExtraction(null); setConflicts([]); setPrediction({ status: 'idle' }); setCompare({ status: 'idle' });
    setPlan({ status: 'idle' }); setBenchmarks({ status: 'idle', data: [] }); setScen({}); setStep('workload'); setMode('prediction'); setAnalysisId('');
    setNonce((n) => n + 1);
  }

  function applyExtraction(x: Extraction): void {
    setExtraction(x);
    setForm((f) => { const r = applyFindings(f, x.fields); setConflicts((c) => [...new Set([...c.filter((k) => r.form[k]?.pending), ...r.conflicts])]); return r.form; });
  }
  async function onUpload(kind: 'model' | 'training' | 'dataset', files: FileList): Promise<void> {
    setUploadError('');
    for (const file of Array.from(files)) {
      setUploadBusy(`Uploading and statically analysing ${file.name}…`);
      try { applyExtraction((await api.uploadFile(analysisId, kind, file)).extraction); } catch (e) { setUploadError(`${file.name}: ${message(e)}`); }
    }
    setUploadBusy(null);
  }
  async function onRemove(file: UploadedFile): Promise<void> {
    try { const r = await api.removeFile(analysisId, file.file_id); setExtraction(r.extraction); } catch (e) { setUploadError(message(e)); }
  }
  async function onHfImport(modelId: string): Promise<void> {
    setUploadBusy(`Fetching ${modelId}/config.json…`);
    try { applyExtraction((await api.hfImport(analysisId, modelId)).extraction); } finally { setUploadBusy(null); }
  }

  async function selectHardware(o: HardwareOption): Promise<void> {
    try {
      const r = await api.hardwareOption(o.id);
      setForm((f) => {
        let next = { ...f };
        for (const k of CATALOGUE_KEYS) if (next[k]?.source === 'catalogue' && !(k in r.values)) next[k] = { value: null, source: 'catalogue', status: 'ok', detail: `not provided by ${o.label}`, reviewed: false };
        next = setManual(next, 'hardware.source', r.values['hardware.source'] as Json); next = setManual(next, 'hardware.selection_id', o.id);
        const applied = applyCatalogue(next, Object.fromEntries(Object.entries(r.values).filter(([k]) => !['hardware.source', 'hardware.selection_id'].includes(k))), o.label);
        if (applied.conflicts.length) setConflicts((c) => [...new Set([...c, ...applied.conflicts])]);
        return applied.form;
      });
    } catch (e) { setSessionError(message(e)); }
  }
  function customHardware(): void {
    setForm((f) => { let next = { ...f }; for (const k of CATALOGUE_KEYS) if (next[k]?.source === 'catalogue') next[k] = { value: null, source: 'manual', status: 'ok', reviewed: true };
      next = setManual(next, 'hardware.source', 'custom'); return setManual(next, 'hardware.selection_id', null); });
  }

  function go(next: Step): void { setStep(next); window.scrollTo({ top: 0, behavior: 'smooth' }); }
  function goField(key: string): void {
    setStep(section[key] === 'hardware' ? 'hardware' : 'workload');
    setTimeout(() => { const el = document.getElementById(`field-${key}`); el?.scrollIntoView({ behavior: 'smooth', block: 'center' }); (el as HTMLElement | null)?.focus(); }, 80);
  }

  async function runPrediction(): Promise<void> {
    go('results'); setMode('prediction'); setPrediction({ status: 'loading' }); setCompare({ status: 'idle' }); setPlan({ status: 'idle' }); setScen({});
    try { setPrediction({ status: 'ok', data: await api.predict(analysisId, body()) }); document.getElementById('results')?.scrollIntoView({ behavior: 'smooth' }); }
    catch (e) { setPrediction({ status: 'error', error: message(e) }); }
  }
  async function runCompare(): Promise<void> {
    setCompare({ status: 'loading' });
    try { setCompare({ status: 'ok', data: await api.compare(analysisId, body()) }); } catch (e) { setCompare({ status: 'error', error: message(e) }); }
  }
  async function preparePlan(): Promise<void> {
    setPlan({ status: 'loading' });
    try { setPlan({ status: 'ok', data: await api.benchmarkPlan(analysisId, body()) }); } catch (e) { setPlan({ status: 'error', error: message(e) }); }
  }
  async function downloadScript(): Promise<void> {
    try { const blob = await api.benchmarkScript(analysisId, body()); const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = 'run_benchmark.py'; a.click(); URL.revokeObjectURL(url); }
    catch (e) { setPlan((p) => ({ ...p, status: 'error', error: message(e) })); }
  }
  async function importBenchmark(file: File): Promise<void> {
    try { const r = await api.importBenchmark(analysisId, file); setBenchmarks((b) => ({ status: 'ok', data: [r, ...(b.data ?? [])] })); }
    catch (e) { setBenchmarks((b) => ({ ...b, status: 'error', error: message(e) })); }
  }
  async function recalc(sc: Scenario): Promise<void> {
    const changed = sc.change(values); if (!changed) return;
    setScen((p) => ({ ...p, [sc.id]: { status: 'loading' } }));
    try { const result = await api.predict(analysisId, { values: changed, provenance: toProvenance(form) }, false); setScen((p) => ({ ...p, [sc.id]: { status: 'done', result } })); }
    catch (e) { setScen((p) => ({ ...p, [sc.id]: { status: 'error', error: message(e) } })); }
  }
  function applyScenario(sc: Scenario): void {
    const changed = sc.change(values); if (!changed) return;
    setForm((f) => { let next = f; for (const [k, v] of Object.entries(changed)) if (JSON.stringify(v) !== JSON.stringify(values[k])) next = setManual(next, k, v); return next; });
    go('hardware');
  }

  const ctx: FormCtx = { schema, form, values, validation,
    setField: (k, v) => setForm((f) => setManual(f, k, v)), accept: (k) => setForm((f) => acceptPending(f, k)),
    dismiss: (k) => setForm((f) => dismissPending(f, k)), review: (k) => setForm((f) => markReviewed(f, [k])) };
  const visibleKeys = schema.fields.filter((f) => visible(f.key, values, f.profiles)).map((f) => f.key);
  const advanced = new Set(schema.fields.filter((f) => f.advanced).map((f) => f.key));
  // Collapsed advanced defaults are disclosed in "How we calculated this" rather than in the banner.
  const reviewKeys = needsReview(form, visibleKeys.filter((k) => section[k] !== 'hardware' && !(advanced.has(k) && form[k]?.source === 'default')));
  const hwReview = needsReview(form, visibleKeys.filter((k) => section[k] === 'hardware' && !(advanced.has(k) && form[k]?.source === 'default')));
  const issueLabel = (f: string, m: string) => `${labels[f] ?? f}: ${m}`;
  const blockersFor = (inSection: (k: string) => boolean): string[] => validation ? [
    ...validation.unsupported.filter((u) => inSection(u.field)).map((u) => issueLabel(u.field, u.message)),
    ...validation.errors.filter((e) => inSection(e.field)).map((e) => issueLabel(e.field, e.message)),
    ...validation.missing.filter((m) => m.required_for.includes('memory') && inSection(m.field)).map((m) => `${m.label} is required`)] : ['Validating…'];
  const workloadBlockers = blockersFor((k) => section[k] !== 'hardware');
  const allBlockers = blockersFor(() => true);
  const reached = (s: Step) => s === 'workload' || s === 'hardware' || prediction.status !== 'idle';

  return <main className="mx-auto max-w-[1180px] px-5 py-10 sm:px-8 lg:px-10">
    <div className="flex flex-wrap items-start justify-between gap-4"><div><h1 className="text-3xl font-semibold tracking-tight">Plan your training run</h1><p className="mt-2 text-sm text-[#a7a0bb]">PyTorch only · metadata-based predictions · uploads are parsed statically, never executed.</p></div>
      <button type="button" onClick={() => void reset()} className="flex items-center gap-2 border border-line px-3 py-2 text-xs"><RotateCcw size={14} /> New analysis (clears uploads)</button></div>
    {sessionError && <div className="mt-4"><Alert kind="error">{sessionError}</Alert></div>}
    <nav aria-label="Steps" className="my-8 flex flex-wrap gap-2">{stepLabels.map(([id, label], i) => <button key={id} disabled={!reached(id)} aria-current={step === id ? 'step' : undefined} onClick={() => go(id)} className={`border px-4 py-2 text-xs ${step === id ? 'border-violet bg-violet/15 text-white' : 'border-line text-[#a7a0bb] disabled:opacity-40'}`}>{i + 1}. {label}</button>)}</nav>
    {!analysisId ? <p className="text-sm text-[#a7a0bb]">Starting analysis…</p> : <>
      {step === 'workload' && <WorkloadPage ctx={ctx} reviewKeys={reviewKeys} onReviewAll={() => setForm((f) => markReviewed(f, reviewKeys))} blockers={workloadBlockers} onNext={() => go('hardware')}
        handlers={{ busy: uploadBusy, error: uploadError, extraction, conflicts, onUpload: (k, f) => void onUpload(k, f), onRemove: (f) => void onRemove(f), onHfImport }} />}
      {step === 'hardware' && <>{hwReview.length > 0 && <div className="mb-4"><Alert kind="warn">Review {hwReview.length} autofilled hardware value(s): {hwReview.map((k) => labels[k]).join(', ')}. <button type="button" className="ml-2 underline" onClick={() => setForm((f) => markReviewed(f, hwReview))}>I reviewed these</button></Alert></div>}
        <HardwarePage ctx={ctx} onSelect={selectHardware} onCustom={customHardware} onBack={() => go('workload')} onRun={() => void runPrediction()} busy={prediction.status === 'loading'} blockers={allBlockers} /></>}
      {step === 'results' && <ResultsPage mode={mode} setMode={setMode} prediction={prediction} analysisId={analysisId} labels={labels} compare={compare} onCompare={() => void runCompare()} onGoField={goField}
        onEdit={() => go('workload')} onRecs={() => go('recs')} plan={plan} onPlan={() => void preparePlan()} onDownload={() => void downloadScript()} onImport={(f) => void importBenchmark(f)} benchmarks={benchmarks} />}
      {step === 'recs' && <RecommendationsPage values={values} baseline={prediction.data ?? null} states={scen} onRun={(s) => void recalc(s)} onApply={applyScenario} onBack={() => go('results')} />}
    </>}
  </main>;
}

export default function App(): JSX.Element {
  const [screen, setScreen] = useState<'landing' | 'planner'>('landing');
  const [schema, setSchema] = useState<Schema | null>(null); const [error, setError] = useState('');
  useEffect(() => { if (screen === 'planner' && !schema) api.schema().then(setSchema).catch((e) => setError(message(e))); }, [screen, schema]);
  if (screen === 'landing') return <LandingPage onStart={() => setScreen('planner')} />;
  return <div className="min-h-screen bg-ink text-white"><TopBar onLogo={() => setScreen('landing')} onBack={() => setScreen('landing')} />
    {schema ? <Planner schema={schema} /> : <main className="mx-auto max-w-[1180px] px-5 py-10">{error ? <Alert kind="error">{error}</Alert> : <p className="text-sm text-[#a7a0bb]">Loading input schema…</p>}</main>}</div>;
}

function TopBar({ onLogo, onBack }: { onLogo: () => void; onBack: () => void }): JSX.Element { return <header className="border-b border-white/10 bg-ink/80 backdrop-blur-xl"><div className="mx-auto flex max-w-[1180px] items-center justify-between px-5 py-5 sm:px-8 lg:px-10"><button onClick={onLogo} className="flex items-center gap-3" aria-label="Return to planner start"><span className="grid h-8 w-8 place-items-center bg-gradient-to-br from-violet to-cyan text-xs font-extrabold text-black">CC</span><span className="text-sm font-bold tracking-[0.14em]">COMPUTE <span className="text-violet">CAST</span></span></button><nav className="hidden items-center gap-8 text-xs text-[#a7a0bb] md:flex"><button onClick={onBack} className="transition hover:text-white">Overview</button><span>Estimator</span></nav><button className="text-[#a7a0bb] md:hidden" aria-label="Open navigation"><Menu size={20} /></button></div></header>; }
function Landing({ onStart }: { onStart: () => void }): JSX.Element { return <section className="relative min-h-screen overflow-hidden bg-black"><div className="absolute inset-0" style={{ backgroundImage: 'radial-gradient(ellipse 42% 54% at 2% 30%, rgba(199,43,177,.82) 0%, rgba(80,19,105,.5) 38%, transparent 72%), radial-gradient(ellipse 64% 48% at 55% 84%, rgba(24,28,206,.82) 0%, rgba(22,7,123,.72) 42%, transparent 76%), radial-gradient(ellipse 50% 32% at 80% 108%, rgba(47,220,214,.78) 0%, rgba(8,79,148,.56) 34%, transparent 70%), radial-gradient(ellipse 35% 52% at 104% 48%, rgba(20,25,194,.9) 0%, rgba(10,5,72,.62) 58%, transparent 100%), linear-gradient(145deg,#09000e 0%,#00000a 43%,#02001d 70%,#02000b 100%)' }} /><div className="absolute -right-24 top-12 h-[42vh] w-[52vw] rounded-[45%] bg-black/75 blur-2xl" /><div className="relative mx-auto flex min-h-screen max-w-[1280px] flex-col px-6 pb-10 pt-8 sm:px-10 lg:px-16"><LandingNav onStart={onStart} /><div className="flex flex-1 items-center pb-20 pt-28 sm:pt-36"><div className="max-w-4xl"><h1 className="text-[clamp(3.5rem,10vw,8.8rem)] font-extrabold leading-[.88] tracking-[-.075em] text-[#eeeeee]">COMPUTE <span className="text-[#8e7cea]">CAST</span></h1><p className="mt-9 max-w-[610px] text-lg font-light leading-[1.28] text-[#e0dce8] sm:text-2xl">estimate GPU resources and recommend optimizations for model training</p><p className="mt-5 max-w-[520px] text-sm leading-6 text-[#bdb7cc]">Know whether your model fits <strong className="font-semibold text-white">before you rent the GPUs</strong>. Get VRAM, runtime and cost estimates, then the cheapest fixes.</p><div className="mt-10 flex flex-col items-start gap-4 sm:flex-row sm:items-center"><button onClick={onStart} className="group flex items-center gap-3 bg-white px-6 py-4 text-sm font-semibold text-black transition hover:bg-violet hover:text-white">Open estimator <ArrowRight size={17} className="transition group-hover:translate-x-1" /></button></div></div></div></div></section>; }
function LandingNav({ onStart }: { onStart: () => void }): JSX.Element { return <nav className="flex items-center justify-between"><div className="flex items-center gap-3"><span className="grid h-8 w-8 place-items-center border border-white/30 text-[10px] font-bold">CC</span><span className="text-xs font-bold tracking-[.18em]">COMPUTE CAST</span></div><button onClick={onStart} className="border border-white/30 px-4 py-2 text-xs text-white transition hover:border-white hover:bg-white hover:text-black">Launch tool</button></nav>; }
function HowItWorks({ onStart }: { onStart: () => void }): JSX.Element {
  const items = [{ n: '01', title: 'Describe your workload', body: 'Upload PyTorch model/training scripts or configs, search Hugging Face, or type the settings. Files are parsed statically and never executed.', tag: 'model.py · config.json · train.yaml' },
    { n: '02', title: 'Choose your hardware', body: 'Search Nebius GPUs with VRAM and price per GPU-hour, or enter your own machine.', tag: 'GPU · VRAM · $/GPU-hr' },
    { n: '03', title: 'See the fit, then optimize', body: 'Get peak VRAM against your usable limit, runtime scenarios and cost, then benchmark locally to confirm.', tag: 'Fits · Tight · Exceeds' }];
  return <main className="bg-black px-6 pb-24 pt-20 sm:px-10 lg:px-16"><div className="mx-auto max-w-[1180px]"><p className="eyebrow text-[#8e7cea]">HOW IT WORKS</p><h2 className="mt-3 max-w-2xl text-3xl font-semibold tracking-tight sm:text-4xl">Three steps from model to <span className="text-[#8e7cea]">compute plan</span></h2>
    <div className="mt-10 grid gap-4 md:grid-cols-3">{items.map((i) => <article key={i.n} className="border border-white/15 bg-white/[0.03] p-6"><span className="text-xs font-bold text-[#8e7cea]">{i.n}</span><h3 className="mt-4 text-lg font-semibold">{i.title}</h3><p className="mt-3 text-sm leading-6 text-[#bdb7cc]">{i.body}</p><p className="mt-5 inline-block border border-white/20 px-2 py-1 font-mono text-[11px] text-[#a7a0bb]">{i.tag}</p></article>)}</div>
    <div className="mt-10 border border-white/15 p-6" aria-label="Example result, illustrative"><p className="eyebrow">EXAMPLE RESULT (ILLUSTRATIVE)</p><div className="mt-4 h-4 border border-white/20"><div className="h-full w-[62%] bg-[#7258dd]" /></div><div className="mt-2 flex justify-between text-xs text-[#a7a0bb]"><span>Peak VRAM</span><span>Usable limit</span></div></div>
    <button onClick={onStart} className="mt-10 bg-white px-6 py-4 text-sm font-semibold text-black transition hover:bg-violet hover:text-white">Open estimator</button></div></main>;
}
function LandingPage({ onStart }: { onStart: () => void }): JSX.Element { return <><Landing onStart={onStart} /><HowItWorks onStart={onStart} /></>; }
