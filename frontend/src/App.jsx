import React, { useState } from 'react';
import { requestEstimate } from './api.js';

const fields = [
  ['examples', 'Training examples', 100000, 1, 1],
  ['sequence_length', 'Processed sequence length (tokens)', 512, 1, 1],
  ['batch_size', 'Batch size', 16, 1, 1],
  ['epochs', 'Epochs', 10, 1, 1],
  ['seconds_per_batch', 'Measured seconds per training batch', 0.2, 0.001, 'any'],
  ['hourly_price_usd', 'Instance price (USD/hour)', 1, 0, 'any'],
  ['target_vram_gb', 'Target GPU memory (GB)', 24, 0.001, 'any'],
  ['measured_peak_vram_gb', 'Measured peak GPU memory (GB, optional)', '', 0.001, 'any'],
];
export default function App() {
  const [form, setForm] = useState(Object.fromEntries(fields.map(([key,,value]) => [key, value])));
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  async function submit(event) {
    event.preventDefault(); setBusy(true); setError(''); setResult(null);
    try {
      const payload = Object.fromEntries(Object.entries(form).map(([key,value]) => [key, value === '' ? null : Number(value)]));
      setResult(await requestEstimate(payload));
    } catch (e) { setError(e.message || 'Cannot reach API. Start the Python backend.'); }
    finally { setBusy(false); }
  }
  return <main>
    <header><span className="tag">HACKATHON STARTER</span><h1>GPU resource planner</h1><p>Turn a short training benchmark into a runtime and cost estimate.</p></header>
    <p className="notice">Defaults are illustrative. Enter benchmark measurements for your actual workload and target GPU before making spending decisions.</p>
    <form onSubmit={submit}><div className="grid">{fields.map(([key,label,,min,step]) => <label key={key}>{label}<input type="number" min={min} step={step} required={key !== 'measured_peak_vram_gb'} value={form[key]} onChange={e => { setForm({...form, [key]:e.target.value}); setResult(null); }}/></label>)}</div><button disabled={busy}>{busy ? 'Calculating…' : 'Estimate resources'}</button></form>
    {error && <p role="alert" className="error">{error}</p>}
    {result && <section aria-live="polite"><h2>Your estimate</h2><div className="metrics"><article><strong>{result.training_steps.toLocaleString()}</strong>Training batches</article><article><strong>{result.training_hours} h</strong>Training / GPU-hours (1 GPU)</article><article><strong>${result.training_cost_usd.toFixed(2)}</strong>Training compute · USD</article></div><p>Memory: {({'unknown':'No measurement supplied','exceeds_capacity':'Measured peak reaches or exceeds capacity','below_capacity_check_headroom':'Measured peak is below capacity; check headroom'})[result.memory_status]}</p><h3>Assumptions</h3><ul>{result.assumptions.map(item => <li key={item}>{item}</li>)}</ul></section>}
  </main>;
}
