// GPT-2 MVP E2E (flow 1): example files -> autofill -> Nebius H100 -> estimate -> comparison ->
// benchmark config review/package/import (+ simulated Nebius job) -> edit + re-run (original vs current) -> cache cleanup on page close.
// Requires the backend (uvicorn) and the frontend dev server (npm run dev) running.
// Env: E2E_BASE_URL (default http://127.0.0.1:5173), E2E_CACHE_DIR (must equal the backend's
// UPLOAD_CACHE_DIR; default <tmp>/compute_cast_uploads), BENCH_PYTHON (a Python with torch; when
// unset the local package run/import step is skipped), E2E_BENCH_PASSWORD (the backend's
// BENCH_DEMO_PASSWORD when it runs with BENCH_CLOUD_ENABLED=1 BENCH_NEBIUS_BACKEND=fake; when unset
// the cloud tab is only checked to be unavailable), E2E_OUT (screenshots; default a temp folder).
const { chromium } = require('playwright');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');
const BASE = process.env.E2E_BASE_URL || 'http://127.0.0.1:5173';
const CACHE = process.env.E2E_CACHE_DIR || path.join(os.tmpdir(), 'compute_cast_uploads');
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'cc-e2e-'));
const shots = process.env.E2E_OUT || path.join(dir, 'shots'); fs.mkdirSync(shots, { recursive: true });
const log = (...a) => console.log('[e2e]', ...a);
const expect = (cond, msg) => { if (!cond) throw new Error('ASSERTION FAILED: ' + msg); log('ok -', msg); };

(async () => {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ acceptDownloads: true, viewport: { width: 1280, height: 900 } })).newPage();
  page.on('pageerror', (e) => log('PAGE ERROR', e.message));
  const f = (k) => page.locator('#field-' + k.replace(/\./g, '\\.'));
  await page.goto(BASE + '/');
  await page.getByRole('button', { name: 'Open estimator' }).first().click();
  await page.getByText('Hugging Face model', { exact: true }).waitFor();
  const tabs = await page.getByRole('navigation', { name: 'Steps' }).innerText();
  expect(!/\d\./.test(tabs) && /Workload/.test(tabs) && /Recommendations/.test(tabs), 'step tabs have no numbers');
  expect(await page.getByText(/upload a model config and\/or script/).isVisible() && await page.getByText('(only PyTorch is supported)').isVisible(), 'OR upload label under the HF box');
  await page.getByRole('button', { name: 'Preview GPT-2 small, supported settings' }).click();
  await page.getByRole('dialog').getByText(/n_layer/).waitFor();
  await page.getByRole('dialog').getByRole('tab', { name: 'train.py' }).click();
  await page.getByRole('dialog').getByText(/from_pretrained/).first().waitFor();
  await page.screenshot({ path: path.join(shots, '00-example-preview.png') });
  log('ok - example files preview in a dialog');
  await page.getByRole('button', { name: 'Close preview' }).click();
  const exampleReq = page.waitForRequest((r) => r.url().includes('/examples/'));
  await page.getByRole('button', { name: 'GPT-2 small, supported settings', exact: true }).click();
  const analysisId = (await exampleReq).url().split('/api/analysis/')[1].split('/')[0];
  await page.getByLabel('Uploaded files').waitFor();
  await page.waitForTimeout(900);
  expect(await f('model.num_layers').inputValue() === '12', 'model section filled from config.json');
  expect(await f('model.num_layers').isDisabled(), 'checkpoint dimensions locked (custom_only)');
  expect(await f('training.microbatch_size').inputValue() === '8', 'training section filled from training_args.json');
  expect(await f('data.sequence_length').inputValue() === '512', 'data section filled from data_config.json');
  expect(await f('data.num_training_examples').inputValue() === '100000', 'training example count filled');
  const chips = await page.getByLabel('Uploaded files').innerText();
  expect(/MODEL/i.test(chips) && /TRAINING/i.test(chips) && /DATA/i.test(chips), 'files grouped into sections');
  expect((await page.locator('form').innerText()).includes('Execution assumptions'), 'execution assumptions panel present');
  await page.getByText('Execution assumptions').first().click();
  const fixedSelect = page.getByLabel('precision mode');
  expect(await fixedSelect.locator('option').count() === 1 && await fixedSelect.inputValue() === 'bf16_autocast', 'fixed assumption is a one-option dropdown');
  await page.screenshot({ path: path.join(shots, '01-workload-example.png'), fullPage: true });
  const review = page.getByRole('button', { name: 'Mark all reviewed' }); if (await review.count()) await review.click();
  await page.getByRole('button', { name: 'Continue to hardware' }).click();

  expect(await page.getByRole('combobox', { name: /Search/ }).count() === 0, 'no search bar, dropdown only');
  expect(await page.getByRole('radio', { name: 'Nebius AI Cloud' }).getAttribute('aria-checked') === 'true', 'Nebius AI Cloud selected by default');
  expect(await f('hardware.price_per_hour').inputValue() === '0', 'price defaults to 0');
  expect(await page.getByText('GPU name', { exact: true }).count() > 0, 'field labelled GPU name');
  const gpuSelect = page.getByLabel('Nebius AI Cloud GPU');
  const nebiusOptions = await gpuSelect.locator('option').allInnerTexts();
  expect(nebiusOptions.some((o) => /H100.*eu-north1.*80 GiB.*GPU-hour/.test(o)) && !nebiusOptions.some((o) => /B300|RTX/.test(o)), 'Nebius dropdown lists runtime-complete GPUs with region and price');
  await page.getByRole('radio', { name: 'GPU catalogue' }).click();
  const catalogOptions = await page.getByLabel('Catalogue GPU').locator('option').allInnerTexts();
  expect(catalogOptions.some((o) => /A100/.test(o)) && !catalogOptions.some((o) => /H100/.test(o)), 'catalogue lists only GPUs not already on Nebius');
  await page.getByRole('radio', { name: 'Nebius AI Cloud' }).click();
  await page.getByLabel('Nebius AI Cloud GPU').selectOption({ label: nebiusOptions.find((o) => /H100.*eu-north1/.test(o)) });
  await page.waitForTimeout(900);
  expect(await f('hardware.vram_gib').inputValue() === '80', 'VRAM from catalogue');
  expect(await f('hardware.usable_vram_gib').inputValue() === '72', 'usable limit defaults to 90% of VRAM');
  expect(!(await f('hardware.vram_gib').isDisabled()) && !(await f('hardware.usable_vram_gib').isDisabled()), 'total and usable VRAM both editable');
  await f('hardware.vram_gib').fill('60');
  await page.waitForTimeout(900);
  expect(await f('hardware.usable_vram_gib').inputValue() === '54', 'usable limit follows total VRAM until edited');
  await f('hardware.vram_gib').fill('80');
  await page.waitForTimeout(900);
  expect(!(await f('hardware.bf16_dense_tflops').isDisabled()), 'catalogue specs stay editable without choosing Custom GPU');
  const tflops = await f('hardware.bf16_dense_tflops').inputValue();
  await f('hardware.bf16_dense_tflops').fill('500');
  await page.waitForTimeout(400);
  await page.getByRole('button', { name: /restore .* specs/ }).click();
  await page.waitForTimeout(900);
  expect(await f('hardware.bf16_dense_tflops').inputValue() === tflops, 'restoring the GPU replaces edited specs (no frozen fields)');
  expect(await f('hardware.price_unit').inputValue() === 'gpu_hour', 'price unit autofilled');
  await page.screenshot({ path: path.join(shots, '02-hardware.png'), fullPage: true });
  await page.getByRole('button', { name: 'Run estimate' }).click();

  await page.getByText(/usable VRAM limit per GPU \(/).waitFor({ timeout: 20000 });
  const res = await page.locator('#results').innerText();
  expect(/Estimated fit/i.test(res), 'fit verdict shown');
  expect(/72\.00 GiB of 80\.00 GiB total/.test(res), 'usable/total denominator shown');
  expect(/124,439,808 parameters/.test(res), 'GPT-2 parameter count');
  expect(/per optimizer update/.test(res) && !/optimizer updates ×/.test(res), 'runtime shown without the equation');
  expect(/\$0\.\d\d|< \$0\.01/.test(res), 'job cost shown');
  expect(/Benchmark run cost/i.test(res), 'benchmark cost shown');
  expect(/This is the original run/.test(res), 'original snapshot');
  await page.getByRole('button', { name: 'Compare Nebius GPUs' }).click();
  await page.getByText(/Same GPT-2 workload/).waitFor({ timeout: 30000 });
  await page.getByText('How we calculated this').click();
  const how = await page.locator('#results').innerText();
  expect(/16·P/.test(how) && /Your numbers/i.test(how) && /optimizer updates ×/.test(how), 'equations with your numbers under How we calculated this');
  await page.screenshot({ path: path.join(shots, '03-results.png'), fullPage: true });

  await page.getByRole('radio', { name: /Benchmark mode/ }).click();
  expect(await page.getByRole('radio', { name: /Nebius AI Cloud/ }).getAttribute('aria-checked') === 'true', 'Nebius AI Cloud is the default benchmark target');
  await page.getByRole('radio', { name: /Your GPU/ }).click();
  await page.getByText('Review benchmark setup', { exact: true }).waitFor({ timeout: 20000 });
  expect(await page.getByRole('button', { name: /Show setup/ }).getAttribute('aria-expanded') === 'false', 'benchmark setup collapsed by default');
  await page.getByRole('button', { name: /Show setup/ }).click();
  let bench = await page.locator('#results').innerText();
  expect(/synthetic data/i.test(bench), 'synthetic-input scope explained');
  expect(/detected/i.test(bench) && /assumption/i.test(bench) && /default/i.test(bench), 'provenance shown (detected/default/assumption)');
  expect(/Memory check/i.test(bench) && /fits/i.test(bench), 'memory check against VRAM with safety margin');
  // Small shapes so the optional CPU smoke run is quick; edits are kept across regeneration.
  for (const [k, v] of [['training.microbatch_size', '1'], ['training.gradient_accumulation_steps', '1'], ['input.sequence_length', '32'], ['measurement.warmup_updates', '0'], ['measurement.measured_updates', '1']]) {
    await page.getByLabel(k, { exact: true }).fill(v);
  }
  await page.waitForTimeout(1200);
  expect(/your choice/i.test(await page.locator('#results').innerText()), 'edited fields marked as your choice');
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Download local benchmark package' }).click()]);
  const zip = path.join(dir, 'compute-cast-benchmark.zip'); await download.saveAs(zip);
  const pkg = path.join(dir, 'pkg');
  execFileSync('python3', ['-c', 'import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])', zip, pkg]);
  const root = path.join(pkg, 'compute-cast-benchmark');
  const cfg = JSON.parse(fs.readFileSync(path.join(root, 'benchmark_config.json'), 'utf8'));
  expect(cfg.training.microbatch_size === 1 && cfg.input.source === 'synthetic', 'package config carries the reviewed edits');
  expect(['benchmark.py', 'model_registry.py', 'models/decoder.py', 'requirements.txt', 'README.md'].every((n) => fs.existsSync(path.join(root, n))), 'package has trusted runner files');
  if (process.env.BENCH_PYTHON) {
    execFileSync(process.env.BENCH_PYTHON, ['benchmark.py', '--config', 'benchmark_config.json', '--cpu-smoke'], { cwd: root, stdio: 'inherit' });
    await page.locator('input[type=file][accept*=json]').last().setInputFiles(path.join(root, 'results.json'));
    await page.getByText('Predicted vs measured').waitFor({ timeout: 20000 });
    bench = await page.locator('#results').innerText();
    expect(/CPU smoke test/i.test(bench) && /not used for calibration/i.test(bench), 'imported smoke result labelled, no calibration');
    expect(/matches current config/i.test(bench), 'imported result matches the reviewed config');
  } else log('skipped - local package run/import (set BENCH_PYTHON)');
  await page.getByLabel('Paste results.json').fill('{"schema_version": "1.0", "status": "succeeded"}');
  await page.getByRole('button', { name: 'Validate and compare' }).click();
  await page.getByText(/needs effective_config/).waitFor({ timeout: 10000 });
  log('ok - malformed results rejected');
  await page.screenshot({ path: path.join(shots, '04-benchmark-local.png'), fullPage: true });

  await page.getByRole('radio', { name: /Nebius AI Cloud/ }).click();
  if (process.env.E2E_BENCH_PASSWORD) {
    // Backend started with BENCH_CLOUD_ENABLED=1 BENCH_NEBIUS_BACKEND=fake (simulated provider, no Nebius calls).
    await page.getByText(/simulated provider/i).first().waitFor({ timeout: 10000 });
    await page.getByLabel('Display name').fill('E2E');
    await page.getByLabel('Demo password').fill('wrong-password');
    await page.getByRole('button', { name: 'Sign in' }).click();
    await page.getByText('Incorrect password.').waitFor();
    log('ok - wrong password rejected');
    expect(await page.getByLabel('Demo password').getAttribute('type') === 'password', 'password input is masked');
    await page.getByLabel('Demo password').fill(process.env.E2E_BENCH_PASSWORD);
    await page.getByRole('button', { name: 'Sign in' }).click();
    await page.getByText(/Signed in as/).waitFor();
    expect(await page.getByRole('button', { name: 'Start Nebius benchmark' }).isDisabled(), 'submit needs charge confirmation');
    await page.getByRole('checkbox').check();
    await page.getByRole('button', { name: 'Start Nebius benchmark' }).click();
    await page.getByText(/job [0-9a-f]{20}/).waitFor({ timeout: 10000 });
    await page.getByText(/results missing/i).waitFor({ timeout: 60000 });
    expect(/never retried automatically/.test(await page.locator('#results').innerText()), 'simulated job ends failed without fake results');
    await page.screenshot({ path: path.join(shots, '05-benchmark-nebius-simulated.png'), fullPage: true });
  } else {
    await page.getByText(/Switched off|Not configured/).first().waitFor({ timeout: 10000 });
    log('ok - cloud benchmarks shown unavailable (set E2E_BENCH_PASSWORD with a fake-backend server to test login/submit)');
  }

  await page.getByRole('radio', { name: /Estimate mode/ }).click();
  await page.getByRole('button', { name: 'Edit inputs' }).click();
  await f('training.microbatch_size').fill('16');
  await page.waitForTimeout(700);
  await page.getByRole('button', { name: 'Continue to hardware' }).click();
  await page.getByRole('button', { name: 'Run estimate' }).click();
  await page.getByText(/Original microbatch size per GPU/).waitFor({ timeout: 20000 });
  expect(/8.*current: 16/.test(await page.getByText(/Original microbatch size per GPU/).innerText()), 'change vs original shown');
  expect(/vs original/.test(await page.locator('#results').innerText()), 'results show change markers vs the original estimate');

  // Recommendations. The model call is intercepted with a canned (server-format) answer: no API key, no charge.
  await page.route('**/api/recommendations/status', (r) => r.fulfill({ json: { configured: true, model: 'NVIDIA Nemotron 3 Super (nvidia/nemotron-3-super-120b-a12b) via Nebius Token Factory',
    code_kinds: ['Training script (train.py)'], goals: {}, note: '' } }));
  let adviceBody = null;
  await page.route('**/api/analysis/*/advice', (r) => { adviceBody = JSON.parse(r.request().postData()); r.fulfill({ json: {
    model: 'NVIDIA Nemotron 3 Super', original_summary: 'GPT-2 small (124M) with microbatch 16 x accumulation 4 on one H100.',
    suggestions_summary: 'Halve the microbatch and double accumulation to keep the effective batch; checkpointing is optional.',
    recommendations: [
      { id: 'mb', title: 'Microbatch 8, accumulation 8', category: 'batch_accumulation', explanation: 'Keeps the effective batch at 64 and lowers activation memory.',
        parameter_changes: { 'training.microbatch_size': 8, 'training.gradient_accumulation_steps': 8 }, hardware_option_id: null, hardware: null, expected_effect: 'Less memory.',
        tradeoffs: ['Slightly more launches'], affected_files: [], issues: [], violations: [], estimable: true, applicable: true,
        new_values: { 'training.microbatch_size': 8, 'training.gradient_accumulation_steps': 8 }, estimate_note: null },
      { id: 'ckpt', title: 'Activation checkpointing', category: 'activation_checkpointing', explanation: 'Recompute activations in backward.', parameter_changes: {},
        hardware_option_id: null, hardware: null, expected_effect: 'Much less activation memory.', tradeoffs: ['~30% more compute'], affected_files: [], issues: [], violations: [],
        estimable: false, applicable: false, new_values: {}, estimate_note: 'Not estimated: this tool models single-GPU batch, sequence and hardware changes only.' },
      { id: 'zero', title: 'ZeRO-2 across 4 GPUs with checkpointing', category: 'combined', explanation: 'Shard optimizer state and gradients; recompute activations.', parameter_changes: {},
        hardware_option_id: null, hardware: null, expected_effect: 'About 3.5x shorter run.', tradeoffs: ['Needs 4 GPUs'], affected_files: [], issues: [], violations: [],
        strategy: { data_parallel: 4, tensor_parallel: 1, pipeline_parallel: 1, gpu_count: 4, sharding: 'zero2', activation_checkpointing: 'full', attention: 'math' },
        projection: { status: 'ok', approximate: true, per_gpu_capacity_gib: 5.1, allocated_peak_gib: 3.9, usable_vram_gib: 72, percent_of_usable: 7, fits: true,
          memory_gib: { training_state: 0.9 }, training_hours: 0.04, cost_usd: 0.4, gpu_count: 4, efficiency: 0.92, compute_overhead: 1.33, global_effective_batch: 64,
          optimizer_updates: 100, assumptions: ['Approximate projection.'] },
        implementation_steps: ['Wrap the model with DeepSpeed ZeRO-2'], launch_snippet: 'deepspeed --num_gpus 4 train.py',
        estimable: false, applicable: false, new_values: {}, estimate_note: 'Approximate projection.' }],
    changed_files: [], files_check: null, assumptions: [], missing_information: [], files_reviewed: ['train.py', 'train_baseline.py'], note: 'AI suggestions are unverified.' } }); });
  await page.route('**/api/analysis/*/advice/apply', (r) => r.fulfill({ json: { applied: {}, original_kept: true } }));
  await page.getByRole('button', { name: 'Improve this plan' }).click();
  await page.getByText('Powered by NVIDIA Nemotron 3 Super').waitFor();
  expect(await page.getByText('Reduce memory', { exact: true }).count() === 1 && await page.getByText(/what-ifs/i).count() === 0, 'reduce-memory goal; no quick what-ifs');
  await page.getByText('Lowest cost', { exact: true }).click();
  await page.getByText('Consider other GPUs').click();
  await page.getByRole('button', { name: 'Use example training script' }).click();
  await page.getByRole('list', { name: 'Code for review' }).getByText('train_baseline.py').waitFor();
  await page.getByRole('button', { name: 'Get recommendations' }).click();
  await page.getByText('Your current plan').waitFor({ timeout: 10000 });
  expect(adviceBody.preferences.goal === 'lowest_cost' && adviceBody.preferences.resources === 'alternatives' && 'train_baseline.py' in adviceBody.files
    && adviceBody.values['training.microbatch_size'] === 16, 'request carries goal, resources, extra code and final values');
  const recs = await page.locator('main').innerText();
  expect(/Suggested changes/.test(recs) && /not estimated/i.test(recs), 'summaries and not-estimated label shown');
  expect(/Approximate projection/i.test(recs) && /4 GPUs/i.test(recs) && /ZeRO-2/i.test(recs) && /deepspeed --num_gpus 4/.test(recs), 'strategy plan shows its approximate projection, steps and launch snippet');
  await page.getByRole('button', { name: 'Recalculate estimate' }).first().click();
  await page.getByText('Recalculated vs your original estimate').waitFor({ timeout: 20000 });
  await page.screenshot({ path: path.join(shots, '06-recommendations.png'), fullPage: true });
  await page.getByRole('button', { name: 'Apply changes' }).first().click();
  await page.getByText(/usable VRAM limit per GPU \(/).waitFor({ timeout: 20000 });
  expect(await f('training.microbatch_size').count() === 0, 'apply re-ran the estimate on the results page');
  await page.getByRole('button', { name: 'Edit inputs' }).click();
  expect(await f('training.microbatch_size').inputValue() === '8' && await f('training.gradient_accumulation_steps').inputValue() === '8', 'applied changes written to the form');
  await page.screenshot({ path: path.join(shots, '07-applied.png'), fullPage: true });

  const cacheDir = path.join(CACHE, analysisId);
  expect(fs.existsSync(cacheDir), 'analysis cache folder exists (check E2E_CACHE_DIR)');
  await page.close({ runBeforeUnload: true });
  await new Promise((r) => setTimeout(r, 1500));
  expect(!fs.existsSync(cacheDir), 'cache removed after page close');
  await browser.close();
  log('screenshots in', shots);
  log('FLOW 1 PASSED');
})().catch((e) => { console.error(e); process.exit(1); });
