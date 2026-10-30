// GPT-2 MVP E2E (flow 1): example files -> autofill -> Nebius H100 -> estimate -> comparison ->
// benchmark script/import -> edit + re-run (original vs current) -> cache cleanup on page close.
// Requires the backend (uvicorn) and the frontend dev server (npm run dev) running.
// Env: E2E_BASE_URL (default http://127.0.0.1:5173), E2E_CACHE_DIR (must equal the backend's
// UPLOAD_CACHE_DIR; default <tmp>/compute_cast_uploads), BENCH_PYTHON (a Python with torch; when
// unset the benchmark run/import step is skipped), E2E_OUT (screenshots; default a temp folder).
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
  await page.getByText('A · Hugging Face model').waitFor();
  const exampleReq = page.waitForRequest((r) => r.url().includes('/examples/'));
  await page.getByRole('button', { name: 'GPT-2 small, supported settings' }).click();
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

  await page.getByLabel('Search GPUs and Nebius platforms').fill('h100');
  const results = page.getByLabel('Hardware search results').locator('li');
  await page.getByLabel('Hardware search results').getByText(/H100/).first().waitFor();
  expect(await results.count() <= 10, 'at most 10 hardware results');
  expect(/80 GiB/.test(await results.first().innerText()) && /GPU-hour/.test(await results.first().innerText()), 'result shows VRAM and price unit');
  await results.first().locator('button').click();
  await page.waitForTimeout(900);
  expect(await f('hardware.vram_gib').inputValue() === '80', 'VRAM from catalogue');
  expect(await f('hardware.usable_vram_gib').inputValue() === '80', 'usable limit defaults to 100% of VRAM');
  expect(await f('hardware.bf16_dense_tflops').isDisabled(), 'catalogue specs locked unless Custom GPU');
  expect(await f('hardware.price_unit').inputValue() === 'gpu_hour', 'price unit autofilled');
  await page.screenshot({ path: path.join(shots, '02-hardware.png'), fullPage: true });
  await page.getByRole('button', { name: 'Run estimate' }).click();

  await page.getByText(/usable VRAM limit per GPU \(/).waitFor({ timeout: 20000 });
  const res = await page.locator('#results').innerText();
  expect(/Estimated fit/i.test(res), 'fit verdict shown');
  expect(/80\.00 GiB of 80\.00 GiB total/.test(res), 'usable/total denominator shown');
  expect(/124,439,808 parameters/.test(res), 'GPT-2 parameter count');
  expect(/optimizer updates ×/.test(res), 'runtime breakdown shown');
  expect(/\$0\.\d\d|< \$0\.01/.test(res), 'job cost shown');
  expect(/Benchmark run cost/i.test(res), 'benchmark cost shown');
  expect(/This is the original run/.test(res), 'original snapshot');
  await page.getByRole('button', { name: 'Compare Nebius GPUs' }).click();
  await page.getByText(/Same GPT-2 workload/).waitFor({ timeout: 30000 });
  await page.getByText('How we calculated this').click();
  expect(/16·P/.test(await page.locator('#results').innerText()), 'formulas shown');
  await page.screenshot({ path: path.join(shots, '03-results.png'), fullPage: true });

  await page.getByRole('tab', { name: 'Benchmark' }).click();
  await page.getByRole('button', { name: 'Prepare benchmark' }).click();
  await page.getByText('Run on Nebius').waitFor();
  expect((await page.locator('code').first().innerText()).startsWith('python run_benchmark.py'), 'CLI command shown');
  expect(await page.getByRole('button', { name: /Start paid Nebius benchmark/ }).isDisabled(), 'Nebius run unavailable, not simulated');
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Download run_benchmark.py' }).click()]);
  const script = path.join(dir, 'run_benchmark.py'); await download.saveAs(script);
  expect(fs.readFileSync(script, 'utf8').includes('"architecture": "gpt2"'), 'script embeds GPT-2 config');
  if (process.env.BENCH_PYTHON) {
    const out = path.join(dir, 'benchmark_result.json');
    execFileSync(process.env.BENCH_PYTHON, [script, '--device', 'cpu', '--steps', '1', '--warmup', '0', '--output', out], { cwd: dir, stdio: 'inherit' });
    await page.locator('input[type=file][accept*=json]').last().setInputFiles(out);
    await page.getByText('Measured · benchmark').waitFor({ timeout: 20000 });
    expect(/matches current config/i.test(await page.locator('#results').innerText()), 'imported benchmark matches config');
  } else log('skipped - benchmark run/import (set BENCH_PYTHON)');

  await page.getByRole('tab', { name: 'Estimate' }).click();
  await page.getByRole('button', { name: 'Edit inputs' }).click();
  await f('training.microbatch_size').fill('16');
  await page.waitForTimeout(700);
  await page.getByRole('button', { name: 'Continue to hardware' }).click();
  await page.getByRole('button', { name: 'Run estimate' }).click();
  await page.getByText(/Original microbatch size per GPU/).waitFor({ timeout: 20000 });
  expect(/8.*current: 16/.test(await page.getByText(/Original microbatch size per GPU/).innerText()), 'change vs original shown');

  const cacheDir = path.join(CACHE, analysisId);
  expect(fs.existsSync(cacheDir), 'analysis cache folder exists (check E2E_CACHE_DIR)');
  await page.close({ runBeforeUnload: true });
  await new Promise((r) => setTimeout(r, 1500));
  expect(!fs.existsSync(cacheDir), 'cache removed after page close');
  await browser.close();
  log('screenshots in', shots);
  log('FLOW 1 PASSED');
})().catch((e) => { console.error(e); process.exit(1); });
