// E2E: uploaded-file flow (MLP), review/edit, hardware search, prediction, comparison,
// benchmark script download -> run (CPU dry run) -> import, re-run with edits (snapshot diff),
// no-overwrite of manual edits on re-extraction, and cache cleanup on page close.
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
const files = path.join(dir, 'files'); fs.mkdirSync(files, { recursive: true });
fs.writeFileSync(path.join(files, 'model.py'), `import torch.nn as nn
import os
open(${JSON.stringify(path.join(dir, 'EXECUTED_SENTINEL'))}, "w").write("x")
HIDDEN = 64
def build_model(config):
    return nn.Sequential(nn.Linear(config["input_size"], config["hidden_size"]), nn.ReLU(), nn.Linear(config["hidden_size"], config["output_size"]))
model = nn.Sequential(nn.Linear(100, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))
`);
fs.writeFileSync(path.join(files, 'train.py'), `import torch
BATCH = 16
loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, drop_last=False)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()
epochs = 3
gradient_accumulation_steps = int(os.environ["ACC"])
`);
fs.writeFileSync(path.join(files, 'sample.csv'), 'f' + Array.from({ length: 100 }, (_, i) => i).join(',f') + ',target\n' +
  Array.from({ length: 40 }, (_, r) => Array.from({ length: 101 }, (_, i) => ((r * i) % 7) / 7).join(',')).join('\n') + '\n');
fs.rmSync(path.join(dir, 'EXECUTED_SENTINEL'), { force: true });
const log = (...a) => console.log('[e2e]', ...a);
const expect = (cond, msg) => { if (!cond) throw new Error('ASSERTION FAILED: ' + msg); log('ok -', msg); };

(async () => {
  const browser = await chromium.launch();
  const context = await browser.newContext({ acceptDownloads: true, viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();
  page.on('pageerror', (e) => log('PAGE ERROR', e.message));
  await page.goto(BASE + '/');
  await page.getByRole('button', { name: 'Open estimator' }).first().click();
  await page.getByText('PyTorch only').first().waitFor();
  const uploadReq = page.waitForRequest((r) => r.url().includes('/uploads'));
  await page.getByLabel('Upload model scripts / configs').setInputFiles(path.join(files, 'model.py'));
  const analysisId = (await uploadReq).url().split('/api/analysis/')[1].split('/')[0];
  log('active analysis', analysisId);
  await page.getByText('model.py').first().waitFor();
  await page.getByLabel('Upload training scripts / configs').setInputFiles(path.join(files, 'train.py'));
  await page.getByText('train.py').first().waitFor();
  await page.getByLabel('Upload a dataset sample (optional)').setInputFiles(path.join(files, 'sample.csv'));
  await page.getByText(/Sample statistics · sample.csv/).waitFor();
  expect(!fs.existsSync(path.join(dir, 'EXECUTED_SENTINEL')), 'uploaded model.py was not executed');
  await page.waitForTimeout(800);
  expect(await page.locator('#field-model\\.profile').inputValue() === 'mlp_regression', 'profile inferred from model.py');
  expect(await page.locator('#field-model\\.input_size').inputValue() === '100', 'input size extracted');
  expect(await page.locator('#field-model\\.hidden_size').inputValue() === '64', 'hidden size resolved from constant');
  expect(await page.locator('#field-training\\.batch_size_per_gpu').inputValue() === '16', 'batch size extracted');
  expect(await page.locator('#field-training\\.optimizer').inputValue() === 'AdamW', 'optimizer extracted');
  const acc = page.locator('[data-field="training.gradient_accumulation_steps"]');
  expect((await acc.innerText()).includes('Could not be resolved statically'), 'unresolved accumulation flagged in red with message');
  expect(await page.locator('#field-training\\.gradient_accumulation_steps').getAttribute('aria-invalid') === 'true', 'unresolved field marked invalid');
  expect((await page.locator('[data-field="dataset.num_train_examples"]').innerText()).includes('*'), 'full example count marked with asterisk');
  expect(await page.locator('#field-dataset\\.num_train_examples').inputValue() === '', 'sample row count NOT used as full dataset size');
  const continueBtn = page.getByRole('button', { name: 'Continue to hardware' });
  expect(await continueBtn.isDisabled(), 'progression blocked while required fields missing');
  await page.screenshot({ path: path.join(shots, '01-workload-extracted.png'), fullPage: true });

  // User completes and edits.
  await page.locator('#field-training\\.gradient_accumulation_steps').fill('2');
  await page.locator('#field-training\\.parameter_dtype').selectOption('fp32');
  await page.locator('#field-training\\.precision').selectOption('fp32');
  await page.locator('#field-dataset\\.num_train_examples').fill('200000');
  await page.locator('#field-training\\.batch_size_per_gpu').fill('24'); // manual edit
  await page.locator('#field-dataset\\.assume_data_ready').selectOption('true');
  await page.waitForTimeout(900);
  expect(await page.locator('#field-training\\.gradient_dtype').inputValue() === 'fp32', 'gradient dtype default derived from parameter dtype');
  // Re-upload the same training script: manual batch edit must be kept, extracted value offered.
  await page.getByLabel('Upload training scripts / configs').setInputFiles(path.join(files, 'train.py'));
  await page.getByText(/Your edits were kept for/).waitFor();
  expect(await page.locator('#field-training\\.batch_size_per_gpu').inputValue() === '24', 'manual edit not overwritten by re-extraction');
  expect((await page.locator('[data-field="training.batch_size_per_gpu"]').innerText()).includes('Use extracted'), 'extracted value offered beside field');
  const reviewBtn = page.getByRole('button', { name: 'I reviewed these' });
  if (await reviewBtn.count()) await reviewBtn.first().click();
  await page.waitForTimeout(600);
  expect(!(await continueBtn.isDisabled()), 'continue enabled once required fields valid');
  await continueBtn.click();

  // Hardware
  await page.getByLabel('Search GPUs and Nebius platforms').fill('h100');
  const results = page.getByLabel('Hardware search results').locator('li');
  await results.first().waitFor();
  expect(await results.count() <= 10, `at most 10 results shown (${await results.count()})`);
  const firstText = await results.first().innerText();
  expect(/80 GiB VRAM/.test(firstText) && /per GPU-hour/.test(firstText), 'result shows VRAM and price with unit: ' + firstText.split('\n').join(' | '));
  await results.first().locator('button').click();
  await page.waitForTimeout(900);
  expect(await page.locator('#field-hardware\\.total_vram_gib').inputValue() === '80', 'VRAM autofilled from catalogue');
  expect(await page.locator('#field-hardware\\.usable_vram_gib').inputValue() === '72', 'usable VRAM defaults to 90%');
  expect(Number(await page.locator('#field-hardware\\.price_per_hour').inputValue()) > 0, 'price autofilled');
  expect(await page.locator('#field-hardware\\.price_unit').inputValue() === 'gpu_hour', 'price unit autofilled as per GPU-hour');
  await page.getByLabel('Search GPUs and Nebius platforms').fill('b200');
  await page.getByLabel('Hardware search results').getByText(/B200/).first().waitFor();
  expect(/no catalogue specs/i.test(await results.first().innerText()), 'missing catalogue specs shown visibly');
  await page.getByText('How do I find my local hardware specs?').click();
  expect((await page.locator('code').first().innerText()).includes('nvidia-smi'), 'copyable local hardware command shown');
  await page.screenshot({ path: path.join(shots, '02-hardware.png'), fullPage: true });
  const hwReview = page.getByRole('button', { name: 'I reviewed these' });
  if (await hwReview.count()) await hwReview.first().click();
  await page.getByRole('button', { name: 'Run prediction' }).click();

  // Results
  await page.getByText(/of the usable VRAM limit per GPU/).waitFor({ timeout: 20000 });
  const res = await page.locator('#results').innerText();
  expect(/Fits within usable VRAM/i.test(res), 'fit status shown');
  expect(/72\.00 GiB of 80\.00 GiB total/.test(res), 'usable and total VRAM denominator shown');
  expect(/\$[0-9.,]+/.test(res), 'estimated cost shown');
  expect(/Estimated benchmark cost/i.test(res) && /warm-up/.test(res), 'benchmark cost shown');
  expect(/This is the original run/.test(res), 'first run is the original snapshot');
  await page.getByRole('button', { name: 'Compare Nebius GPUs' }).click();
  await page.getByText(/Same workload, GPU count/).waitFor({ timeout: 30000 });
  await page.getByText('How we calculated this').click();
  expect(/B\*\(I \+ H\)\*b_act/.test(await page.locator('#results').innerText()), 'formulas shown in How we calculated this');
  await page.screenshot({ path: path.join(shots, '03-results-prediction.png'), fullPage: true });

  // Benchmark mode: download script, run it (CPU dry run with torch), import result.
  await page.getByRole('tab', { name: 'benchmark' }).click();
  await page.getByRole('button', { name: 'Prepare benchmark' }).click();
  await page.getByText('Run on Nebius').waitFor();
  const cmd = await page.locator('code').first().innerText();
  expect(cmd.startsWith('python run_benchmark.py --model model.py') && cmd.includes('--data-sample sample.csv') && cmd.includes('--output'), 'CLI command matches uploads: ' + cmd);
  expect(await page.getByRole('button', { name: /Start paid Nebius benchmark/ }).isDisabled(), 'Nebius run disabled/unavailable (no fake execution)');
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Download run_benchmark.py' }).click()]);
  const scriptPath = path.join(files, 'run_benchmark.py'); await download.saveAs(scriptPath);
  expect(fs.readFileSync(scriptPath, 'utf8').includes('"profile": "mlp_regression"'), 'downloaded script embeds the analysis config');
  if (process.env.BENCH_PYTHON) {
    const out = path.join(files, 'benchmark_result.json');
    execFileSync(process.env.BENCH_PYTHON, [scriptPath, '--model', 'model.py', '--data-sample', 'sample.csv', '--output', out, '--device', 'cpu', '--steps', '3', '--warmup', '1'], { cwd: files, stdio: 'inherit' });
    expect(fs.existsSync(path.join(dir, 'EXECUTED_SENTINEL')), 'local benchmark (user-run) executed user model code as documented');
    await page.locator('input[type=file][accept*=json]').last().setInputFiles(out);
    await page.getByText('Measured · benchmark').waitFor();
    const bench = await page.locator('#results').innerText();
    expect(/matches current config/i.test(bench), 'imported benchmark matches current config');
    expect(/not representative/i.test(bench), 'CPU dry run flagged not representative');
    await page.screenshot({ path: path.join(shots, '04-results-benchmark.png'), fullPage: true });
  } else log('skipped - benchmark run/import (set BENCH_PYTHON to a Python with torch)');

  // Edit and re-run: original/current comparison.
  await page.getByRole('tab', { name: 'prediction' }).click();
  await page.getByRole('button', { name: 'Edit inputs' }).click();
  await page.locator('#field-training\\.batch_size_per_gpu').fill('48');
  await page.waitForTimeout(700);
  await page.getByRole('button', { name: 'Continue to hardware' }).click();
  await page.getByRole('button', { name: 'Run prediction' }).click();
  await page.getByText(/Original batch size per gpu/i).waitFor({ timeout: 20000 });
  const diff = await page.getByText(/Original batch size per gpu/i).innerText();
  expect(/24.*current: 48/.test(diff), 'change shown relative to original: ' + diff);
  await page.screenshot({ path: path.join(shots, '05-results-diff.png'), fullPage: true });

  // Page close -> beacon clears the analysis cache.
  const cacheDir = path.join(CACHE, analysisId);
  expect(fs.existsSync(cacheDir), 'analysis cache folder exists before close (check E2E_CACHE_DIR)');
  await page.close({ runBeforeUnload: true });
  await new Promise((r) => setTimeout(r, 1500));
  expect(!fs.existsSync(cacheDir), 'cache folder removed after page close (beacon)');
  await browser.close();
  log('screenshots in', shots);
  log('UPLOAD FLOW PASSED');
})().catch((e) => { console.error(e); process.exit(1); });
