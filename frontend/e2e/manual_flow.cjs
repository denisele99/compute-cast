// E2E: manual-entry flow (classic transformer), unsupported-profile handling, HF error state,
// custom hardware with missing specs (no fabricated runtime/cost), multi-GPU validation.
// Requires the backend and frontend dev server running. Env: E2E_BASE_URL, E2E_OUT (see upload_flow.cjs).
const { chromium } = require('playwright');
const fs = require('fs');
const os = require('os');
const path = require('path');
const BASE = process.env.E2E_BASE_URL || 'http://127.0.0.1:5173';
const shots = process.env.E2E_OUT || fs.mkdtempSync(path.join(os.tmpdir(), 'cc-e2e-shots-'));
const log = (...a) => console.log('[e2e]', ...a);
const expect = (cond, msg) => { if (!cond) throw new Error('ASSERTION FAILED: ' + msg); log('ok -', msg); };
(async () => {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
  page.on('pageerror', (e) => log('PAGE ERROR', e.message));
  await page.goto(BASE + '/');
  await page.getByRole('button', { name: 'Open estimator' }).first().click();
  await page.getByText('PyTorch only').first().waitFor();
  const f = (k) => page.locator('#field-' + k.replace(/\./g, '\\.'));
  const block = (k) => page.locator(`[data-field="${k}"]`);
  await page.waitForTimeout(600);
  expect((await block('model.profile').innerText()).includes('*'), 'profile marked required');
  await f('model.profile').selectOption('transformer_lm_classic');
  await page.waitForTimeout(500);
  // HF lookup: either real results or an explicit error (e.g. offline) — never silent.
  await page.getByLabel('Hugging Face model search').fill('gpt2');
  await page.getByRole('button', { name: 'Search models' }).click();
  await page.getByText(/Could not reach the Hugging Face Hub|Hugging Face Hub returned|Use config/).first().waitFor({ timeout: 20000 });
  log('ok - HF search shows results or an explicit error');
  const fill = async (k, v) => f(k).fill(String(v));
  const pick = async (k, v) => f(k).selectOption(String(v));
  await fill('model.vocab_size', 50257); await fill('model.hidden_size', 768); await fill('model.num_hidden_layers', 12); await fill('model.num_attention_heads', 12);
  await pick('model.position_embedding', 'learned'); await fill('model.max_position_embeddings', 1024); await pick('model.tie_word_embeddings', 'true'); await pick('model.linear_bias', 'true');
  await pick('training.task', 'causal_language_modeling'); await fill('training.batch_size_per_gpu', 8); await fill('training.epochs', 1);
  await pick('training.precision', 'bf16'); await pick('training.parameter_dtype', 'fp32'); await pick('training.optimizer', 'AdamW');
  await fill('training.sequence_length', 1024); await pick('training.attention_backend', 'sdpa');
  await fill('dataset.num_train_examples', 50000);
  await page.waitForTimeout(900);
  expect((await block('training.attention_backend').innerText()).includes('Unsupported'), 'SDPA attention explained as unsupported beside the field');
  expect(await page.getByRole('button', { name: 'Continue to hardware' }).isDisabled(), 'unsupported setting blocks progression');
  await pick('training.attention_backend', 'eager');
  await page.getByText('Advanced model settings').click();
  await fill('model.num_key_value_heads', 4);
  await page.waitForTimeout(800);
  expect(/Grouped\/multi-query attention is not supported/.test(await block('model.num_key_value_heads').innerText()), 'GQA explained as unsupported');
  await fill('model.num_key_value_heads', 12);
  await page.waitForTimeout(800);
  for (const k of ['training.loss', 'training.input_dtype', 'training.gradient_dtype', 'training.optimizer_state_dtype'])
    expect((await block(k).innerText()).includes('default assumption'), `${k} filled as visible default assumption`);
  await page.screenshot({ path: path.join(shots, '06-manual-workload.png'), fullPage: true });
  const rb = page.getByRole('button', { name: 'I reviewed these' }); if (await rb.count()) await rb.first().click();
  await page.getByRole('button', { name: 'Continue to hardware' }).click();
  // Custom hardware, no specs: memory works, runtime/cost explicitly incomplete.
  await page.getByRole('button', { name: 'Enter custom hardware' }).click();
  await fill('hardware.gpu_name', 'My workstation GPU'); await fill('hardware.total_vram_gib', 24);
  await page.waitForTimeout(800);
  expect(await f('hardware.usable_vram_gib').inputValue() === '21.6', 'usable VRAM default 90% for custom hardware');
  await fill('hardware.gpu_count_available', 2); await fill('training.gpu_count_used', 2);
  await page.waitForTimeout(800);
  expect(/choose DDP, ZeRO or FSDP explicitly/.test(await block('training.distributed_strategy').innerText()), 'multi-GPU without strategy is an explicit error');
  expect(await page.getByRole('button', { name: 'Run prediction' }).isDisabled(), 'estimation blocked by strategy error');
  await pick('training.distributed_strategy', 'ddp');
  await page.waitForTimeout(800);
  await page.getByRole('button', { name: 'Run prediction' }).click();
  await page.getByText(/of the usable VRAM limit per GPU/).waitFor({ timeout: 20000 });
  const res = await page.locator('#results').innerText();
  expect(/21\.60 GiB of 24\.00 GiB total/.test(res), 'custom usable/total VRAM used');
  expect(/124,439,808 parameters/.test(res), 'GPT-2 parameter count from formula');
  expect(/Runtime\/cost incomplete/i.test(res) && /peak_compute_tflops|bandwidth/i.test(res), 'missing runtime inputs listed instead of numbers');
  expect((res.match(/Elapsed job time is not available/g) || []).length === 2, 'both cost scenarios unavailable (no fabricated cost)');
  expect(/benchmark duration cannot be estimated/.test(res), 'benchmark cost unavailable without runtime');
  await page.screenshot({ path: path.join(shots, '07-manual-results.png'), fullPage: true });
  await browser.close();
  log('MANUAL FLOW PASSED');
})().catch((e) => { console.error(e); process.exit(1); });
