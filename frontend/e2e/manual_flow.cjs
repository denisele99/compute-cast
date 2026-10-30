// GPT-2 MVP E2E (flow 2): Hugging Face model (locked dims) -> exclusive model sources -> ZIP upload ->
// conflicts need an explicit hypothetical -> custom GPU with VRAM only (memory yes, runtime/cost no) ->
// error details in hover popovers. Env: E2E_BASE_URL, E2E_OUT (see upload_flow.cjs).
const { chromium } = require('playwright');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');
const BASE = process.env.E2E_BASE_URL || 'http://127.0.0.1:5173';
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'cc-e2e2-'));
const shots = process.env.E2E_OUT || dir;
const examples = path.resolve(__dirname, '..', '..', 'examples', 'gpt2_mvp');
const log = (...a) => console.log('[e2e]', ...a);
const expect = (cond, msg) => { if (!cond) throw new Error('ASSERTION FAILED: ' + msg); log('ok -', msg); };
const zip = path.join(dir, 'bundle.zip');
execFileSync('python3', ['-c', `import zipfile,sys; z=zipfile.ZipFile(sys.argv[1],'w'); z.write(sys.argv[2]+'/config.json','proj/config.json'); z.write(sys.argv[2]+'/data_config.json','proj/data_config.json'); z.writestr('proj/notes.txt','x'); z.close()`, zip, examples]);

(async () => {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1280, height: 900 } })).newPage();
  page.on('pageerror', (e) => log('PAGE ERROR', e.message));
  const f = (k) => page.locator('#field-' + k.replace(/\./g, '\\.'));
  await page.goto(BASE + '/');
  await page.getByRole('button', { name: 'Open estimator' }).first().click();
  await page.getByText('Hugging Face model', { exact: true }).waitFor();
  const options = await page.getByLabel('Hugging Face model').locator('option').allInnerTexts();
  expect(options.length === 2 && options[0].startsWith('openai-community/gpt2') && options[1].startsWith('distilbert/distilgpt2'), 'only the two supported HF models');
  await page.waitForTimeout(500);
  const missing = page.locator('[data-field="model.num_layers"]');
  await missing.getByRole('button').hover();
  expect(/Required for the memory estimate/.test(await page.getByRole('tooltip').innerText()), 'missing-field message shown in hover popover, not under the box');
  await page.getByLabel('Hugging Face model').selectOption('distilbert/distilgpt2');
  await page.getByRole('button', { name: 'Use', exact: true }).click();
  await page.getByRole('button', { name: 'Remove Hugging Face model' }).waitFor();
  await page.waitForTimeout(800);
  expect(await f('model.num_layers').inputValue() === '6', 'distilgpt2 config used (6 layers)');
  expect(await f('model.num_layers').isDisabled(), 'HF dimensions locked');

  await page.getByLabel('Upload files').setInputFiles(zip);
  await page.getByText(/Skipped: bundle.zip: proj\/notes.txt \(unsupported type\)/).waitFor();
  log('ok - ZIP expanded; unsupported member skipped visibly');
  await page.waitForTimeout(800);
  expect(await page.getByText('Two model sources.').isVisible() && await page.getByRole('button', { name: 'Continue to hardware' }).isDisabled(), 'two model sources rejected visibly');
  await page.getByRole('button', { name: 'Remove Hugging Face model' }).click();
  await page.waitForTimeout(900);
  expect(await f('model.num_layers').inputValue() === '12', 'uploaded config.json now defines the model');

  await page.getByRole('button', { name: 'GPT-2 with unsupported settings (shows conflicts)', exact: true }).click();
  await page.getByText('Unsupported settings in your files').waitFor();
  await page.waitForTimeout(800);
  const continueBtn = page.getByRole('button', { name: 'Continue to hardware' });
  expect(await continueBtn.isDisabled(), 'conflicts block progression');
  for (const box of await page.getByRole('checkbox', { name: /\(hypothetical\)/ }).all()) await box.check();
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(shots, '04-conflicts-hypothetical.png'), fullPage: true });
  const review = page.getByRole('button', { name: 'Mark all reviewed' }); if (await review.count()) await review.click();
  await page.getByLabel('Hypothetical custom configuration').check();
  await f('model.num_layers').fill('24');
  await page.waitForTimeout(800);
  expect(!(await continueBtn.isDisabled()), 'continue enabled after choosing hypotheticals');
  await continueBtn.click();

  await page.getByRole('button', { name: 'Custom GPU' }).click();
  await f('hardware.gpu_name').fill('My workstation GPU');
  await f('hardware.vram_gib').fill('24');
  await page.waitForTimeout(800);
  expect(await f('hardware.usable_vram_gib').inputValue() === '24', 'usable limit defaults to VRAM for custom GPU');
  await page.getByRole('button', { name: 'Run estimate' }).click();
  await page.getByText(/usable VRAM limit per GPU \(/).waitFor({ timeout: 20000 });
  const res = await page.locator('#results').innerText();
  expect(/HYPOTHETICAL SCENARIO/i.test(res) && /instead of fp16/.test(res) && /Layers \(n_layer\): 24 instead of 12/.test(res), 'hypothetical scenario labelled');
  expect(/Needs: .*Peak BF16 dense/.test(res), 'runtime missing inputs listed');
  expect(/Runtime is unavailable, so cost cannot be estimated/.test(res), 'no fabricated cost');
  await page.screenshot({ path: path.join(shots, '05-custom-results.png'), fullPage: true });
  await browser.close();
  log('FLOW 2 PASSED');
})().catch((e) => { console.error(e); process.exit(1); });
