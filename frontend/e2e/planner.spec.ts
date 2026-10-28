import { test, expect } from '@playwright/test';

test('manual workload validates, connects to API, estimates, compares and preserves first run', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByText('PyTorch scripts only.', { exact: false })).toBeVisible();
  await page.getByLabel('Training examples', { exact: false }).fill('10');
  await page.getByLabel('Batch size per GPU', { exact: false }).fill('20');
  await page.getByRole('button', { name: 'Review hardware' }).click();
  await expect(page.getByText('Batch size cannot exceed training examples.')).toBeVisible();
  await page.getByLabel('Batch size per GPU', { exact: false }).fill('2');
  await page.getByRole('button', { name: 'Review hardware' }).click();
  await expect(page.getByRole('heading', { name: 'Hardware', exact: true })).toBeVisible();
  await page.getByLabel('Price (USD / GPU-hour)', { exact: false }).fill('2');
  await page.getByLabel('Matching measured seconds per microbatch', { exact: false }).fill('1');
  await page.getByLabel('Matching measured peak VRAM per GPU', { exact: false }).fill('6');
  await page.getByRole('button', { name: 'Run estimate', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Results', exact: true })).toBeVisible();
  await expect(page.getByText('25.0% of physical capacity', { exact: false })).toBeVisible();
  await expect(page.getByText('50 microbatches per rank', { exact: false })).toBeVisible();
  await page.getByRole('button', { name: 'Benchmark', exact: true }).click();
  await expect(page.getByRole('link', { name: 'Download benchmark script' })).toBeVisible();
  const script = await page.request.get('/api/benchmark/script');
  expect(await script.text()).toContain('build_benchmark');
  await expect(page.getByRole('button', { name: /Run benchmark on Nebius/ })).toBeDisabled();
  await page.getByRole('button', { name: 'Edit inputs' }).click();
  await page.getByLabel('Price (USD / GPU-hour)', { exact: false }).fill('3');
  await page.getByRole('button', { name: 'Run estimate', exact: true }).click();
  await expect(page.getByText('hardware.price_per_gpu_hour:', { exact: false })).toBeVisible();
  await page.getByRole('button', { name: 'New analysis' }).click();
  await expect(page.getByRole('heading', { name: 'Model & training inputs' })).toBeVisible();
});

test('Python/YAML uploads use real parser, require review, and flag conflicts', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Upload model and training files').setInputFiles([
    { name: 'train.py', mimeType: 'text/plain', buffer: Buffer.from('batch_size = 8\nepochs = dynamic_epochs()') },
    { name: 'model.yaml', mimeType: 'text/plain', buffer: Buffer.from('total_params: 1000000\nnum_examples: 100\n') },
  ]);
  await expect(page.getByLabel('Batch size per GPU', { exact: false })).toHaveValue('8');
  await expect(page.getByLabel('Total model parameters', { exact: false })).toHaveValue('1000000');
  await expect(page.getByText('Conflicting or dynamic values', { exact: false })).toBeVisible();
  await page.getByRole('button', { name: 'Review hardware' }).click();
  await expect(page.getByText('Resolve upload errors and confirm', { exact: false })).toBeVisible();
  await page.getByLabel('I have verified the autofilled fields', { exact: false }).check();
  await page.getByRole('button', { name: 'Review hardware' }).click();
  await expect(page.getByRole('heading', { name: 'Hardware', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Run estimate', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Results', exact: true })).toBeVisible();
  await expect(page.getByText('Unknown — more memory metadata', { exact: false })).toBeVisible();
});

test('catalog search autofills snapshot prices and handles unknown memory', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Review hardware' }).click();
  const dropdown = page.getByLabel('GPU / platform', { exact: true });
  await expect(dropdown.locator('option')).toHaveCount(11);
  await dropdown.selectOption({ index: 1 });
  await expect(page.getByLabel('Price (USD / GPU-hour)', { exact: false })).toHaveValue('2.2');
  expect(Number(await page.getByLabel('Memory per GPU (GiB)', { exact: false }).inputValue())).toBeCloseTo(131.316, 2);
  await page.getByLabel('Search Nebius platforms / regions').fill('rtx');
  await expect(dropdown.locator('option')).toHaveCount(5);
  await dropdown.selectOption({ index: 2 });
  await expect(page.getByLabel('Memory per GPU (GiB)', { exact: false })).toHaveValue('');
  await page.getByRole('button', { name: 'Run estimate', exact: true }).click();
  await expect(page.getByText('Required: enter a value', { exact: false })).toBeVisible();
});

test('multiple GPUs require explicit strategy and changed workload clears stale timing', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Review hardware' }).click();
  await page.getByLabel('Available identical GPUs', { exact: false }).fill('4');
  await page.getByLabel('How many GPUs are used', { exact: false }).fill('2');
  await page.getByRole('button', { name: 'Run estimate', exact: true }).click();
  await expect(page.getByText('Choose DDP or an explicit sharding strategy', { exact: false })).toBeVisible();
  await page.getByLabel('Distribution / sharding', { exact: false }).selectOption('ddp');
  await page.getByLabel('Matching measured seconds per microbatch', { exact: false }).fill('1');
  await page.getByLabel('Available identical GPUs', { exact: false }).fill('8');
  await expect(page.getByLabel('Matching measured seconds per microbatch', { exact: false })).toHaveValue('');
  await page.getByRole('button', { name: 'Run estimate', exact: true }).click();
  await expect(page.getByText('31,250 microbatches per rank', { exact: false })).toBeVisible();
});

test('benchmark report import updates results and refresh clears upload cache', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Upload model and training files').setInputFiles({ name: 'train.py', mimeType: 'text/plain', buffer: Buffer.from('batch_size = 8\nepochs = 2\nnum_examples = 100') });
  await expect(page.getByLabel('Batch size per GPU', { exact: false })).toHaveValue('8');
  await page.getByLabel('I have verified the autofilled fields', { exact: false }).check();
  await page.getByRole('button', { name: 'Review hardware' }).click();
  const response = page.waitForResponse(response => response.url().endsWith('/api/analysis/estimate'));
  await page.getByRole('button', { name: 'Run estimate', exact: true }).click();
  const result = await (await response).json();
  await page.getByRole('button', { name: 'Benchmark', exact: true }).click();
  const report = { schema_version: 1, config: result.config, steps: 20, seconds_per_batch: .25, measured_peak_vram_gib: 4, gpu_name: 'Test GPU', capacity_gib: 24, scope: 'whole_training_microbatch', note: 'Local report' };
  await page.getByLabel('Import benchmark report').setInputFiles({ name: 'benchmark.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(report)) });
  await expect(page.getByText('Imported measurements from Test GPU.', { exact: false })).toBeVisible();
  await page.getByRole('button', { name: 'Run estimate again' }).click();
  await expect(page.getByText('Imported local benchmark measurements', { exact: false })).toBeVisible();
  await expect(page.getByText('16.7% of physical capacity', { exact: false })).toBeVisible();
  await page.screenshot({ path: '/tmp/compute-cast-connected-results.png', fullPage: true });
  await page.reload();
  await expect(page.getByRole('heading', { name: 'Model & training inputs' })).toBeVisible();
  await expect(page.getByText('train.py', { exact: true })).toHaveCount(0);
  await expect(page.getByLabel('Batch size per GPU', { exact: false })).toHaveValue('16');
});

test('metadata-only prediction uses explicit throughput assumptions without a benchmark', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Total model parameters', { exact: false }).fill('1000000');
  await page.getByText('Advanced model / training settings', { exact: true }).click();
  await page.getByLabel('Forward/backward FLOPs per microbatch', { exact: false }).fill('10000000000');
  await page.getByLabel('HBM traffic bytes per microbatch', { exact: false }).fill('1000000000');
  await page.getByLabel('Assume data is already on the GPU', { exact: false }).check();
  await page.getByRole('button', { name: 'Review hardware' }).click();
  await page.getByText('Advanced hardware settings · optional', { exact: true }).click();
  await page.getByLabel('GPU dense compute peak', { exact: false }).fill('30');
  await page.getByLabel('GPU FP32 peak for optimizer', { exact: false }).fill('30');
  await page.getByLabel('GPU memory bandwidth', { exact: false }).fill('900');
  await page.getByLabel('Price (USD / GPU-hour)', { exact: false }).fill('2');
  await page.getByRole('button', { name: 'Run estimate', exact: true }).click();
  await expect(page.getByText('Timing source: metadata-only planning estimate', { exact: false })).toBeVisible();
  await expect(page.getByText('Planning scenarios:', { exact: false })).toBeVisible();
});
