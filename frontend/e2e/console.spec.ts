import { expect, test } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test('explains a persisted regression, preserves provenance, and opens run detail', async ({
  page,
  request,
}) => {
  const response = await request.get('/v1/comparisons');
  expect(response.ok()).toBeTruthy();
  const comparisons = await response.json();
  const regression = comparisons.find((result: { verdict: string }) => result.verdict === 'fail');
  expect(regression).toBeDefined();
  await page.goto(`/compare?comparison=${regression.id}`);
  await expect(page.getByRole('heading', { name: 'Compare models' })).toBeVisible();
  await expect(page.getByText('FAIL', { exact: true })).toBeVisible();
  await expect(page.getByRole('img', { name: /Accuracy difference/ })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Where behavior changed' })).toBeVisible();
  await expect(page.getByRole('tab', { name: /Regressions/ })).toHaveAttribute(
    'aria-selected',
    'true',
  );
  await page.getByRole('tab', { name: /Regressions/ }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('tab', { name: /Fixes/ })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('tab', { name: /Fixes/ })).toBeFocused();
  await page.getByRole('link', { name: 'Inspect run provenance' }).click();
  await expect(page.getByRole('heading', { name: 'Execution provenance' })).toBeVisible();
});

test('registers an immutable candidate and surfaces server validation', async ({ page }) => {
  const name = `browser-test-${Date.now()}`;
  await page.goto('/candidates');
  await page.getByRole('button', { name: 'Register candidate', exact: true }).click();
  await page.getByLabel('Name', { exact: true }).fill(name);
  await page
    .getByLabel('Hugging Face model ID')
    .fill('distilbert/distilbert-base-uncased-finetuned-sst-2-english');
  await page.getByLabel('Model revision').fill('a'.repeat(40));
  await page.getByRole('button', { name: 'Register candidate', exact: true }).click();
  await expect(page.getByRole('heading', { name, exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Register candidate', exact: true }).click();
  await page.getByLabel('Name', { exact: true }).fill(name);
  await page
    .getByLabel('Hugging Face model ID')
    .fill('distilbert/distilbert-base-uncased-finetuned-sst-2-english');
  await page.getByLabel('Model revision').fill('a'.repeat(40));
  await page.getByRole('button', { name: 'Register candidate', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('already exists');
});

test('saved comparisons restore their margin', async ({ page }) => {
  await page.route('**/v1/comparisons', async (route) => {
    const response = await route.fetch();
    const data = await response.json();
    data[0].margin = 0.05;
    await route.fulfill({ response, json: data });
  });
  await page.goto('/compare');
  await expect(page.getByLabel('Accepted loss in accuracy points')).toHaveValue('5');
});

test('generative registration supplies compatible settings', async ({ page }) => {
  const name = `generative-browser-${Date.now()}`;
  await page.goto('/candidates');
  await page.getByRole('button', { name: 'Register candidate', exact: true }).click();
  await page.getByLabel('Inference type').selectOption('generative');
  await expect(page.getByLabel('Batch size')).toBeDisabled();
  await expect(page.getByLabel('Batch size')).toHaveValue('1');
  await expect(page.getByLabel('Quantization')).toBeDisabled();
  await expect(page.getByLabel('Positive class index')).toBeDisabled();
  await page.getByLabel('Name', { exact: true }).fill(name);
  await page.getByLabel('Hugging Face model ID').fill('Qwen/Qwen3-0.6B');
  await page.getByLabel('Model revision').fill('c'.repeat(40));
  await page.getByLabel('Prompt version').selectOption('sentiment-few-v1');
  await page.getByRole('button', { name: 'Register candidate', exact: true }).click();
  const card = page
    .getByRole('article')
    .filter({ has: page.getByRole('heading', { name, exact: true }) });
  await expect(card).toContainText('sentiment-few-v1');
  await expect(card).toContainText('BF16');
  await expect(card).toContainText('8 tokens · greedy');
});

test('fetches the final prediction page when a run completes', async ({ page }) => {
  let hits = 0;
  await page.route('**/v1/runs/test-run', (route) =>
    route.fulfill({
      json: {
        id: 'test-run',
        candidate_id: 'candidate',
        dataset_id: 'dataset',
        status: ++hits === 1 ? 'running' : 'succeeded',
        attempts: 1,
        max_attempts: 3,
        env: {},
        metrics: null,
        error: null,
        code_version: 'test',
        created_at: '2026-10-04T12:00:00Z',
        candidate_name: 'Completion test',
        dataset_name: 'Fixture',
        progress: hits === 1 ? 0 : 1,
        total: 1,
      },
    }),
  );
  await page.route('**/v1/runs/test-run/predictions?*', (route) =>
    route.fulfill({
      json:
        hits === 1
          ? []
          : [
              {
                idx: 0,
                text: 'Final prediction after completion',
                label: 1,
                pred: 1,
                prob_pos: 0.9,
                correct: true,
                slices: [],
              },
            ],
    }),
  );
  await page.goto('/runs/test-run');
  await expect(page.getByText('Final prediction after completion')).toBeVisible({ timeout: 10000 });
});

test('main pages pass automated WCAG accessibility checks', async ({ page }) => {
  for (const path of ['/compare', '/candidates', '/runs']) {
    await page.goto(path);
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
    await expect(page.getByRole('status')).toHaveCount(0);
    const results = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
      .analyze();
    expect(results.violations).toEqual([]);
  }
});

test('comparison shows a dataset mismatch error', async ({ page }) => {
  await page.route('**/v1/comparisons?*', (route) => route.continue());
  await page.route('**/v1/comparisons', async (route) => {
    if (route.request().method() === 'POST')
      return route.fulfill({
        status: 422,
        contentType: 'application/problem+json',
        body: JSON.stringify({
          detail: 'Runs must use the same immutable, content-hashed dataset.',
        }),
      });
    return route.continue();
  });
  await page.goto('/compare');
  await expect(page.getByText('FAIL', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Compare runs' }).click();
  await expect(page.getByRole('alert')).toContainText('same immutable');
});

test('usable at a narrow viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/compare');
  await expect(page.getByRole('heading', { name: 'Compare models' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Compare runs' })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
});
