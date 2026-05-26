import { expect, test } from '@playwright/test'

test('OPD explorer smoke', async ({ page }) => {
  await page.goto('/')

  await expect(page).toHaveTitle(/OPD Training Explorer/i)
  await expect(
    page.getByRole('heading', { name: 'OPD Training Explorer' }),
  ).toBeVisible()

  const tinyDemoRun = page.getByRole('button', { name: /tiny-demo/i })
  await expect(tinyDemoRun).toBeVisible()
  await expect(tinyDemoRun).toHaveAttribute('aria-pressed', 'true')

  await expect(page.getByRole('heading', { name: 'Phase timing per step' })).toBeVisible()

  await expect(
    page.locator('.chart-plot svg.recharts-surface').first(),
  ).toBeVisible()
  await expect(page.getByText(/generation/i).first()).toBeVisible()
})
