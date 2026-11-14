// Uses an existing Playwright/Chromium installation. No browser download is required.
import assert from 'node:assert/strict';
import {mkdir, readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const artifacts = path.join(root, 'artifacts'); await mkdir(artifacts, {recursive: true});
await mkdir(path.join(artifacts, 'downloads'), {recursive: true});
const browser = await chromium.launch({headless: true,
  downloadsPath: path.join(artifacts, 'downloads'),
  ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {}), args: ['--no-sandbox']});
const context = await browser.newContext({viewport: {width: 1512, height: 1100}, acceptDownloads: true});
const page = await context.newPage(), errors = [];
page.on('pageerror', error => errors.push(error.message));
const base = process.env.CAPBODY_URL || 'http://127.0.0.1:8091';
async function drawSourceBox(box, imageWidth, imageHeight, factor = 1, pan = [0, 0]) {
  const r = await page.locator('#canvas').boundingBox();
  const scale = Math.min((r.width-32)/imageWidth, (r.height-32)/imageHeight)*factor;
  const x = r.x+(r.width-imageWidth*scale)/2+pan[0], y = r.y+(r.height-imageHeight*scale)/2+pan[1];
  await page.mouse.move(x+box[0]*scale, y+box[1]*scale); await page.mouse.down();
  await page.mouse.move(x+box[2]*scale, y+box[3]*scale, {steps: 8}); await page.mouse.up();
}
async function saveProject(name) {
  const downloadPromise = page.waitForEvent('download'); await page.locator('#save-project').click();
  const download = await downloadPromise, filename = path.join(artifacts, name); await download.saveAs(filename);
  return {filename, project: JSON.parse(await readFile(filename, 'utf8'))};
}
try {
  await page.goto(base); await page.waitForSelector('#engine.ready');
  await page.screenshot({path: path.join(artifacts, 'workspace-empty.png'), fullPage: true});
  await page.locator('#image-input').setInputFiles(process.env.CAPBODY_TEST_IMAGE || path.join(root, 'data/test.png'));
  await page.waitForFunction(() => document.querySelector('#file-meta').textContent.includes('1080 × 960'));
  assert.equal(await page.locator('#analyze').isEnabled(), false);
  await drawSourceBox([766, 579, 802, 605], 1080, 960);
  await drawSourceBox([864, 548, 904, 577], 1080, 960);
  assert.equal(await page.locator('#cap-example-count').textContent(), '2 / 3');
  const before = await saveProject('examples.json');
  for (let i = 0; i < 2; i++) {
    const expected = [[766, 579, 802, 605], [864, 548, 904, 577]][i];
    before.project.cap_exemplars[i].forEach((v, j) => assert.ok(Math.abs(v-expected[j]) < 1, 'Canvas coordinates must match source pixels'));
  }
  const responsePromise = page.waitForResponse(r => r.url().endsWith('/count/cap-body'), {timeout: 120000});
  await page.locator('#analyze').click(); const response = await responsePromise;
  assert.equal(response.status(), 200); const result = await response.json();
  await page.locator('#busy').waitFor({state: 'hidden', timeout: 120000});
  assert.equal(Number(await page.locator('#auto-count').textContent()), result.summary.automatic_count);
  const pending = Number(await page.locator('#review-count').textContent()); assert.ok(pending > 0);
  await page.locator('.object-row').first().click();
  await page.locator('#confirm').click();
  assert.equal(Number(await page.locator('#current-count').textContent()), result.summary.automatic_count+1);
  await page.locator('#undo').click();
  assert.equal(Number(await page.locator('#review-count').textContent()), pending);
  await page.locator('#redo').click();
  assert.equal(Number(await page.locator('#review-count').textContent()), pending-1);
  const saved = await saveProject('reviewed-project.json');
  await page.locator('#project-input').setInputFiles(saved.filename);
  await page.waitForFunction(() => document.querySelector('#message').textContent.includes('Project restored'));
  const restored = await saveProject('restored-project.json');
  assert.deepEqual(restored.project.result, saved.project.result);
  assert.deepEqual(restored.project.report, saved.project.report);
  const csvDownload = page.waitForEvent('download'); await page.locator('#export-csv').click();
  const csvFile = await csvDownload; await csvFile.saveAs(path.join(artifacts, 'reviewed.csv'));
  const lines = (await readFile(path.join(artifacts, 'reviewed.csv'), 'utf8')).split('\r\n');
  assert.equal(lines.length, saved.project.report.current+1);
  const pngDownload = page.waitForEvent('download'); await page.locator('#export-png').click();
  await (await pngDownload).saveAs(path.join(artifacts, 'reviewed.png'));
  await page.screenshot({path: path.join(artifacts, 'workspace-result.png'), fullPage: true});
  await page.setViewportSize({width: 390, height: 844});
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Mobile layout should not overflow');
  await page.screenshot({path: path.join(artifacts, 'workspace-mobile.png'), fullPage: true});
  await page.setViewportSize({width: 1512, height: 1100});
  // A deterministic review fixture tests UI actions independently of model quality.
  const fixture = structuredClone(saved.project);
  fixture.result.objects = [
    {id: 'b001', kind: 'body', box: [660, 600, 738, 816], score: .8, status: 'accepted', partner_id: null, source: 'model', reason: 'body_detection'},
    {id: 'c001', kind: 'cap', box: [668, 580, 710, 605], score: .8, status: 'review', partner_id: null, source: 'model', reason: 'unmatched_cap'},
    {id: 'c002', kind: 'cap', box: [865, 548, 904, 577], score: .8, status: 'review', partner_id: null, source: 'model', reason: 'unmatched_cap'}];
  fixture.result.links = []; fixture.result.summary = {automatic_count: 1};
  await page.locator('#project-input').setInputFiles({name: 'review-fixture.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(fixture))});
  await page.waitForFunction(() => document.querySelector('#current-count').textContent === '1' && document.querySelector('#review-count').textContent === '2');
  await page.locator('#filter').selectOption('all');
  await page.locator('[data-id=c001]').click(); await page.locator('#confirm').click();
  assert.equal(await page.locator('#current-count').textContent(), '2');
  await page.locator('#pair-body').selectOption('b001'); await page.locator('#pair').click();
  assert.equal(await page.locator('#current-count').textContent(), '1', 'Pairing must remove a duplicate count');
  await page.locator('#unpair').click();
  assert.equal(await page.locator('#review-count').textContent(), '2');
  await page.locator('[data-id=b001]').click(); await page.locator('#dismiss').click();
  assert.equal(await page.locator('#current-count').textContent(), '0');
  await page.locator('#restore').click(); await page.locator('#confirm').click();
  assert.equal(await page.locator('#current-count').textContent(), '1');
  await page.locator('#add-tool').click(); await drawSourceBox([450, 300, 500, 600], 1080, 960);
  assert.equal(await page.locator('#current-count').textContent(), '2');
  await page.locator('#undo').click(); assert.equal(await page.locator('#current-count').textContent(), '1');
  await page.locator('#redo').click(); assert.equal(await page.locator('#current-count').textContent(), '2');
  // Browser EXIF normalization and coordinates after both zoom and pan.
  await page.locator('#image-input').setInputFiles(path.join(root, 'tests/fixtures/orientation.jpg'));
  await page.waitForFunction(() => document.querySelector('#file-meta').textContent.includes('60 × 120'));
  await page.locator('#zoom-in').click(); await page.locator('#canvas').focus();
  const bounds = await page.locator('#canvas').boundingBox();
  await page.keyboard.down('Space'); await page.mouse.move(bounds.x+100, bounds.y+100);
  await page.mouse.down(); await page.mouse.move(bounds.x+120, bounds.y+110, {steps: 5}); await page.mouse.up(); await page.keyboard.up('Space');
  await drawSourceBox([10, 20, 30, 35], 60, 120, 1.25, [20, 10]);
  const oriented = await saveProject('orientation-project.json');
  assert.equal(oriented.project.image.width, 60); assert.equal(oriented.project.image.height, 120);
  oriented.project.cap_exemplars[0].forEach((v, j) => assert.ok(Math.abs(v-[10, 20, 30, 35][j]) < 1));
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({api: result.summary, browser: 'real inference, pairing, additions, dismissal/restoration, undo/redo, JSON restore, CSV/PNG, mobile, EXIF and zoom/pan coordinates passed'}, null, 2));
} finally { await browser.close(); }
