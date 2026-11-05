// node tests/check_simulation_parametrization_v2_report.cjs <report.html>
// DOM double for sparse slider stops and image-row continuity.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const page = fs.readFileSync(process.argv[2], 'utf8');
assert(page.includes('Adjusted conclusion ranges (interpolated)'));
const script = page.match(/<script>([\s\S]*?)<\/script>/)[1];
class Element {
  constructor(tag) {
    this.tagName = tag.toUpperCase(); this.children = []; this.attributes = {};
    this.style = {}; this.className = ''; this.textContent = ''; this.value = '';
    this.classList = {add() {}, remove() {}, toggle() {}};
  }
  append(...items) {
    for (const item of items) {
      if (item.tagName === 'FRAGMENT') this.append(...item.children);
      else { item.parent = this; this.children.push(item); }
    }
  }
  replaceChildren(...items) { this.children = []; this.append(...items); }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  addEventListener() {}
  querySelectorAll(tag) { return descendants(this).filter(item => item.tagName === tag.toUpperCase()); }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter(item => item !== this); }
}
class FakeImage extends Element {
  constructor() { super('img'); }
  set src(value) { this._src = value; queueMicrotask(() => this.onload?.()); }
  get src() { return this._src; }
}
function descendants(root) { return root.children.flatMap(child => [child, ...descendants(child)]); }
const elements = Object.fromEntries(['app', 'modal', 'modalview', 'modalimage', 'zoom-in', 'zoom-out', 'zoom-reset', 'modal-close']
  .map(id => [id, new Element(id)]));
const document = {
  getElementById: id => elements[id],
  createElement: tag => new Element(tag),
  createElementNS: (_, tag) => new Element(tag),
  createDocumentFragment: () => new Element('fragment'),
  addEventListener() {},
};
const context = vm.createContext({document, Image: FakeImage, requestAnimationFrame: callback => callback()});
vm.runInContext(script, context, {timeout: 30000});
const data = vm.runInContext('DATA', context);
const flush = () => new Promise(resolve => setImmediate(resolve));

(async () => {
  await flush();
  assert.deepEqual(Object.values(data.sources).map(source => source.snr_role), ['low', 'mid', 'high']);
  for (const [sid, source] of Object.entries(data.sources)) {
    assert.equal(descendants(elements.app).filter(item => item.tagName === 'H3' &&
      item.textContent.startsWith(`${source.snr_role} SNR · ${sid}`)).length, 3);
    for (const kind of ['amp', 'phase', 'noise']) {
      const n = data.boundaries[sid][kind].length;
      assert.equal(n, kind === 'noise' ? 2 : 4);
      for (const stop of source.sliders[kind])
        for (const id of Object.values(stop.cases)) assert(data.cases[id], `Dead ${sid}/${kind} slider case ${id}`);
      const card = descendants(elements.app).find(item => item.className === 'slider-card' &&
        descendants(item).some(child => child.attributes['aria-label'] === `${sid} ${kind === 'phase' ? 'Phase φ (degrees)' : kind === 'amp' ? 'Amplitude A' : 'Thermal noise a'} stop`));
      assert(card);
      assert.equal(descendants(card).filter(item => item.attributes['data-bound-role']).length, n);
      for (const marker of data.boundaries[sid][kind]) {
        const values = source.sliders[kind].map(stop => stop.value);
        assert(marker.value >= Math.min(...values) && marker.value <= Math.max(...values));
        if (kind === 'noise') assert(Math.abs(marker.r - marker.value / source.snr0) < 1e-12);
      }
    }
  }
  const sid = '0744-064', kind = 'phase', source = data.sources[sid];
  const card = descendants(elements.app).find(item => item.className === 'slider-card' &&
    descendants(item).some(child => child.attributes['aria-label'] === `${sid} Phase φ (degrees) stop`));
  assert(card);
  const slider = descendants(card).find(item => item.tagName === 'INPUT');
  const images = descendants(card).find(item => item.className === 'images');
  const antennaButtons = descendants(card).filter(item => item.tagName === 'BUTTON').slice(0, 2);
  for (const [index, antenna] of source.antennas.entries()) {
    antennaButtons[index].onclick(); await flush();
    const available = source.sliders[kind].filter(stop => {
      const id = stop.cases[`${antenna}:0`]; return id && data.cases[id];
    });
    assert.equal(Number(slider.max) + 1, available.length);
    const phase75 = available.findIndex(stop => stop.value === 75);
    if (phase75 < 0) continue;
    slider.value = String(phase75); slider.oninput();
    assert.equal(images.children.length, 6, 'image row collapsed before replacement files loaded');
    await flush();
    assert.equal(images.children.length, 6);
    assert(images.children.every(figure => figure.children[0].tagName === 'IMG'));
    assert(descendants(card).some(item => item.textContent.includes('φ = 75°')));
  }
  console.log('PASS: source roles, interpolated bounds, sparse phase selection, stable six-image row');
})().catch(error => { console.error(error); process.exitCode = 1; });
