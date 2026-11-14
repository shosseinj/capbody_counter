import {COLORS, copy, csv, editResult, standardResult, summary, validateProject} from './state.mjs';

const $ = id => document.getElementById(id);
const canvas = $('canvas'), ctx = canvas.getContext('2d');
let state = {application: 'CapBody Counter', version: 1, mode: 'cap_body', image: null,
  cap_exemplars: [], body_exemplars: [], result: null};
let bitmap = null, selected = null, tool = 'cap', busy = false, space = false, gesture = null;
let camera = {x: 0, y: 0, scale: 1}, history = [], future = [];
const reasons = {unmatched_cap: 'Cap without a matched body. Confirm only if it represents a separate bottle.',
  ambiguous_match: 'More than one plausible pairing. Select a body or confirm a separate bottle.',
  short_body: 'Short body proposal; it may be a cap or a non-bottle object.',
  body_detection: 'Automatic body detection. Check that it is a real, separate bottle.',
  matched_cap: 'Paired with a body. This cap does not add another bottle.',
  manual_confirm: 'Confirmed by you.', manual_reject: 'Dismissed by you.', manual_add: 'Bottle added by you.',
  manual_restore: 'Restored for review.', manual_unlink: 'Link removed. Review this cap again.',
  manual_link: 'Paired by you; counted through the body.'};

function message(text, type = 'info') {
  $('message').textContent = text; $('message').className = `message ${type}`; $('message').hidden = !text;
}
function snapshot() { return copy({mode: state.mode, cap_exemplars: state.cap_exemplars,
  body_exemplars: state.body_exemplars, result: state.result}); }
function checkpoint() { history.push(snapshot()); if (history.length > 40) history.shift(); future = []; }
function restoreSnapshot(s) { Object.assign(state, s); selected = null; tool = state.result ? 'inspect' : 'cap'; render(); }
function undo() { if (busy || !history.length) return; future.push(snapshot()); restoreSnapshot(history.pop()); }
function redo() { if (busy || !future.length) return; history.push(snapshot()); restoreSnapshot(future.pop()); }
function action(type, extra = {}) {
  try {
    const result = editResult(state.result, {type, id: selected, ...extra});
    checkpoint(); state.result = result; render(); message('Review updated. You can undo any change.', 'success');
  } catch (error) { message(error.message); }
}
function setTool(value) { if (busy) return; tool = value; gesture = null; render(); }

function fit() {
  if (!bitmap) return;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  camera.scale = Math.min((w-32)/bitmap.width, (h-32)/bitmap.height);
  camera.x = (w-bitmap.width*camera.scale)/2; camera.y = (h-bitmap.height*camera.scale)/2; draw();
}
function zoom(factor, x = canvas.clientWidth/2, y = canvas.clientHeight/2) {
  if (!bitmap || busy) return;
  const scale = Math.max(.03, Math.min(12, camera.scale*factor));
  camera.x = x-(x-camera.x)*scale/camera.scale; camera.y = y-(y-camera.y)*scale/camera.scale;
  camera.scale = scale; draw();
}
function imagePoint(event, clamp = true) {
  const r = canvas.getBoundingClientRect();
  let x = (event.clientX-r.left-camera.x)/camera.scale, y = (event.clientY-r.top-camera.y)/camera.scale;
  if (clamp && bitmap) { x = Math.max(0, Math.min(bitmap.width, x)); y = Math.max(0, Math.min(bitmap.height, y)); }
  return [x, y];
}
function rectangle(a, b) { return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])]; }
function drawBox(context, box, color, label, width = 1.5, dashed = false, scale = camera.scale) {
  context.strokeStyle = color; context.lineWidth = width/scale;
  context.setLineDash(dashed ? [5/scale, 3/scale] : []);
  context.strokeRect(box[0], box[1], box[2]-box[0], box[3]-box[1]); context.setLineDash([]);
  if (label) {
    const size = 10/scale; context.font = `500 ${size}px system-ui`;
    const labelWidth = context.measureText(label).width+8/scale, labelHeight = 16/scale;
    const y = Math.max(0, box[1]-labelHeight);
    context.fillStyle = color; context.fillRect(box[0], y, labelWidth, labelHeight);
    context.fillStyle = 'white'; context.fillText(label, box[0]+4/scale, y+11.5/scale);
  }
}
function drawEvidence(context, result, scale = camera.scale, showSelection = true) {
  const objects = new Map(result.objects.map(o => [o.id, o]));
  context.lineWidth = 1.2/scale; context.strokeStyle = COLORS.linked; context.setLineDash([4/scale, 3/scale]);
  for (const link of result.links) {
    const body = objects.get(link.body_id)?.box, cap = objects.get(link.cap_id)?.box;
    if (!body || !cap) continue;
    context.beginPath(); context.moveTo((body[0]+body[2])/2, body[1]);
    context.lineTo((cap[0]+cap[2])/2, (cap[1]+cap[3])/2); context.stroke();
  }
  context.setLineDash([]);
  for (const obj of result.objects) {
    context.globalAlpha = obj.status === 'rejected' ? .35 : 1;
    drawBox(context, obj.box, COLORS[obj.status], obj.id, showSelection && obj.id === selected ? 3 : 1.4,
      obj.status === 'review', scale);
  }
  context.globalAlpha = 1;
}
function draw() {
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  if (!bitmap) return;
  ctx.save(); ctx.translate(camera.x, camera.y); ctx.scale(camera.scale, camera.scale);
  ctx.drawImage(bitmap, 0, 0);
  if (state.result) drawEvidence(ctx, state.result);
  for (const [role, boxes, color] of [['cap', state.cap_exemplars, '#b17c26'], ['body', state.body_exemplars, '#168a90']]) {
    boxes.forEach((box, i) => drawBox(ctx, box, color, `${role} example ${i+1}`, 2, true));
  }
  if (gesture?.type === 'draw') drawBox(ctx, rectangle(gesture.start, gesture.end), tool === 'add' ? COLORS.accepted : COLORS.review, '', 2, true);
  ctx.restore(); canvas.style.cursor = space || tool === 'pan' ? 'grab' : tool === 'inspect' ? 'pointer' : 'crosshair';
}
new ResizeObserver(() => {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(canvas.clientWidth*dpr); canvas.height = Math.round(canvas.clientHeight*dpr); fit();
}).observe($('canvas-wrap'));

canvas.addEventListener('pointerdown', event => {
  if (!bitmap || busy) return;
  if (event.button !== 0 && event.button !== 1) return;
  canvas.setPointerCapture(event.pointerId);
  if (space || event.button === 1) {
    gesture = {type: 'pan', x: event.clientX, y: event.clientY, startX: camera.x, startY: camera.y}; return;
  }
  const point = imagePoint(event, false);
  if (point[0] < 0 || point[1] < 0 || point[0] > bitmap.width || point[1] > bitmap.height) return;
  if (tool === 'inspect') {
    const hits = (state.result?.objects || []).filter(o => point[0] >= o.box[0] && point[0] <= o.box[2]
      && point[1] >= o.box[1] && point[1] <= o.box[3]).sort((a, b) =>
      (a.box[2]-a.box[0])*(a.box[3]-a.box[1])-(b.box[2]-b.box[0])*(b.box[3]-b.box[1]));
    selected = hits[0]?.id || null; render(); return;
  }
  if (tool === 'add' && !state.result) { message('Analyze the image before adding missing bottles.'); return; }
  gesture = {type: 'draw', start: point, end: point}; draw();
});
canvas.addEventListener('pointermove', event => {
  if (!gesture) return;
  if (gesture.type === 'pan') {
    camera.x = gesture.startX+event.clientX-gesture.x; camera.y = gesture.startY+event.clientY-gesture.y;
  } else gesture.end = imagePoint(event);
  draw();
});
canvas.addEventListener('pointerup', () => {
  if (!gesture) return;
  if (gesture.type === 'draw') {
    const box = rectangle(gesture.start, gesture.end).map(v => Math.round(v*100)/100);
    if (box[2]-box[0] >= 3 && box[3]-box[1] >= 3) {
      if (tool === 'add') action('add', {box});
      else {
        const key = `${tool}_exemplars`;
        if (state[key].length >= 3) message('Use at most three examples of each kind. Clear examples to start again.');
        else {
          checkpoint(); state[key].push(box); state.result = null; selected = null;
          message(`${tool === 'cap' ? 'Cap' : 'Body'} example added. Analyze when your examples are ready.`, 'success');
        }
      }
    }
  }
  gesture = null; render();
});
canvas.addEventListener('pointercancel', () => { gesture = null; draw(); });
canvas.addEventListener('wheel', event => {
  if (!bitmap) return; event.preventDefault();
  const r = canvas.getBoundingClientRect(); zoom(Math.exp(-event.deltaY*.001), event.clientX-r.left, event.clientY-r.top);
}, {passive: false});
window.addEventListener('keydown', event => {
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) return;
  if (event.code === 'Space') { event.preventDefault(); space = true; draw(); }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
    event.preventDefault(); event.shiftKey ? redo() : undo();
  }
  if (event.key === 'Escape') { gesture = null; selected = null; draw(); renderSelection(); }
});
window.addEventListener('keyup', event => { if (event.code === 'Space') { space = false; draw(); } });
window.addEventListener('blur', () => { space = false; gesture = null; draw(); });

async function loadImage(file) {
  if (busy || !file) return;
  try {
    if (file.size > 15*1024**2) throw new Error('Choose an image smaller than 15 MiB.');
    if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) throw new Error('Use a PNG, JPEG or WebP image.');
    const decoded = await createImageBitmap(file, {imageOrientation: 'from-image'});
    if (decoded.width*decoded.height > 25_000_000) { decoded.close(); throw new Error('Use an image smaller than 25 megapixels.'); }
    // Encode the displayed orientation once; uploaded pixels and example coordinates are identical.
    const normalized = document.createElement('canvas'); normalized.width = decoded.width; normalized.height = decoded.height;
    normalized.getContext('2d').drawImage(decoded, 0, 0); decoded.close();
    const blob = await new Promise(resolve => normalized.toBlob(resolve, 'image/png'));
    if (!blob || blob.size > 15*1024**2) throw new Error('The oriented image exceeds 15 MiB. Resize it before uploading.');
    const data = await new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = reject; reader.readAsDataURL(blob); });
    const next = await createImageBitmap(blob); bitmap?.close(); bitmap = next;
    state = {application: 'CapBody Counter', version: 1, mode: $('mode').value,
      image: {name: file.name, data_url: data, width: bitmap.width, height: bitmap.height},
      cap_exemplars: [], body_exemplars: [], result: null};
    selected = null; history = []; future = []; tool = state.mode === 'cap_body' ? 'cap' : 'inspect';
    render(); fit(); message(state.mode === 'cap_body' ? 'Draw one to three close-fitting cap examples, then analyze.' : 'Image ready for standard counting.', 'success');
  } catch (error) { message(error.message || 'Could not read this image.'); }
  $('image-input').value = '';
}
$('image-input').addEventListener('change', event => loadImage(event.target.files[0]));
$('drop-zone').addEventListener('keydown', event => { if (['Enter', ' '].includes(event.key)) { event.preventDefault(); $('image-input').click(); } });
for (const name of ['dragenter', 'dragover']) $('drop-zone').addEventListener(name, event => { event.preventDefault(); $('drop-zone').classList.add('dragover'); });
for (const name of ['dragleave', 'drop']) $('drop-zone').addEventListener(name, event => { event.preventDefault(); $('drop-zone').classList.remove('dragover'); });
$('drop-zone').addEventListener('drop', event => loadImage(event.dataTransfer.files[0]));

async function analyze() {
  if (!state.image || busy) return;
  busy = true; $('workspace').inert = true; render(); message('');
  try {
    const body = new FormData(); body.append('file', await (await fetch(state.image.data_url)).blob(), 'oriented.png');
    let route = '/count';
    if (state.mode === 'cap_body') {
      route = '/count/cap-body'; body.append('cap_exemplars', JSON.stringify(state.cap_exemplars));
      body.append('body_exemplars', JSON.stringify(state.body_exemplars));
      body.append('body_threshold', $('body-threshold').value); body.append('cap_threshold', $('cap-threshold').value);
    } else {
      body.append('prompt', $('prompt').value); body.append('threshold', $('body-threshold').value); body.append('include_image', 'false');
    }
    const response = await fetch(route, {method: 'POST', body});
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Check your examples and detection settings.');
    checkpoint(); state.result = state.mode === 'standard' ? standardResult(result) : result;
    selected = null; tool = 'inspect'; $('filter').value = 'review';
    const pending = summary(state.result).pending;
    message(pending ? `Analysis complete. ${pending} proposals need your review before the count is final.` : 'Analysis complete. Check the detections before using the total.', 'success');
  } catch (error) { message(error.message || 'Analysis failed. Check the local service and try again.'); }
  finally { busy = false; $('workspace').inert = false; render(); }
}

function renderList() {
  const list = $('object-list'); list.replaceChildren();
  if (!state.result) { const p = document.createElement('p'); p.className = 'list-empty'; p.textContent = 'Detections will appear here after analysis.'; list.append(p); return; }
  const filter = $('filter').value;
  const rows = state.result.objects.filter(o => filter === 'all' || o.status === filter);
  if (!rows.length) { const p = document.createElement('p'); p.className = 'list-empty'; p.textContent = filter === 'review' ? 'No pending proposals. Check counted objects or export your report.' : 'No objects in this category.'; list.append(p); }
  for (const obj of rows) {
    const button = document.createElement('button'); button.className = `object-row ${selected === obj.id ? 'selected' : ''}`; button.dataset.id = obj.id;
    const dot = document.createElement('span'); dot.className = 'status-dot'; dot.style.background = COLORS[obj.status];
    const name = document.createElement('span'); name.className = 'object-name'; name.textContent = `${obj.kind === 'cap' ? 'Cap' : obj.source === 'manual' ? 'Added bottle' : 'Body'} ${obj.id}`;
    const small = document.createElement('small'); small.textContent = obj.status === 'linked' ? `Paired with ${obj.partner_id}` : obj.status === 'review' ? 'Needs your review' : obj.status === 'accepted' ? 'Included in count' : 'Dismissed'; name.append(small);
    const score = document.createElement('span'); score.className = 'score'; score.textContent = `${Math.round(obj.score*100)}%`;
    button.append(dot, name, score); button.addEventListener('click', () => { selected = obj.id; tool = 'inspect'; render(); }); list.append(button);
  }
}
function renderSelection() {
  const obj = state.result?.objects.find(o => o.id === selected); $('selected-object').hidden = !obj;
  if (!obj) return;
  $('selected-title').textContent = `${obj.kind === 'cap' ? 'Cap' : 'Body'} ${obj.id}`;
  $('selected-score').textContent = `${Math.round(obj.score*100)}% score`;
  $('selected-reason').textContent = reasons[obj.reason] || 'Review this proposal.';
  $('confirm').hidden = obj.status !== 'review'; $('dismiss').hidden = obj.status === 'rejected';
  $('restore').hidden = obj.status !== 'rejected';
  $('pair-controls').hidden = obj.kind !== 'cap' || obj.status === 'rejected' || obj.status === 'linked';
  $('unpair').hidden = !obj.partner_id;
  const select = $('pair-body'); select.replaceChildren();
  for (const body of state.result.objects.filter(o => o.kind === 'body' && o.status !== 'rejected' && !o.partner_id)) {
    const option = document.createElement('option'); option.value = body.id; option.textContent = `Body ${body.id}`; select.append(option);
  }
  $('pair').disabled = select.options.length === 0;
}
function render() {
  const counts = summary(state.result), hasImage = !!state.image;
  $('auto-count').textContent = state.result ? counts.automatic : '—';
  $('review-count').textContent = state.result ? counts.pending : '—';
  $('current-count').textContent = state.result ? counts.current : '—';
  $('count-status').textContent = !state.result ? 'Analyze an image to begin' : counts.final ? 'Review complete · check detections' : 'Provisional · review pending';
  $('mode').value = state.mode;
  $('example-controls').hidden = state.mode !== 'cap_body'; $('standard-controls').hidden = state.mode !== 'standard';
  $('cap-example-count').textContent = `${state.cap_exemplars.length} / 3`;
  $('body-example-count').textContent = state.body_exemplars.length ? `${state.body_exemplars.length} / 3` : 'optional';
  $('empty-state').hidden = hasImage; $('busy').hidden = !busy;
  $('busy-detail').textContent = state.mode === 'cap_body' ? 'Detecting bodies and caps in two passes. Please keep this page open.' : 'Running CountGD++ on your image. Please keep this page open.';
  $('file-meta').textContent = hasImage ? `${state.image.name} · ${state.image.width} × ${state.image.height}` : 'Your image stays on this machine.';
  $('analyze').disabled = busy || !hasImage || (state.mode === 'cap_body' && !state.cap_exemplars.length);
  $('analyze').firstChild.textContent = state.result ? 'Analyze again ' : 'Analyze image ';
  $('canvas-title').textContent = !hasImage ? 'Mark examples' : tool === 'cap' ? 'Draw a cap example' : tool === 'body' ? 'Draw a body example' : tool === 'add' ? 'Draw a missing bottle' : 'Inspect the evidence';
  $('compute-note').textContent = `${state.mode === 'cap_body' ? 'Two model passes' : 'One model pass'} · local inference`;
  for (const button of document.querySelectorAll('[data-tool]')) { button.classList.toggle('active', button.dataset.tool === tool); button.disabled = !hasImage || (button.dataset.tool === 'add' && !state.result); }
  $('undo').disabled = !history.length || busy; $('redo').disabled = !future.length || busy;
  $('clear-examples').disabled = !state.cap_exemplars.length && !state.body_exemplars.length;
  $('save-project').disabled = !hasImage; $('export-png').disabled = !state.result; $('export-csv').disabled = !state.result;
  $('result-bar').hidden = !state.result;
  if (state.result) { $('baseline-label').textContent = `Raw ${state.result.baseline?.prompt || 'model'} proposals: ${state.result.baseline?.count ?? '—'}`; $('timing-label').textContent = `${state.result.elapsed_seconds}s · ${state.result.links.length} cap–body pairs`; }
  renderList(); renderSelection(); draw();
}

function download(blob, name) {
  const url = URL.createObjectURL(blob), link = document.createElement('a'); link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function project() {
  return {...copy(state), report: state.result ? summary(state.result) : null,
    settings: {body_threshold: Number($('body-threshold').value), cap_threshold: Number($('cap-threshold').value), prompt: $('prompt').value}};
}
async function openProject(file) {
  if (!file || busy) return;
  try {
    if (file.size > 36_000_000) throw new Error('Project file is too large.');
    const next = validateProject(JSON.parse(await file.text()));
    const decoded = await createImageBitmap(await (await fetch(next.image.data_url)).blob());
    if (decoded.width !== next.image.width || decoded.height !== next.image.height) { decoded.close(); throw new Error('Project image dimensions do not match.'); }
    bitmap?.close(); bitmap = decoded;
    state = {application: next.application, version: next.version, mode: next.mode, image: next.image,
      cap_exemplars: next.cap_exemplars, body_exemplars: next.body_exemplars, result: next.result};
    for (const [id, key] of [['body-threshold', 'body_threshold'], ['cap-threshold', 'cap_threshold']]) {
      const value = next.settings?.[key]; if (Number.isFinite(value) && value >= .05 && value <= .9) { $(id).value = value; $(`${id}-value`).textContent = value.toFixed(2); }
    }
    if (typeof next.settings?.prompt === 'string') $('prompt').value = next.settings.prompt.slice(0, 500);
    selected = null; history = []; future = []; tool = state.result ? 'inspect' : 'cap'; render(); fit(); message('Project restored, including your review decisions.', 'success');
  } catch (error) { message(error.message || 'Could not open this project.'); }
  $('project-input').value = '';
}
function exportPNG() {
  if (!bitmap || !state.result) return;
  const output = document.createElement('canvas'); output.width = bitmap.width; output.height = bitmap.height+40;
  const context = output.getContext('2d'); context.fillStyle = '#153e34'; context.fillRect(0, 0, output.width, 40);
  const counts = summary(state.result); context.fillStyle = 'white'; context.font = '13px system-ui';
  context.fillText(`CapBody Counter | ${counts.current} counted | ${counts.pending} pending | ${counts.final ? 'Review complete' : 'PROVISIONAL'}`, 12, 25);
  context.translate(0, 40); context.drawImage(bitmap, 0, 0); drawEvidence(context, state.result, 1, false);
  output.toBlob(blob => { if (blob) download(blob, 'capbody-result.png'); }, 'image/png');
}

document.querySelectorAll('[data-tool]').forEach(button => button.addEventListener('click', () => setTool(button.dataset.tool)));
$('mode').addEventListener('change', event => { checkpoint(); state.mode = event.target.value; state.result = null; selected = null; tool = state.mode === 'cap_body' ? 'cap' : 'inspect'; render(); });
$('clear-examples').addEventListener('click', () => { checkpoint(); state.cap_exemplars = []; state.body_exemplars = []; state.result = null; selected = null; render(); });
$('analyze').addEventListener('click', analyze); $('filter').addEventListener('change', renderList);
$('undo').addEventListener('click', undo); $('redo').addEventListener('click', redo);
$('zoom-in').addEventListener('click', () => zoom(1.25)); $('zoom-out').addEventListener('click', () => zoom(.8)); $('fit').addEventListener('click', fit);
for (const id of ['body-threshold', 'cap-threshold']) $(id).addEventListener('input', () => { $(`${id}-value`).textContent = Number($(id).value).toFixed(2); });
$('confirm').addEventListener('click', () => action('accept')); $('dismiss').addEventListener('click', () => action('reject'));
$('restore').addEventListener('click', () => action('restore')); $('unpair').addEventListener('click', () => action('unlink'));
$('pair').addEventListener('click', () => action('link', {body_id: $('pair-body').value}));
$('save-project').addEventListener('click', () => download(new Blob([JSON.stringify(project(), null, 2)], {type: 'application/json'}), 'capbody-project.json'));
$('load-project').addEventListener('click', () => $('project-input').click()); $('project-input').addEventListener('change', event => openProject(event.target.files[0]));
$('export-png').addEventListener('click', exportPNG); $('export-csv').addEventListener('click', () => download(new Blob([csv(state.result)], {type: 'text/csv;charset=utf-8'}), 'capbody-counts.csv'));
$('method-button').addEventListener('click', () => $('method-dialog').showModal()); $('close-dialog').addEventListener('click', () => $('method-dialog').close());
async function health() {
  try { const response = await fetch('/health'); const data = await response.json(); if (!response.ok || !data.ready) throw new Error();
    $('engine').className = 'engine ready'; $('engine').replaceChildren(Object.assign(document.createElement('span'), {}), document.createTextNode(`${String(data.device).toUpperCase()} engine ready`));
  } catch { $('engine').className = 'engine error'; $('engine').replaceChildren(document.createElement('span'), document.createTextNode('Engine unavailable')); }
}
render(); health(); setInterval(health, 30000);
