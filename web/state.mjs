// CapBody Counter review state. Copyright (c) 2026 CapBody Counter contributors.
export const COLORS = {accepted: '#16845a', linked: '#17978c', review: '#ce8a17', rejected: '#8a9690'};
export const copy = value => structuredClone(value);

export function summary(result) {
  if (!result) return {automatic: 0, pending: 0, current: 0, final: false};
  const pending = result.objects.filter(o => o.status === 'review').length;
  return {automatic: result.summary.automatic_count, pending,
    current: result.objects.filter(o => o.status === 'accepted').length,
    final: pending === 0};
}

export function editResult(original, action) {
  const result = copy(original);
  const find = id => {
    const obj = result.objects.find(o => o.id === id);
    if (!obj) throw new Error('Select an object first.');
    return obj;
  };
  const detach = obj => {
    const cap = obj.kind === 'cap' ? obj : result.objects.find(o => o.id === obj.partner_id);
    if (!cap || !cap.partner_id) return;
    const body = find(cap.partner_id);
    body.partner_id = null;
    cap.partner_id = null;
    cap.status = 'review';
    cap.reason = 'manual_unlink';
    result.links = result.links.filter(l => l.cap_id !== cap.id);
  };
  if (action.type === 'add') {
    let index = 1;
    while (result.objects.some(o => o.id === `m${String(index).padStart(3, '0')}`)) index++;
    result.objects.push({id: `m${String(index).padStart(3, '0')}`, kind: 'body', box: action.box,
      score: 1, status: 'accepted', partner_id: null, source: 'manual', reason: 'manual_add'});
  } else {
    const obj = find(action.id);
    if (action.type === 'accept') {
      if (obj.status === 'linked') throw new Error('This cap already belongs to a counted body.');
      obj.status = 'accepted'; obj.reason = 'manual_confirm';
    } else if (action.type === 'reject') {
      detach(obj); obj.status = 'rejected'; obj.reason = 'manual_reject';
    } else if (action.type === 'restore') {
      obj.status = 'review'; obj.reason = 'manual_restore';
    } else if (action.type === 'unlink') {
      detach(obj);
    } else if (action.type === 'link') {
      const body = find(action.body_id);
      if (obj.kind !== 'cap' || body.kind !== 'body' || body.status === 'rejected')
        throw new Error('Choose a cap and a valid bottle body.');
      if (body.partner_id && body.partner_id !== obj.id) throw new Error('That body is already paired with another cap.');
      detach(obj);
      obj.partner_id = body.id; obj.status = 'linked'; obj.reason = 'manual_link';
      body.partner_id = obj.id; body.status = 'accepted'; body.reason = 'manual_confirm';
      result.links.push({body_id: body.id, cap_id: obj.id, cost: null, source: 'manual'});
    } else throw new Error('Unknown review action.');
  }
  return result;
}

export function standardResult(result) {
  return {application: 'CapBody Counter', schema_version: 1, mode: 'standard',
    width: result.width, height: result.height, elapsed_seconds: result.elapsed_seconds,
    model_source: result.model_source, baseline: {prompt: result.target, count: result.count},
    summary: {automatic_count: result.count, pending_review: 0, matched_pairs: 0}, links: [],
    objects: result.detections.map((d, i) => ({...d, id: `b${String(i+1).padStart(3, '0')}`,
      kind: 'body', status: 'accepted', partner_id: null, source: 'model', reason: 'body_detection'}))};
}

export function csv(result) {
  const rows = [['id', 'evidence', 'source', 'status', 'x1', 'y1', 'x2', 'y2', 'score', 'paired_cap', 'report_status']];
  for (const o of result.objects.filter(o => o.status === 'accepted')) {
    rows.push([o.id, o.kind === 'cap' ? 'cap_only_confirmed' : o.partner_id ? 'body_and_cap' : 'body',
      o.source, o.status, ...o.box, o.score, o.partner_id || '', summary(result).final ? 'reviewed' : 'provisional']);
  }
  return rows.map(row => row.map(v => `"${String(v).replaceAll('"', '""')}"`).join(',')).join('\r\n');
}

export function validateProject(project) {
  if (!project || project.application !== 'CapBody Counter' || project.version !== 1)
    throw new Error('This is not a supported CapBody project file.');
  const image = project.image;
  if (!image || !/^data:image\/(png|jpeg|webp);base64,[A-Za-z0-9+/=]+$/.test(image.data_url)
      || image.data_url.length > 35_000_000 || !Number.isInteger(image.width) || !Number.isInteger(image.height)
      || image.width <= 0 || image.height <= 0 || image.width*image.height > 25_000_000
      || typeof image.name !== 'string' || image.name.length > 500)
    throw new Error('Project image is invalid or too large.');
  const validBox = b => Array.isArray(b) && b.length === 4 && b.every(Number.isFinite)
    && 0 <= b[0] && b[0] < b[2] && b[2] <= image.width
    && 0 <= b[1] && b[1] < b[3] && b[3] <= image.height;
  for (const group of [project.cap_exemplars, project.body_exemplars]) {
    if (!Array.isArray(group) || group.length > 3 || !group.every(validBox))
      throw new Error('Project example coordinates are invalid.');
  }
  if (!['cap_body', 'standard'].includes(project.mode)) throw new Error('Invalid project mode.');
  if (project.result !== null) {
    const r = project.result;
    if (!r || r.width !== image.width || r.height !== image.height || r.mode !== project.mode
        || !Array.isArray(r.objects) || r.objects.length > 2000 || !Array.isArray(r.links)
        || !r.summary || !Number.isInteger(r.summary.automatic_count) || r.summary.automatic_count < 0
        || r.summary.automatic_count > 2000) throw new Error('Project results are invalid.');
    const ids = new Map();
    for (const o of r.objects) {
      if (!o || typeof o.id !== 'string' || !/^[bcm][0-9]{3,6}$/.test(o.id) || ids.has(o.id)
          || !['cap', 'body'].includes(o.kind) || !Object.hasOwn(COLORS, o.status)
          || !['model', 'manual'].includes(o.source)
          || !validBox(o.box) || !Number.isFinite(o.score) || o.score < 0 || o.score > 1
          || (o.status === 'linked' && o.kind !== 'cap')) throw new Error('Project contains an invalid object.');
      ids.set(o.id, o);
    }
    const linked = new Set();
    for (const l of r.links) {
      const b = ids.get(l.body_id), c = ids.get(l.cap_id);
      if (!b || !c || b.kind !== 'body' || c.kind !== 'cap' || b.status !== 'accepted'
          || c.status !== 'linked' || b.partner_id !== c.id || c.partner_id !== b.id
          || linked.has(b.id) || linked.has(c.id)) throw new Error('Project cap–body links are inconsistent.');
      linked.add(b.id); linked.add(c.id);
    }
    for (const o of r.objects) {
      if ((o.partner_id != null || o.status === 'linked') && !linked.has(o.id))
        throw new Error('Project has a missing cap–body link.');
    }
  }
  return project;
}
