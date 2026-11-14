import assert from 'node:assert/strict';
import {csv, editResult, summary, validateProject} from '../web/state.mjs';

const base = {mode: 'cap_body', width: 100, height: 200, summary: {automatic_count: 1}, links: [],
  objects: [{id: 'b001', kind: 'body', status: 'accepted', partner_id: null, source: 'model', score: .8, box: [0, 20, 50, 190]},
    {id: 'c001', kind: 'cap', status: 'review', partner_id: null, source: 'model', score: .7, box: [10, 15, 35, 30]}]};
assert.deepEqual(summary(base), {automatic: 1, current: 1, pending: 1, final: false});
const accepted = editResult(base, {type: 'accept', id: 'c001'});
assert.equal(summary(accepted).current, 2);
assert.equal(summary(accepted).final, true);
assert.equal(summary(base).pending, 1, 'History snapshots must stay immutable');
const paired = editResult(accepted, {type: 'link', id: 'c001', body_id: 'b001'});
assert.equal(summary(paired).current, 1, 'Linking an accepted cap removes the duplicate count');
assert.equal(paired.links.length, 1);
assert.throws(() => editResult(paired, {type: 'accept', id: 'c001'}));
const rejected = editResult(paired, {type: 'reject', id: 'b001'});
assert.equal(summary(rejected).current, 0);
assert.equal(summary(rejected).pending, 1);
assert.equal(rejected.links.length, 0);
assert.equal(rejected.objects[1].partner_id, null);
const unlinked = editResult(paired, {type: 'unlink', id: 'c001'});
assert.equal(summary(unlinked).current, 1);
assert.equal(summary(unlinked).pending, 1);
const added = editResult(base, {type: 'add', box: [55, 20, 90, 190]});
assert.equal(summary(added).current, 2);
assert.equal(new Set(added.objects.map(o => o.id)).size, 3);
assert.equal(csv(base).split('\r\n').length, 2, 'CSV excludes pending proposals');
assert.match(csv(base), /provisional/);
const project = {application: 'CapBody Counter', version: 1, mode: 'cap_body',
  image: {data_url: 'data:image/png;base64,AAAA', width: 100, height: 200, name: 'test.png'},
  cap_exemplars: [[10, 15, 35, 30]], body_exemplars: [], result: paired};
assert.deepEqual(validateProject(JSON.parse(JSON.stringify(project))), project);
const invalid = structuredClone(project); invalid.result.links = [];
assert.throws(() => validateProject(invalid), /missing/);
const unsafe = structuredClone(project); unsafe.image.data_url = 'data:image/svg+xml;base64,AAAA';
assert.throws(() => validateProject(unsafe));
console.log('Review state: matching, corrections, immutable undo snapshots, CSV and project validation passed.');
