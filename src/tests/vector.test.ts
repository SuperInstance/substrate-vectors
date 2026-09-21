import { test } from 'node:test';
import assert from 'node:assert';
import { Vector, centroid, kMeans } from '../index.ts';

test('cosine 1.0 for self', () => {
  const v = new Vector([1, 0, 0]);
  assert.ok(Math.abs(v.cosine(v) - 1.0) < 1e-9, `expected ~1.0, got ${v.cosine(v)}`);
});

test('cosine 0 for orthogonal', () => {
  const a = new Vector([1, 0, 0]);
  const b = new Vector([0, 1, 0]);
  assert.ok(Math.abs(a.cosine(b)) < 0.01);
});

test('fromText is deterministic', () => {
  const a = Vector.fromText('hello world');
  const b = Vector.fromText('hello world');
  assert.ok(Math.abs(a.cosine(b) - 1.0) < 1e-9, `expected ~1.0, got ${a.cosine(b)}`);
});

test('centroid is average', () => {
  const a = new Vector([1, 0, 0]);
  const b = new Vector([0, 1, 0]);
  const c = centroid([a, b]);
  assert.ok(Math.abs(c.data[0] - 0.5) < 0.01);
  assert.ok(Math.abs(c.data[1] - 0.5) < 0.01);
});

test('clusterKMeans groups similar vectors', () => {
  const cluster1 = [new Vector([1, 0]), new Vector([1.1, 0]), new Vector([0.9, 0])];
  const cluster2 = [new Vector([0, 1]), new Vector([0, 1.1]), new Vector([0, 0.9])];
  const all = [...cluster1, ...cluster2];
  const result = kMeans(all, 2);
  assert.ok(result.assignments.length === 6, `expected 6 assignments, got ${result.assignments.length}`);
  // All cluster1 should have same assignment
  assert.strictEqual(result.assignments[0], result.assignments[1]);
  assert.strictEqual(result.assignments[1], result.assignments[2]);
  assert.strictEqual(result.assignments[3], result.assignments[4]);
  // Different cluster from cluster1
  assert.notStrictEqual(result.assignments[0], result.assignments[3]);
});

test('normalize produces unit vector', () => {
  const v = new Vector([3, 4]);
  const n = v.normalize();
  const mag = Math.sqrt(n.data[0]**2 + n.data[1]**2);
  assert.ok(Math.abs(mag - 1.0) < 0.01);
});
