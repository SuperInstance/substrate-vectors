/**
 * Tests for substrate-vectors
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  Vector, centroid, geometricMedian, topK, kMeans, silhouette,
  cosineMatrix, quantize, dequantize, buildProductQuantizer,
} from '../index.ts';

test('Vector: dot product', () => {
  const a = Vector.of(1, 2, 3);
  const b = Vector.of(4, 5, 6);
  assert.equal(a.dot(b), 32); // 1*4 + 2*5 + 3*6 = 32
});

test('Vector: cosine similarity', () => {
  const a = Vector.of(1, 0, 0);
  const b = Vector.of(1, 0, 0);
  assert.ok(Math.abs(a.cosine(b) - 1.0) < 1e-6);
  const c = Vector.of(0, 1, 0);
  assert.ok(Math.abs(a.cosine(c) - 0.0) < 1e-6);
});

test('Vector: cosine of orthogonal is ~0', () => {
  const a = Vector.of(1, 0, 0, 0);
  const b = Vector.of(0, 0, 1, 0);
  assert.ok(Math.abs(a.cosine(b)) < 1e-6);
});

test('Vector: cosine of antiparallel is ~-1', () => {
  const a = Vector.of(1, 2, 3);
  const b = Vector.of(-1, -2, -3);
  assert.ok(Math.abs(a.cosine(b) - (-1)) < 1e-6);
});

test('Vector: euclidean distance', () => {
  const a = Vector.of(0, 0);
  const b = Vector.of(3, 4);
  assert.equal(a.euclidean(b), 5); // 3-4-5 triangle
});

test('Vector: euclidean squared is faster and ordered', () => {
  const a = Vector.of(0, 0);
  const b = Vector.of(3, 4);
  const c = Vector.of(1, 1);
  assert.equal(a.euclideanSq(b), 25);
  assert.ok(a.euclideanSq(c) < a.euclideanSq(b));
});

test('Vector: manhattan distance', () => {
  const a = Vector.of(0, 0);
  const b = Vector.of(3, 4);
  assert.equal(a.manhattan(b), 7);
});

test('Vector: chebyshev distance', () => {
  const a = Vector.of(0, 0);
  const b = Vector.of(3, 4);
  assert.equal(a.chebyshev(b), 4); // max(|3|, |4|)
});

test('Vector: norm', () => {
  const a = Vector.of(3, 4);
  assert.equal(a.norm(), 5);
});

test('Vector: normalize gives unit length', () => {
  const a = Vector.of(3, 4);
  const u = a.normalize();
  assert.ok(Math.abs(u.norm() - 1.0) < 1e-6);
});

test('Vector: normalize of zero is zero', () => {
  const a = Vector.zeros(3);
  const u = a.normalize();
  assert.equal(u.norm(), 0);
});

test('Vector: add / sub / scale', () => {
  const a = Vector.of(1, 2, 3);
  const b = Vector.of(4, 5, 6);
  assert.deepEqual(a.add(b).toArray(), [5, 7, 9]);
  assert.deepEqual(b.sub(a).toArray(), [3, 3, 3]);
  assert.deepEqual(a.scale(2).toArray(), [2, 4, 6]);
});

test('Vector: lerp at t=0 returns a, t=1 returns b', () => {
  const a = Vector.of(0, 0, 0);
  const b = Vector.of(10, 20, 30);
  assert.deepEqual(a.lerp(b, 0).toArray(), [0, 0, 0]);
  assert.deepEqual(a.lerp(b, 1).toArray(), [10, 20, 30]);
  assert.deepEqual(a.lerp(b, 0.5).toArray(), [5, 10, 15]);
});

test('Vector: mul is Hadamard product', () => {
  const a = Vector.of(1, 2, 3);
  const b = Vector.of(4, 5, 6);
  assert.deepEqual(a.mul(b).toArray(), [4, 10, 18]);
});

test('Vector: fromText is deterministic', () => {
  const a = Vector.fromText('hello world');
  const b = Vector.fromText('hello world');
  assert.equal(a.dim, 1024);
  assert.ok(Math.abs(a.cosine(b) - 1.0) < 1e-4);
});

test('Vector: fromText different texts are different', () => {
  const a = Vector.fromText('hello world');
  const b = Vector.fromText('goodbye world');
  assert.ok(Math.abs(a.cosine(b)) < 0.5);
});

test('Vector: random is unit length', () => {
  const r = Vector.random(128);
  assert.ok(Math.abs(r.norm() - 1.0) < 1e-5);
});

test('Vector: zeros is all zero', () => {
  const z = Vector.zeros(5);
  for (let i = 0; i < 5; i++) assert.equal(z.get(i), 0);
});

test('Vector: base64 roundtrip', () => {
  const a = Vector.of(1.5, 2.5, 3.5, 4.5);
  const b = Vector.fromBase64(a.toBase64());
  assert.equal(b.dim, 4);
  for (let i = 0; i < 4; i++) assert.ok(Math.abs(a.get(i) - b.get(i)) < 1e-6);
});

test('centroid is arithmetic mean', () => {
  const vs = [Vector.of(1, 0, 0), Vector.of(0, 1, 0), Vector.of(0, 0, 1)];
  const c = centroid(vs);
  const expected = [1/3, 1/3, 1/3];
  for (let i = 0; i < 3; i++) assert.ok(Math.abs(c.get(i) - expected[i]) < 1e-6);
});

test('topK returns k nearest in order', () => {
  const q = Vector.of(0, 0, 0);
  const cs = [Vector.of(10, 10, 10), Vector.of(1, 1, 1), Vector.of(5, 5, 5), Vector.of(2, 2, 2)];
  const r = topK(q, cs, 2, 'euclidean');
  assert.equal(r[0].index, 1); // (1,1,1) closest
  assert.equal(r[1].index, 3); // (2,2,2) next
});

test('kMeans clusters well-separated points', () => {
  // Three tight clusters
  const pts: Vector[] = [];
  for (let i = 0; i < 10; i++) pts.push(Vector.of(i * 0.01, 0));
  for (let i = 0; i < 10; i++) pts.push(Vector.of(10 + i * 0.01, 0));
  for (let i = 0; i < 10; i++) pts.push(Vector.of(20 + i * 0.01, 0));
  const { assignments, centroids } = kMeans(pts, 3, 100);
  assert.equal(centroids.length, 3);
  // Points 0-9 should be in same cluster, 10-19 in another, 20-29 in another
  const c0 = assignments[0];
  for (let i = 1; i < 10; i++) assert.equal(assignments[i], c0);
  const c1 = assignments[10];
  for (let i = 11; i < 20; i++) assert.equal(assignments[i], c1);
  const c2 = assignments[20];
  for (let i = 21; i < 30; i++) assert.equal(assignments[i], c2);
  assert.notEqual(c0, c1);
  assert.notEqual(c1, c2);
});

test('silhouette of tight clusters is high', () => {
  const pts: Vector[] = [];
  for (let i = 0; i < 20; i++) pts.push(Vector.of(Math.random() * 0.1, 0));
  for (let i = 0; i < 20; i++) pts.push(Vector.of(10 + Math.random() * 0.1, 0));
  const assignments = pts.map((_, i) => i < 20 ? 0 : 1);
  const s = silhouette(pts, assignments);
  assert.ok(s > 0.9, `expected high silhouette, got ${s}`);
});

test('cosineMatrix is symmetric', () => {
  const vs = [Vector.of(1, 2, 3), Vector.of(4, 5, 6), Vector.of(7, 8, 9)];
  const m = cosineMatrix(vs);
  const n = vs.length;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      assert.ok(Math.abs(m[i * n + j] - m[j * n + i]) < 1e-6);
    }
  }
});

test('quantize roundtrip preserves approximate values', () => {
  const v = Vector.of(0.5, -0.5, 0, 0.999, -0.999);
  const q = quantize(v);
  const d = dequantize(q);
  for (let i = 0; i < v.dim; i++) assert.ok(Math.abs(v.get(i) - d.get(i)) < 0.02);
});

test('product quantization: build + encode + decode roundtrip', () => {
  const dim = 32;
  const vectors: Vector[] = [];
  for (let i = 0; i < 50; i++) {
    const data = new Float32Array(dim);
    for (let j = 0; j < dim; j++) data[j] = Math.random();
    vectors.push(new Vector(data));
  }
  const pq = buildProductQuantizer(vectors, 4, 8);
  assert.equal(pq.m, 4);
  assert.equal(pq.k, 8);
  assert.equal(pq.subDim, 8);
  for (const v of vectors.slice(0, 5)) {
    const codes = pq.encode(v);
    assert.equal(codes.length, 4);
    const reconstructed = pq.decode(codes);
    // Reconstructed should be close to original
    const dist = v.euclidean(reconstructed);
    assert.ok(dist < 5, `reconstruction error too high: ${dist}`);
  }
});
