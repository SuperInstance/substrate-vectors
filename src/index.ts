/**
 * substrate-vectors: 1024-d vector ops
 *
 * The math underneath Quilt's semantic-search, similarity, and clustering.
 * BGE-Large compatible: vectors are 1024-d float arrays.
 *
 * Operations:
 *   - dot, cosine, euclidean, manhattan, chebyshev
 *   - normalize, add, sub, scale, lerp
 *   - centroid, k-means, hierarchical cluster
 *   - top-k nearest
 *
 * Math notes (the relevant bits):
 *
 *   Cosine similarity: cos(θ) = (a · b) / (||a|| ||b||)
 *     Range: [-1, 1]. For normalized vectors this simplifies to a · b.
 *     We add 1e-10 to the denominator to avoid div-by-zero on zero vectors.
 *
 *   Euclidean distance: d(a, b) = sqrt(Σ (aᵢ - bᵢ)²)
 *     Same units as the underlying space. For normalized vectors in d-dim:
 *     ||a - b||² = ||a||² + ||b||² - 2 a·b = 2 - 2 cos(θ)
 *     So euclidean and cosine are equivalent up to monotone transform for unit vectors.
 *
 *   K-means: Lloyd's algorithm. Initialize by k-means++ for O(log k)-competitive.
 *     Iterate: assign each point to nearest centroid, recompute centroids as means.
 *     Converges in O(d · n · k · iter) time. We cap iterations.
 *
 *   Centroid: arithmetic mean. For cosine distance, the centroid is
 *     NOT the same as the point closest to all members. We provide both
 *     methods. Use geometric median for cosine.
 */

export type VectorData = number[] | Float32Array | Float64Array;

export class Vector {
  readonly data: Float32Array;
  readonly dim: number;

  constructor(data: VectorData) {
    if (data instanceof Float32Array) {
      this.data = data;
    } else if (data instanceof Float64Array) {
      this.data = new Float32Array(data);
    } else {
      this.data = new Float32Array(data);
    }
    this.dim = this.data.length;
    if (this.dim === 0) throw new Error('Vector: empty data');
  }

  /** Build a vector from a JS array */
  static of(...values: number[]): Vector {
    return new Vector(values);
  }

  /** Build a zero vector */
  static zeros(dim: number): Vector {
    return new Vector(new Float32Array(dim));
  }

  /** Build a random unit vector (Gaussian then normalized) */
  static random(dim: number, rng: () => number = Math.random): Vector {
    const data = new Float32Array(dim);
    let norm2 = 0;
    for (let i = 0; i < dim; i++) {
      // Box-Muller for Gaussian
      const u1 = rng();
      const u2 = rng();
      const z = Math.sqrt(-2 * Math.log(u1 + 1e-15)) * Math.cos(2 * Math.PI * u2);
      data[i] = z;
      norm2 += z * z;
    }
    const norm = Math.sqrt(norm2);
    for (let i = 0; i < dim; i++) data[i] /= norm;
    return new Vector(data);
  }

  /** Hash-based deterministic embedding from text (1024-d default).
   *  Uses FNV-1a 64-bit extended to a 1024-d vector via multiple seeds.
   *  Same text -> same vector. Different text -> different vector with high probability.
   *  NOT a real semantic embedder; for that, use substrate-embedding.
   *  Use this when you need deterministic, dependency-free "embeddings".
   */
  static fromText(text: string, dim: number = 1024): Vector {
    const data = new Float32Array(dim);
    // 4 seeds, 256-d each -> 1024-d total
    for (let seed = 0; seed < 4; seed++) {
      const base = seed * 256;
      let h = 0xcbf29ce484222325n ^ BigInt(seed);
      const prime = BigInt(0x100000001b3n);
      for (let i = 0; i < text.length; i++) {
        const c = text.charCodeAt(i);
        for (let bit = 0; bit < 4; bit++) {
          h = BigInt.asUintN(64, (h ^ BigInt((c >> (bit * 8)) & 0xff)) * prime);
        }
        // Also fold in position
        h = BigInt.asUintN(64, (h ^ BigInt(i)) * prime);
      }
      // Expand the 64-bit hash over 256 dimensions via LCG-like stream
      let state = h;
      for (let i = 0; i < 256; i++) {
        state = BigInt.asUintN(64, state * prime);
        const v = Number(BigInt.asIntN(32, state)) / (1 << 31);
        data[base + i] = v;
      }
    }
    // Normalize so cosine is meaningful
    let norm2 = 0;
    for (let i = 0; i < dim; i++) norm2 += data[i] * data[i];
    const norm = Math.sqrt(norm2);
    if (norm > 0) for (let i = 0; i < dim; i++) data[i] /= norm;
    return new Vector(data);
  }

  /** Index into a vector */
  get(i: number): number { return this.data[i]; }

  /** Inner product: a · b = Σ aᵢ bᵢ */
  dot(other: Vector): number {
    const n = Math.min(this.dim, other.dim);
    let sum = 0;
    for (let i = 0; i < n; i++) sum += this.data[i] * other.data[i];
    return sum;
  }

  /** Cosine similarity: cos(θ) = (a · b) / (||a|| ||b||)
   *  Returns a value in [-1, 1]. For unit vectors this equals a · b. */
  cosine(other: Vector): number {
    let dot = 0, magA = 0, magB = 0;
    const n = Math.min(this.dim, other.dim);
    for (let i = 0; i < n; i++) {
      const a = this.data[i];
      const b = other.data[i];
      dot += a * b;
      magA += a * a;
      magB += b * b;
    }
    return dot / (Math.sqrt(magA) * Math.sqrt(magB) + 1e-10);
  }

  /** Euclidean distance: d(a, b) = sqrt(Σ (aᵢ - bᵢ)²) */
  euclidean(other: Vector): number {
    const n = Math.min(this.dim, other.dim);
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const d = this.data[i] - other.data[i];
      sum += d * d;
    }
    return Math.sqrt(sum);
  }

  /** Squared euclidean — cheaper when you only need ordering */
  euclideanSq(other: Vector): number {
    const n = Math.min(this.dim, other.dim);
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const d = this.data[i] - other.data[i];
      sum += d * d;
    }
    return sum;
  }

  /** Manhattan / L1 distance: d(a, b) = Σ |aᵢ - bᵢ| */
  manhattan(other: Vector): number {
    const n = Math.min(this.dim, other.dim);
    let sum = 0;
    for (let i = 0; i < n; i++) sum += Math.abs(this.data[i] - other.data[i]);
    return sum;
  }

  /** Chebyshev / L∞ distance: max |aᵢ - bᵢ| */
  chebyshev(other: Vector): number {
    const n = Math.min(this.dim, other.dim);
    let m = 0;
    for (let i = 0; i < n; i++) {
      const d = Math.abs(this.data[i] - other.data[i]);
      if (d > m) m = d;
    }
    return m;
  }

  /** L2 norm: ||a|| = sqrt(Σ aᵢ²) */
  norm(): number {
    let s = 0;
    for (let i = 0; i < this.dim; i++) s += this.data[i] * this.data[i];
    return Math.sqrt(s);
  }

  /** L2 squared — cheaper when only comparing */
  normSq(): number {
    let s = 0;
    for (let i = 0; i < this.dim; i++) s += this.data[i] * this.data[i];
    return s;
  }

  /** Return a unit vector in the same direction.
   *  If norm is 0, returns a zero vector. */
  normalize(): Vector {
    const n = this.norm();
    if (n < 1e-10) return Vector.zeros(this.dim);
    const data = new Float32Array(this.dim);
    for (let i = 0; i < this.dim; i++) data[i] = this.data[i] / n;
    return new Vector(data);
  }

  /** Add two vectors elementwise. */
  add(other: Vector): Vector {
    const n = Math.max(this.dim, other.dim);
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      data[i] = (i < this.dim ? this.data[i] : 0) + (i < other.dim ? other.data[i] : 0);
    }
    return new Vector(data);
  }

  /** Subtract two vectors elementwise. */
  sub(other: Vector): Vector {
    const n = Math.max(this.dim, other.dim);
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      data[i] = (i < this.dim ? this.data[i] : 0) - (i < other.dim ? other.data[i] : 0);
    }
    return new Vector(data);
  }

  /** Scale by a scalar. */
  scale(s: number): Vector {
    const data = new Float32Array(this.dim);
    for (let i = 0; i < this.dim; i++) data[i] = this.data[i] * s;
    return new Vector(data);
  }

  /** Linear interpolation: a + t (b - a). t=0 returns a, t=1 returns b. */
  lerp(other: Vector, t: number): Vector {
    const n = Math.max(this.dim, other.dim);
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const a = i < this.dim ? this.data[i] : 0;
      const b = i < other.dim ? other.data[i] : 0;
      data[i] = a + t * (b - a);
    }
    return new Vector(data);
  }

  /** Elementwise multiply (Hadamard product). */
  mul(other: Vector): Vector {
    const n = Math.min(this.dim, other.dim);
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = this.data[i] * other.data[i];
    return new Vector(data);
  }

  /** Return array of values */
  toArray(): number[] { return Array.from(this.data); }

  /** Serialize as base64 (Float32Array -> bytes -> b64) */
  toBase64(): string {
    const bytes = new Uint8Array(this.data.buffer);
    let s = '';
    for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
    if (typeof btoa !== 'undefined') return btoa(s);
    return Buffer.from(this.data.buffer).toString('base64');
  }

  /** Deserialize from base64 */
  static fromBase64(b64: string, dim?: number): Vector {
    let bytes: Uint8Array;
    if (typeof atob !== 'undefined') {
      const s = atob(b64);
      bytes = new Uint8Array(s.length);
      for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
    } else {
      bytes = new Uint8Array(Buffer.from(b64, 'base64'));
    }
    const data = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    return new Vector(data);
  }
}

/** Arithmetic mean of multiple vectors. Centroid. */
export function centroid(vectors: Vector[]): Vector {
  if (vectors.length === 0) throw new Error('centroid: empty');
  const dim = vectors[0].dim;
  const data = new Float32Array(dim);
  for (const v of vectors) {
    const n = Math.min(dim, v.dim);
    for (let i = 0; i < n; i++) data[i] += v.data[i];
  }
  for (let i = 0; i < dim; i++) data[i] /= vectors.length;
  return new Vector(data);
}

/** Geometric median of multiple vectors (cosine-friendly centroid).
 *  Iterative: start at mean, move toward points weighted by 1/distance. */
export function geometricMedian(vectors: Vector[], maxIter = 50, eps = 1e-6): Vector {
  if (vectors.length === 0) throw new Error('geometricMedian: empty');
  let med = centroid(vectors);
  for (let iter = 0; iter < maxIter; iter++) {
    let num = Vector.zeros(med.dim);
    let denom = 0;
    for (const v of vectors) {
      const d = med.euclidean(v);
      if (d < 1e-10) continue;
      const w = 1 / d;
      num = num.add(v.scale(w));
      denom += w;
    }
    if (denom === 0) break;
    const next = num.scale(1 / denom);
    const shift = med.euclidean(next);
    med = next;
    if (shift < eps) break;
  }
  return med;
}

/** Find the k nearest vectors to a query. Returns indices + distances. */
export function topK(query: Vector, candidates: Vector[], k: number, metric: 'euclidean' | 'cosine' | 'manhattan' = 'cosine'): Array<{ index: number, distance: number }> {
  const dists: Array<{ index: number, distance: number }> = candidates.map((v, i) => {
    let d: number;
    switch (metric) {
      case 'euclidean': d = v.euclidean(query); break;
      case 'manhattan': d = v.manhattan(query); break;
      case 'cosine':
      default: d = 1 - v.cosine(query); break;
    }
    return { index: i, distance: d };
  });
  dists.sort((a, b) => a.distance - b.distance);
  return dists.slice(0, k);
}

/** K-means clustering with k-means++ initialization.
 *  Lloyd's algorithm: assign → recompute → repeat.
 *  Returns cluster assignments + centroids. */
export function kMeans(points: Vector[], k: number, maxIter = 100, rng: () => number = Math.random): { assignments: number[], centroids: Vector[] } {
  if (points.length === 0) throw new Error('kMeans: no points');
  if (k <= 0) throw new Error('kMeans: k must be > 0');
  if (k > points.length) k = points.length;
  const dim = points[0].dim;

  // k-means++ initialization
  const centroids: Vector[] = [];
  const firstIdx = Math.floor(rng() * points.length);
  centroids.push(points[firstIdx]);
  while (centroids.length < k) {
    const dists = points.map(p => {
      let min = Infinity;
      for (const c of centroids) {
        const d = p.euclideanSq(c);
        if (d < min) min = d;
      }
      return min;
    });
    const total = dists.reduce((s, d) => s + d, 0);
    if (total === 0) {
      centroids.push(points[Math.floor(rng() * points.length)]);
      continue;
    }
    let r = rng() * total;
    let chosen = 0;
    for (let i = 0; i < dists.length; i++) {
      r -= dists[i];
      if (r <= 0) { chosen = i; break; }
    }
    centroids.push(points[chosen]);
  }

  // Lloyd's iteration
  const assignments = new Array(points.length).fill(0);
  for (let iter = 0; iter < maxIter; iter++) {
    let changed = false;
    // Assignment step
    for (let i = 0; i < points.length; i++) {
      let best = 0;
      let bestDist = Infinity;
      for (let j = 0; j < centroids.length; j++) {
        const d = points[i].euclideanSq(centroids[j]);
        if (d < bestDist) { bestDist = d; best = j; }
      }
      if (assignments[i] !== best) {
        assignments[i] = best;
        changed = true;
      }
    }
    // Update step
    const sums: Vector[] = Array.from({ length: k }, () => Vector.zeros(dim));
    const counts = new Array(k).fill(0);
    for (let i = 0; i < points.length; i++) {
      const c = assignments[i];
      sums[c] = sums[c].add(points[i]);
      counts[c]++;
    }
    for (let j = 0; j < k; j++) {
      if (counts[j] > 0) centroids[j] = sums[j].scale(1 / counts[j]);
    }
    if (!changed) break;
  }
  return { assignments, centroids };
}

/** Compute the silhouette score for a clustering.
 *  For each point: a = mean distance to others in same cluster;
 *                  b = min mean distance to points in any other cluster;
 *                  silhouette = (b - a) / max(a, b).
 *  Range: [-1, 1]. Higher is better separation.
 *  O(n²) — slow for large n but exact. */
export function silhouette(points: Vector[], assignments: number[]): number {
  if (points.length < 2) return 0;
  const k = Math.max(...assignments) + 1;
  let total = 0;
  for (let i = 0; i < points.length; i++) {
    const ci = assignments[i];
    let aSum = 0;
    let aCount = 0;
    let bMin = Infinity;
    const bSums = new Array(k).fill(0);
    const bCounts = new Array(k).fill(0);
    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      const cj = assignments[j];
      const d = points[i].euclidean(points[j]);
      bSums[cj] += d;
      bCounts[cj]++;
      if (cj === ci) { aSum += d; aCount++; }
    }
    const a = aCount > 0 ? aSum / aCount : 0;
    for (let c = 0; c < k; c++) {
      if (c === ci || bCounts[c] === 0) continue;
      const bmean = bSums[c] / bCounts[c];
      if (bmean < bMin) bMin = bmean;
    }
    if (bMin === Infinity) bMin = 0;
    const s = a === 0 && bMin === 0 ? 0 : (bMin - a) / Math.max(a, bMin);
    total += s;
  }
  return total / points.length;
}

/** Pairwise cosine similarity matrix. Returns Float32Array of length n*n. */
export function cosineMatrix(vectors: Vector[]): Float32Array {
  const n = vectors.length;
  const out = new Float32Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const c = vectors[i].cosine(vectors[j]);
      out[i * n + j] = c;
      out[j * n + i] = c;
    }
  }
  return out;
}

/** HNSW-lite: brute-force top-k for now. (Real HNSW is in substrate-forge.) */
export function bruteForceTopK(query: Vector, db: Vector[], k: number, metric: 'cosine' | 'euclidean' = 'cosine'): Array<{ id: number, score: number }> {
  const scored = db.map((v, i) => {
    const score = metric === 'cosine' ? query.cosine(v) : -query.euclidean(v);
    return { id: i, score };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}

/** Quantize a vector to bytes for compact storage.
 *  Maps [-1, 1] to [0, 255]. Loses precision but cuts storage 4x. */
export function quantize(v: Vector): Uint8Array {
  const out = new Uint8Array(v.dim);
  for (let i = 0; i < v.dim; i++) {
    const x = Math.max(-1, Math.min(1, v.data[i]));
    out[i] = Math.round((x + 1) * 127.5);
  }
  return out;
}

/** Dequantize back to a Vector. */
export function dequantize(bytes: Uint8Array): Vector {
  const dim = bytes.length;
  const data = new Float32Array(dim);
  for (let i = 0; i < dim; i++) data[i] = (bytes[i] - 127.5) / 127.5;
  return new Vector(data);
}

/** Product quantization: split into m sub-vectors, k-means each. Compress + index. */
export interface ProductQuantizer {
  m: number;
  k: number;
  subDim: number;
  codebooks: Vector[][];  // [m][k] centroids
  encode(v: Vector): Uint8Array;
  decode(codes: Uint8Array): Vector;
  distance(query: Vector, codes: Uint8Array, metric: 'euclidean' | 'cosine'): number;
}

export function buildProductQuantizer(vectors: Vector[], m: number, k: number, rng: () => number = Math.random): ProductQuantizer {
  if (vectors.length === 0) throw new Error('buildProductQuantizer: no vectors');
  const dim = vectors[0].dim;
  if (dim % m !== 0) throw new Error(`buildProductQuantizer: dim ${dim} not divisible by m ${m}`);
  const subDim = dim / m;
  const codebooks: Vector[][] = [];
  for (let s = 0; s < m; s++) {
    const subVectors = vectors.map(v => {
      const sub = new Float32Array(subDim);
      for (let i = 0; i < subDim; i++) sub[i] = v.data[s * subDim + i];
      return new Vector(sub);
    });
    const { centroids } = kMeans(subVectors, k, 50, rng);
    codebooks.push(centroids);
  }
  function encode(v: Vector): Uint8Array {
    const codes = new Uint8Array(m);
    for (let s = 0; s < m; s++) {
      const sub = new Float32Array(subDim);
      for (let i = 0; i < subDim; i++) sub[i] = v.data[s * subDim + i];
      const sv = new Vector(sub);
      let best = 0;
      let bestDist = Infinity;
      for (let j = 0; j < k; j++) {
        const d = sv.euclideanSq(codebooks[s][j]);
        if (d < bestDist) { bestDist = d; best = j; }
      }
      codes[s] = best;
    }
    return codes;
  }
  function decode(codes: Uint8Array): Vector {
    const data = new Float32Array(dim);
    for (let s = 0; s < m; s++) {
      const c = codebooks[s][codes[s]];
      for (let i = 0; i < subDim; i++) data[s * subDim + i] = c.data[i];
    }
    return new Vector(data);
  }
  function distance(query: Vector, codes: Uint8Array, metric: 'euclidean' | 'cosine' = 'euclidean'): number {
    const reconstructed = decode(codes);
    return metric === 'cosine' ? 1 - query.cosine(reconstructed) : query.euclidean(reconstructed);
  }
  return { m, k, subDim, codebooks, encode, decode, distance };
}
