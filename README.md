# substrate-vectors

1024-dimensional vector operations. The math underneath Quilt's semantic search, similarity, clustering, and JEV scoring.

```typescript
import { Vector, kMeans, topK, buildProductQuantizer } from 'substrate-vectors';

const a = Vector.fromText('cell-witness');
const b = Vector.fromText('cell-bind');
a.cosine(b);                                          // similarity in [-1, 1]
Vector.kMeans(allVectors, 5);                         // k-means cluster
topK(query, db, 10, 'cosine');                        // top-10 nearest
const pq = buildProductQuantizer(db, 8, 256);         // 32x compression
```

## What's in here

| Operation | Math | Notes |
|-----------|------|-------|
| `dot(a, b)` | Σ aᵢ bᵢ | O(d). Inner product. |
| `cosine(a, b)` | (a · b) / (‖a‖ ‖b‖) | O(d). Range [-1, 1]. For unit vectors = dot product. |
| `euclidean(a, b)` | √(Σ (aᵢ - bᵢ)²) | O(d). Same units as space. |
| `manhattan(a, b)` | Σ \|aᵢ - bᵢ\| | O(d). L1. |
| `chebyshev(a, b)` | max \|aᵢ - bᵢ\| | O(d). L∞. |
| `normalize(v)` | v / ‖v‖ | Returns zero vector if norm ≈ 0. |
| `add / sub / scale / mul / lerp` | elementwise | |
| `centroid(vs)` | arithmetic mean | |
| `geometricMedian(vs)` | Weiszfeld iteration | Cosine-friendly centroid. |
| `kMeans(points, k)` | Lloyd's algorithm + k-means++ | O(d·n·k·iter). |
| `silhouette(points, assignments)` | (b − a) / max(a, b) | O(n²). Higher = better separation. |
| `topK(query, db, k, metric)` | sort by distance | |
| `cosineMatrix(vs)` | n×n symmetric | Returns Float32Array. |
| `quantize(v)` | [-1, 1] → [0, 255] | 4× storage reduction. |
| `buildProductQuantizer(vs, m, k)` | k-means per sub-vector | Compress + ANN-friendly. |

## The math (in plain English)

### Cosine similarity

For two vectors **a** and **b**:

```
cos(θ) = (a · b) / (‖a‖ ‖b‖)
```

- a · b = Σᵢ aᵢbᵢ (the inner product)
- ‖a‖ = √(Σᵢ aᵢ²) (the L2 norm)
- θ = angle between the vectors

Returns 1 if they point the same way, -1 if opposite, 0 if orthogonal. We use a tiny epsilon (1e-10) in the denominator to avoid divide-by-zero when one vector is the zero vector.

**When to use cosine**: when magnitude doesn't matter, only direction. Text embeddings, normalized user vectors, normalized embeddings.

### Euclidean distance

```
d(a, b) = √(Σ (aᵢ − bᵢ)²)
```

Same units as the underlying space. For two unit vectors in d dimensions:

```
‖a − b‖² = ‖a‖² + ‖b‖² − 2(a · b) = 2 − 2cos(θ)
```

So **euclidean and cosine are equivalent up to a monotone transform for unit vectors**. For arbitrary vectors, prefer cosine when you only care about direction; euclidean when magnitude matters.

### K-means

Lloyd's algorithm. Two steps, iterated:

1. **Assignment**: each point goes to the cluster with the nearest centroid (Euclidean).
2. **Update**: each centroid is recomputed as the mean of its assigned points.

Converges in O(d · n · k · iter) time. We use **k-means++ initialization** (Arthur & Vassilvitskii 2007) — first centroid random, each subsequent centroid chosen with probability proportional to D(x)² (squared distance to nearest existing centroid). This gives O(log k)-competitive approximation vs. O(1) for random init.

### Silhouette score

For each point x with cluster assignment c(x):

```
a(x) = mean distance to other points in cluster c(x)
b(x) = min mean distance to points in any other cluster
silhouette(x) = (b(x) − a(x)) / max(a(x), b(x))
```

Range [-1, 1]. +1 = point is far from neighboring clusters. 0 = on the boundary. -1 = probably in the wrong cluster. The dataset score is the mean over all points.

### Product quantization (PQ)

Compress a vector by splitting it into **m** sub-vectors, then running **k-means with k centroids** on each sub-vector space. The vector is now represented by **m bytes** (one centroid index per sub-vector) instead of **m · 4 · d/m = 4d bytes**.

Reconstruct by looking up each code in its sub-vector's codebook. Search by computing distance to all codebook centroids per sub-vector (precomputed distance tables make this O(d) per query).

Trade: ~32× compression with ~5–10% recall loss vs. raw vectors. Used in production vector DBs (FAISS, ScaNN).

## Why FNV-1a for `fromText`?

`Vector.fromText` is a *deterministic* but *not semantic* embedder. Same text → same vector. Different text → different vector (with high probability). It uses FNV-1a 64-bit hashed over 4 sub-seeds, each expanded to 256 dimensions via an LCG-like stream.

**Use it for**: testing, deterministic hashing, "embeddings" you can compute in 5 lines with no model.
**Don't use it for**: real semantic search. For that, use `substrate-embedding` (BGE-compatible).

## API stability

The math operations (dot, cosine, euclidean, normalize, add, sub, scale) are stable. Centroid, topK, kMeans, silhouette, cosineMatrix are stable. Quantize/dequantize are stable. Product quantization may evolve — the PQ interface is `m × k × d` and the codes are `Uint8Array(m)`.

## License

MIT.
