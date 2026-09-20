# substrate-vectors

1024-d vector ops, BGE-Large compatible.

```typescript
import { Vector } from 'substrate-vectors';

const a = Vector.fromText('cell-witness');
const b = Vector.fromText('cell-bind');
a.cosine(b);  // similarity
Vector.clusterKMeans(allVectors, 5);  // k-means
```

Cosine, dot, euclidean, normalize, add, scale, centroid, k-means. Hash-based text embedding (1024-d default).
