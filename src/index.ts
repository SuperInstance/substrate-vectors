/**
 * substrate-vectors: 1024-d vector ops
 */

export class Vector {
  readonly data: number[];
  readonly dim: number;
  
  constructor(data: number[] | Float32Array) {
    if (data instanceof Float32Array) {
      this.data = Array.from(data);
    } else {
      this.data = data;
    }
    this.dim = this.data.length;
  }
  
  cosine(other: Vector): number {
    let dot = 0, magA = 0, magB = 0;
    const n = Math.min(this.dim, other.dim);
    for (let i = 0; i < n; i++) {
      dot += this.data[i] * other.data[i];
      magA += this.data[i] * this.data[i];
      magB += other.data[i] * other.data[i];
    }
    return dot / (Math.sqrt(magA) * Math.sqrt(magB) + 1e-10);
  }
  
  dot(other: Vector): number {
    let sum = 0;
    const n = Math.min(this.dim, other.dim);
    for (let i = 0; i < n; i++) sum += this.data[i] * other.data[i];
    return sum;
  }
  
  euclidean(other: Vector): number {
    let sum = 0;
    const n = Math.min(this.dim, other.dim);
    for (let i = 0; i < n; i++) {
      const d = this.data[i] - other.data[i];
      sum += d * d;
    }
    return Math.sqrt(sum);
  }
  
  normalize(): Vector {
    let mag = 0;
    for (const x of this.data) mag += x * x;
    mag = Math.sqrt(mag) + 1e-10;
    return new Vector(this.data.map(x => x / mag));
  }
  
  add(other: Vector): Vector {
    const n = Math.max(this.dim, other.dim);
    const result = new Array(n).fill(0);
    for (let i = 0; i < this.dim; i++) result[i] += this.data[i];
    for (let i = 0; i < other.dim; i++) result[i] += other.data[i];
    return new Vector(result);
  }
  
  scale(s: number): Vector {
    return new Vector(this.data.map(x => x * s));
  }
  
  // Hash-based text embedding (1024-d)
  static fromText(text: string, dim = 1024): Vector {
    const data = new Array(dim).fill(0);
    for (let i = 0; i < text.length; i++) {
      data[i % dim] = ((data[i % dim] << 5) - data[i % dim] + text.charCodeAt(i)) & 0xffff;
    }
    let mag = 0;
    for (const x of data) mag += x * x;
    mag = Math.sqrt(mag) + 1e-10;
    return new Vector(data.map(x => x / mag));
  }
  
  static centroid(vectors: Vector[]): Vector {
    if (vectors.length === 0) return new Vector([0]);
    const dim = vectors[0].dim;
    const result = new Array(dim).fill(0);
    for (const v of vectors) {
      for (let i = 0; i < dim; i++) result[i] += v.data[i];
    }
    return new Vector(result.map(x => x / vectors.length));
  }
  
  // K-means clustering
  static clusterKMeans(vectors: Vector[], k: number, maxIter = 10): { centers: Vector[]; assignments: number[] } {
    if (vectors.length === 0) return { centers: [], assignments: [] };
    const dim = vectors[0].dim;
    
    // Init centers from first k
    const centers = vectors.slice(0, k).map(v => new Vector([...v.data]));
    const assignments = new Array(vectors.length).fill(0);
    
    for (let iter = 0; iter < maxIter; iter++) {
      // Assign
      let changed = false;
      for (let i = 0; i < vectors.length; i++) {
        let best = 0, bestD = Infinity;
        for (let j = 0; j < k; j++) {
          const d = vectors[i].euclidean(centers[j]);
          if (d < bestD) { bestD = d; best = j; }
        }
        if (assignments[i] !== best) {
          assignments[i] = best;
          changed = true;
        }
      }
      
      // Recompute centers
      for (let j = 0; j < k; j++) {
        const cluster = vectors.filter((_, i) => assignments[i] === j);
        if (cluster.length > 0) centers[j] = Vector.centroid(cluster);
      }
      
      if (!changed) break;
    }
    
    return { centers, assignments };
  }
}
