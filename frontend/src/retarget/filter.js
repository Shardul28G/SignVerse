// filter.js — OneEuro filter applied per (landmark, axis).

class OneEuro {
  constructor({ minCutoff = 1.5, beta = 0.05, dCutoff = 1.0 } = {}) {
    Object.assign(this, { minCutoff, beta, dCutoff });
    this.xPrev = null; this.dxPrev = 0; this.tPrev = null;
  }
  _alpha(cutoff, dt) {
    const r = 2 * Math.PI * cutoff * dt;
    return r / (r + 1);
  }
  filter(x, t) {
    if (this.xPrev == null) { this.xPrev = x; this.tPrev = t; return x; }
    const dt = Math.max(1e-3, t - this.tPrev);
    const dx = (x - this.xPrev) / dt;
    const aD = this._alpha(this.dCutoff, dt);
    const dxHat = aD * dx + (1 - aD) * this.dxPrev;
    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const a = this._alpha(cutoff, dt);
    const xHat = a * x + (1 - a) * this.xPrev;
    this.xPrev = xHat; this.dxPrev = dxHat; this.tPrev = t;
    return xHat;
  }
}

export function makeFilterArray(n, opts) {
  return Array.from({ length: n }, () => ({
    x: new OneEuro(opts), y: new OneEuro(opts), z: new OneEuro(opts),
  }));
}

// Filter an array of {x,y,z,visibility} landmarks. Low-confidence points
// are passed through unfiltered (the filter input is held).
export function filterLandmarks(landmarks, filters, t, visThresh = 0.3) {
  if (!landmarks || !filters) return landmarks;
  const out = new Array(landmarks.length);
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    if (!lm) { out[i] = lm; continue; }
    const f = filters[i];
    if (!f) { out[i] = lm; continue; }
    if ((lm.visibility ?? 1) < visThresh) { out[i] = lm; continue; }
    out[i] = {
      x: f.x.filter(lm.x, t),
      y: f.y.filter(lm.y, t),
      z: f.z.filter(lm.z, t),
      visibility: lm.visibility,
    };
  }
  return out;
}
