// coords.js
import * as THREE from 'three';

// MediaPipe world: +X = subject's left, +Y = down, +Z = behind subject (away from camera).
// Three / VRM    : +X = avatar's right, +Y = up, +Z = toward viewer.
// We want non-mirrored mapping (subject's right hand drives avatar's right hand).
// Subject's right (MP -X) must become avatar's right (Three +X)  -> flip X.
// Y down -> Y up  -> flip Y.
// Z behind subject -> Z away from viewer (avatar's back)  -> flip Z.
export const MP_TO_THREE = new THREE.Matrix4().makeScale(1, -1, -1);

export function mpToThree(lm) {
  return new THREE.Vector3(lm.x, lm.y, lm.z).applyMatrix4(MP_TO_THREE);
}

// Quaternion that rotates unit vector `from` onto unit vector `to`. Stable at 180°.
export function quatFromUnitVectors(from, to) {
  const q = new THREE.Quaternion();
  const f = from.clone().normalize();
  const t = to.clone().normalize();
  const d = f.dot(t);
  if (d < -0.999999) {
    const axis = Math.abs(f.x) > 0.9
      ? new THREE.Vector3(0, 1, 0)
      : new THREE.Vector3(1, 0, 0);
    axis.cross(f).normalize();
    q.setFromAxisAngle(axis, Math.PI);
  } else {
    q.setFromUnitVectors(f, t);
  }
  return q;
}

// Build an orthonormal basis quaternion from three orthogonal axis vectors.
export function quatFromBasis(x, y, z) {
  const m = new THREE.Matrix4().makeBasis(
    x.clone().normalize(),
    y.clone().normalize(),
    z.clone().normalize(),
  );
  return new THREE.Quaternion().setFromRotationMatrix(m);
}

// Build a basis from a forward direction and an "up" hint.
export function basisFromDirAndUp(dir, upHint) {
  const z = dir.clone().normalize();
  const x = new THREE.Vector3().crossVectors(upHint, z).normalize();
  const y = new THREE.Vector3().crossVectors(z, x);
  return quatFromBasis(x, y, z);
}

// Swing-twist split.
export function swingTwist(q, twistAxis) {
  const r = new THREE.Vector3(q.x, q.y, q.z);
  const p = twistAxis.clone().multiplyScalar(r.dot(twistAxis));
  const twist = new THREE.Quaternion(p.x, p.y, p.z, q.w).normalize();
  const swing = q.clone().multiply(twist.clone().invert());
  return { swing, twist };
}

export function clampQuatAngle(q, maxAngle) {
  const w = THREE.MathUtils.clamp(q.w, -1, 1);
  const angle = 2 * Math.acos(w);
  if (angle <= maxAngle) return q;
  const s = Math.sin(angle / 2);
  if (s < 1e-6) return q;
  const axis = new THREE.Vector3(q.x / s, q.y / s, q.z / s);
  return new THREE.Quaternion().setFromAxisAngle(axis, maxAngle);
}
