// stickFigure.js — debug visualizer that draws MediaPipe landmarks as a
// stick figure (line segments + joint dots) using the same coordinate
// transform as the retarget pipeline.
import * as THREE from 'three';
import { mpToThree } from './coords.js';
import * as MP from './landmarks.js';

const POSE_BONES = [
  // torso
  [MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER],
  [MP.LEFT_SHOULDER, MP.LEFT_HIP],
  [MP.RIGHT_SHOULDER, MP.RIGHT_HIP],
  [MP.LEFT_HIP, MP.RIGHT_HIP],
  // arms
  [MP.LEFT_SHOULDER, MP.LEFT_ELBOW],
  [MP.LEFT_ELBOW,    MP.LEFT_WRIST],
  [MP.RIGHT_SHOULDER, MP.RIGHT_ELBOW],
  [MP.RIGHT_ELBOW,    MP.RIGHT_WRIST],
  // legs
  [MP.LEFT_HIP,  MP.LEFT_KNEE],
  [MP.LEFT_KNEE, MP.LEFT_ANKLE],
  [MP.RIGHT_HIP,  MP.RIGHT_KNEE],
  [MP.RIGHT_KNEE, MP.RIGHT_ANKLE],
  // head
  [MP.NOSE, MP.LEFT_EAR],
  [MP.NOSE, MP.RIGHT_EAR],
  [MP.LEFT_SHOULDER, MP.NOSE],
  [MP.RIGHT_SHOULDER, MP.NOSE],
];

const HAND_BONES = [
  // thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // middle
  [0, 9], [9, 10], [10, 11], [11, 12],
  // ring
  [0, 13], [13, 14], [14, 15], [15, 16],
  // pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // palm crossbars
  [5, 9], [9, 13], [13, 17],
];

export class StickFigure {
  constructor(opts = {}) {
    this.scale  = opts.scale  ?? 1.0;     // visual scale multiplier
    this.offset = opts.offset ?? new THREE.Vector3(1.5, 1.0, 0); // beside avatar
    this.colorPose = opts.colorPose ?? 0x1a4a8a;   // dark blue (visible on light bg)
    this.colorLeft = opts.colorLeft ?? 0xc62828;   // red    (subject's right hand in MP)
    this.colorRight = opts.colorRight ?? 0x2e7d32; // green  (subject's left hand in MP)
    this.colorFace = opts.colorFace ?? 0xff9800;   // orange
    this.dotColor = opts.dotColor ?? 0x222222;

    this.root = new THREE.Group();
    this.root.position.copy(this.offset);

    this.poseLines  = this._makeLines(POSE_BONES.length, this.colorPose);
    this.lhLines    = this._makeLines(HAND_BONES.length, this.colorLeft);
    this.rhLines    = this._makeLines(HAND_BONES.length, this.colorRight);
    this.poseDots   = this._makeDots(33, this.dotColor, 0.012);
    this.lhDots     = this._makeDots(21, this.colorLeft,  0.008);
    this.rhDots     = this._makeDots(21, this.colorRight, 0.008);
    this.faceDots   = this._makeDots(468, this.colorFace, 0.005);
  }

  _makeLines(count, color) {
    const positions = new Float32Array(count * 2 * 3);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.LineBasicMaterial({ color, linewidth: 2 });
    const lines = new THREE.LineSegments(geom, mat);
    lines.frustumCulled = false;
    this.root.add(lines);
    return { lines, positions };
  }

  _makeDots(count, color, size) {
    const positions = new Float32Array(count * 3);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({ color, size, sizeAttenuation: true });
    const points = new THREE.Points(geom, mat);
    points.frustumCulled = false;
    this.root.add(points);
    return { points, positions };
  }

  _setLineSegments(target, lm, bones, offset = null) {
    const arr = target.positions;
    let valid = 0;
    for (let i = 0; i < bones.length; i++) {
      const [a, b] = bones[i];
      const la = lm?.[a], lb = lm?.[b];
      if (!la || !lb) {
        arr.fill(0, i * 6, i * 6 + 6);
        continue;
      }
      const va = mpToThree(la).multiplyScalar(this.scale);
      const vb = mpToThree(lb).multiplyScalar(this.scale);
      if (offset) { va.add(offset); vb.add(offset); }
      arr[i * 6 + 0] = va.x; arr[i * 6 + 1] = va.y; arr[i * 6 + 2] = va.z;
      arr[i * 6 + 3] = vb.x; arr[i * 6 + 4] = vb.y; arr[i * 6 + 5] = vb.z;
      valid++;
    }
    target.lines.geometry.attributes.position.needsUpdate = true;
    target.lines.visible = valid > 0;
  }

  _setDots(target, lm, offset = null) {
    const arr = target.positions;
    if (!lm) { target.points.visible = false; return; }
    for (let i = 0; i < arr.length / 3; i++) {
      const p = lm[i];
      if (!p) { arr[i*3] = 0; arr[i*3+1] = 0; arr[i*3+2] = 0; continue; }
      const v = mpToThree(p).multiplyScalar(this.scale);
      if (offset) v.add(offset);
      arr[i*3] = v.x; arr[i*3+1] = v.y; arr[i*3+2] = v.z;
    }
    target.points.geometry.attributes.position.needsUpdate = true;
    target.points.visible = true;
  }

  update(frame) {
    if (!frame) return;
    this._setLineSegments(this.poseLines, frame.pose3D, POSE_BONES);
    this._setDots(this.poseDots, frame.pose3D);

    let lhOffset = null;
    if (frame.pose3D?.[MP.LEFT_WRIST] && frame.leftHandWorld?.[0]) {
      const poseW = mpToThree(frame.pose3D[MP.LEFT_WRIST]).multiplyScalar(this.scale);
      const handW = mpToThree(frame.leftHandWorld[0]).multiplyScalar(this.scale);
      lhOffset = poseW.sub(handW);
    }
    
    let rhOffset = null;
    if (frame.pose3D?.[MP.RIGHT_WRIST] && frame.rightHandWorld?.[0]) {
      const poseW = mpToThree(frame.pose3D[MP.RIGHT_WRIST]).multiplyScalar(this.scale);
      const handW = mpToThree(frame.rightHandWorld[0]).multiplyScalar(this.scale);
      rhOffset = poseW.sub(handW);
    }

    this._setLineSegments(this.lhLines, frame.leftHandWorld,  HAND_BONES, lhOffset);
    this._setLineSegments(this.rhLines, frame.rightHandWorld, HAND_BONES, rhOffset);
    this._setDots(this.lhDots, frame.leftHandWorld, lhOffset);
    this._setDots(this.rhDots, frame.rightHandWorld, rhOffset);
    
    // Draw face landmarks. We apply a slight upward offset so it floats above the hands
    // since the landmarks are usually normalized coordinates (0 to 1).
    this._setDots(this.faceDots, frame.faceLandmarks, new THREE.Vector3(-0.5, 1.5, 0));
  }

  dispose() {
    this.root.traverse((o) => {
      o.geometry?.dispose?.();
      o.material?.dispose?.();
    });
  }
}
