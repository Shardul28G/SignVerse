// calibration.js — optional T-pose calibration step.
import * as THREE from 'three';
import { mpToThree, quatFromUnitVectors } from './coords.js';
import { solveSpine } from './solvers/spineSolver.js';
import * as MP from './landmarks.js';

function averageLandmarks(frames) {
  const n = 33;
  const avg = Array.from({ length: n }, () => ({ x: 0, y: 0, z: 0, visibility: 0 }));
  let count = 0;
  for (const f of frames) {
    if (!f?.pose3D) continue;
    count++;
    for (let i = 0; i < n; i++) {
      const p = f.pose3D[i];
      avg[i].x += p.x;
      avg[i].y += p.y;
      avg[i].z += p.z;
      avg[i].visibility += p.visibility ?? 1;
    }
  }
  if (count === 0) return null;
  for (const a of avg) {
    a.x /= count; a.y /= count; a.z /= count; a.visibility /= count;
  }
  return avg;
}

export function calibrate(frames) {
  const avg = averageLandmarks(frames);
  if (!avg) return null;
  const { hipQuat } = solveSpine(avg);
  const userForward = new THREE.Vector3(0, 0, -1).applyQuaternion(hipQuat);
  const correction  = quatFromUnitVectors(userForward, new THREE.Vector3(0, 0, -1));
  const hipHeight   = mpToThree(avg[MP.LEFT_HIP]).distanceTo(mpToThree(avg[MP.LEFT_SHOULDER]));
  return { correction, hipHeight };
}
