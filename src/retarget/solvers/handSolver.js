// solvers/handSolver.js
import * as THREE from 'three';
import { mpToThree } from '../coords.js';

const FINGERS = {
  thumb:  [1, 2, 3, 4],
  index:  [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring:   [13, 14, 15, 16],
  pinky:  [17, 18, 19, 20],
};

export function solveHand(handLm, side) {
  if (!handLm || handLm.length < 21) return null;
  const lm = handLm.map(mpToThree);

  const wrist     = lm[0];
  const indexMcp  = lm[5];
  const pinkyMcp  = lm[17];

  const palmFwd   = indexMcp.clone().add(pinkyMcp).multiplyScalar(0.5).sub(wrist).normalize();
  const palmRight = (side === 'left' ? indexMcp.clone().sub(pinkyMcp) : pinkyMcp.clone().sub(indexMcp)).normalize();
  const palmUp    = new THREE.Vector3().crossVectors(palmFwd, palmRight).normalize();

  const fingerDirs = {};
  for (const [name, idxs] of Object.entries(FINGERS)) {
    const pts = idxs.map(i => lm[i]);
    const dirs = [];
    let prev = wrist;
    for (const p of pts) {
      dirs.push(p.clone().sub(prev).normalize());
      prev = p;
    }
    // dirs[0] is wrist->CMC/MCP. dirs[1] is MCP->PIP. dirs[2] is PIP->DIP. dirs[3] is DIP->TIP.
    // VRM has 3 bones: Proximal, Intermediate, Distal.
    if (name === 'thumb') {
      // Drive the thumb's base bone using Wrist -> MCP to capture more of its sweeping motion
      const wristToMcp = lm[2].clone().sub(wrist).normalize();
      fingerDirs[name] = [wristToMcp, dirs[2], dirs[3]];
    } else {
      fingerDirs[name] = dirs.slice(1);
    }
  }

  return { palmFwd, palmUp, fingerDirs };
}
