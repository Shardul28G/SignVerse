// solvers/armSolver.js
import * as THREE from 'three';
import { mpToThree, basisFromDirAndUp } from '../coords.js';
import * as MP from '../landmarks.js';

const SIDES = {
  left: {
    sh: MP.LEFT_SHOULDER, el: MP.LEFT_ELBOW, wr: MP.LEFT_WRIST,
    pinky: MP.LEFT_PINKY, index: MP.LEFT_INDEX,
  },
  right: {
    sh: MP.RIGHT_SHOULDER, el: MP.RIGHT_ELBOW, wr: MP.RIGHT_WRIST,
    pinky: MP.RIGHT_PINKY, index: MP.RIGHT_INDEX,
  },
};

// Returns WORLD-SPACE direction vectors per limb segment.
// The retargeter aligns each VRM bone's rest direction onto these.
export function solveArm(side, lm) {
  const S = SIDES[side];
  if (!S) return null;

  const sh = mpToThree(lm[S.sh]);
  const el = mpToThree(lm[S.el]);
  const wr = mpToThree(lm[S.wr]);
  const idx = mpToThree(lm[S.index]);
  const pky = mpToThree(lm[S.pinky]);

  const upperDir = el.clone().sub(sh).normalize();
  const lowerDir = wr.clone().sub(el).normalize();

  // Hand basis: forward = wrist→middle-of-palm, up = pinky→index (or reverse per side)
  const palmAcross = idx.clone().sub(pky).normalize();
  const handBasis  = basisFromDirAndUp(lowerDir, palmAcross);

  return { upperDir, lowerDir, handBasis, shoulderPos: sh, elbowPos: el, wristPos: wr };
}
