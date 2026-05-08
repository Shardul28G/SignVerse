// solvers/spineSolver.js
import * as THREE from 'three';
import { mpToThree, quatFromBasis } from '../coords.js';
import * as MP from '../landmarks.js';

// Returns world-space directions/quats describing the trunk + head.
// All vectors are in Three space (post mpToThree flip).
export function solveSpine(lm) {
  const lh = mpToThree(lm[MP.LEFT_HIP]);
  const rh = mpToThree(lm[MP.RIGHT_HIP]);
  const ls = mpToThree(lm[MP.LEFT_SHOULDER]);
  const rs = mpToThree(lm[MP.RIGHT_SHOULDER]);

  const hipMid      = lh.clone().add(rh).multiplyScalar(0.5);
  const shoulderMid = ls.clone().add(rs).multiplyScalar(0.5);

  // Subject's right axis (after coord flip) = rightHip - leftHip points to +X side.
  const subjectRight = rh.clone().sub(lh).normalize();
  const subjectUp    = shoulderMid.clone().sub(hipMid).normalize();
  const subjectBack  = new THREE.Vector3().crossVectors(subjectRight, subjectUp).normalize();
  const orthoUp      = new THREE.Vector3().crossVectors(subjectBack, subjectRight).normalize();
  const hipQuat      = quatFromBasis(subjectRight, orthoUp, subjectBack);

  // Chest basis from shoulders
  const chestRight = rs.clone().sub(ls).normalize();
  const chestUp    = orthoUp;
  const chestBack  = new THREE.Vector3().crossVectors(chestRight, chestUp).normalize();
  const chestRightOrtho = new THREE.Vector3().crossVectors(chestUp, chestBack).normalize();
  const chestQuat  = quatFromBasis(chestRightOrtho, chestUp, chestBack);

  // Head: ears + nose
  const nose = mpToThree(lm[MP.NOSE]);
  const le   = mpToThree(lm[MP.LEFT_EAR]);
  const re   = mpToThree(lm[MP.RIGHT_EAR]);
  const earMid = le.clone().add(re).multiplyScalar(0.5);
  // After flip: subject's right ear has +X, left ear has -X. Right axis = re - le.
  const headRight = re.clone().sub(le).normalize();
  // "Forward" of head = from ears toward nose. After flip nose is at -Z (in front of avatar).
  const headFwd = nose.clone().sub(earMid).normalize();   // points from ear-mid to nose
  // Avatar's "back" (Z+) is away from where the head looks; nose looks at -Z, so back = -headFwd.
  const headBack = headFwd.clone().multiplyScalar(-1).normalize();
  const headUp   = new THREE.Vector3().crossVectors(headBack, headRight).normalize();
  const headRightOrtho = new THREE.Vector3().crossVectors(headUp, headBack).normalize();
  const headQuat = quatFromBasis(headRightOrtho, headUp, headBack);

  return {
    hipQuat, chestQuat, headQuat,
    hipMid, shoulderMid,
    spineDir: subjectUp,        // hips -> chest direction
    neckDir:  earMid.clone().sub(shoulderMid).normalize(),
  };
}
