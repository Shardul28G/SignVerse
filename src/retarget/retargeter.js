// retargeter.js
// Direction-driven retargeter. Each bone's "forward" axis (parent→child at rest)
// is captured once, then per-frame we swing that axis to align with the target
// world-space direction — all math done in the parent's CURRENT local frame so
// spine/chest movement doesn't corrupt arm placement.
import * as THREE from 'three';
import { quatFromUnitVectors, quatFromBasis } from './coords.js';

export const BONE_CHILDREN = {
  hips:          'spine',
  spine:         'chest',
  chest:         'upperChest',
  upperChest:    'neck',
  neck:          'head',
  leftShoulder:  'leftUpperArm',
  leftUpperArm:  'leftLowerArm',
  leftLowerArm:  'leftHand',
  rightShoulder: 'rightUpperArm',
  rightUpperArm: 'rightLowerArm',
  rightLowerArm: 'rightHand',
};

const FINGER_BONES = {
  left: {
    thumb:  ['leftThumbProximal',  'leftThumbIntermediate',  'leftThumbDistal'],
    index:  ['leftIndexProximal',  'leftIndexIntermediate',  'leftIndexDistal'],
    middle: ['leftMiddleProximal', 'leftMiddleIntermediate', 'leftMiddleDistal'],
    ring:   ['leftRingProximal',   'leftRingIntermediate',   'leftRingDistal'],
    pinky:  ['leftLittleProximal', 'leftLittleIntermediate', 'leftLittleDistal'],
  },
  right: {
    thumb:  ['rightThumbProximal',  'rightThumbIntermediate',  'rightThumbDistal'],
    index:  ['rightIndexProximal',  'rightIndexIntermediate',  'rightIndexDistal'],
    middle: ['rightMiddleProximal', 'rightMiddleIntermediate', 'rightMiddleDistal'],
    ring:   ['rightRingProximal',   'rightRingIntermediate',   'rightRingDistal'],
    pinky:  ['rightLittleProximal', 'rightLittleIntermediate', 'rightLittleDistal'],
  },
};

const ALL_BONES = new Set([
  ...Object.keys(BONE_CHILDREN), ...Object.values(BONE_CHILDREN),
  'leftHand', 'rightHand', 'head', 'upperChest',
  'leftShoulder', 'rightShoulder',
  ...Object.values(FINGER_BONES.left).flat(),
  ...Object.values(FINGER_BONES.right).flat(),
]);

function getBoneNode(vrm, name) {
  return (
    vrm.humanoid?.getNormalizedBoneNode?.(name) ??
    vrm.humanoid?.getBoneNode?.(name) ??
    null
  );
}

export function buildRestState(vrm) {
  vrm.scene.updateMatrixWorld(true);
  const state = {};

  for (const name of ALL_BONES) {
    const node = getBoneNode(vrm, name);
    if (!node) continue;
    // Ensure matrix world of this node and its parent is fresh.
    node.updateWorldMatrix(true, false);
    const parentRestWorldQuat = node.parent.getWorldQuaternion(new THREE.Quaternion());
    state[name] = {
      node,
      restLocalQuat:      node.quaternion.clone(),
      restWorldQuat:      node.getWorldQuaternion(new THREE.Quaternion()),
      restWorldPos:       node.getWorldPosition(new THREE.Vector3()),
      parentRestWorldInv: parentRestWorldQuat.clone().invert(),
    };
  }

  // For direction-driven bones: compute restDir (world) and localRestFwd
  // (restDir expressed in the parent's rest local frame).
  // localRestFwd is invariant: used every frame to swing the bone in parent-local space.
  for (const [bone, child] of Object.entries(BONE_CHILDREN)) {
    if (!state[bone] || !state[child]) continue;
    const dir = state[child].restWorldPos.clone().sub(state[bone].restWorldPos);
    if (dir.lengthSq() < 1e-8) continue;
    const restDir = dir.normalize();
    state[bone].restDir    = restDir;
    // localRestFwd = parentRestWorldInv * restDir
    state[bone].localRestFwd = restDir.clone().applyQuaternion(state[bone].parentRestWorldInv);
  }

  // Fingers rest dir
  for (const side of ['left', 'right']) {
    for (const bones of Object.values(FINGER_BONES[side])) {
      for (let i = 0; i < bones.length - 1; i++) {
        const bone = bones[i];
        const child = bones[i+1];
        if (!state[bone] || !state[child]) continue;
        const dir = state[child].restWorldPos.clone().sub(state[bone].restWorldPos);
        if (dir.lengthSq() < 1e-8) continue;
        const restDir = dir.normalize();
        state[bone].restDir = restDir;
        state[bone].localRestFwd = restDir.clone().applyQuaternion(state[bone].parentRestWorldInv);
        
        // Copy to Distal
        if (i === bones.length - 2) {
           const distal = bones[i+1];
           if (state[distal]) {
             state[distal].restDir = restDir;
             state[distal].localRestFwd = restDir.clone().applyQuaternion(state[distal].parentRestWorldInv);
           }
        }
      }
    }
  }

  // Hands rest basis
  for (const side of ['left', 'right']) {
    const hand = state[`${side}Hand`];
    const index = state[`${side}IndexProximal`];
    const pinky = state[`${side}LittleProximal`];
    if (hand && index && pinky) {
      const palmFwd = index.restWorldPos.clone().add(pinky.restWorldPos).multiplyScalar(0.5).sub(hand.restWorldPos).normalize();
      const palmRight = (side === 'left' ? index.restWorldPos.clone().sub(pinky.restWorldPos) : pinky.restWorldPos.clone().sub(index.restWorldPos)).normalize();
      const palmUp = new THREE.Vector3().crossVectors(palmFwd, palmRight).normalize();
      hand.restPalmFwd = palmFwd;
      hand.restPalmUp = palmUp;
    }
  }

  return state;
}

// Drive a bone by aligning its forward (parent→child) direction onto targetDir.
// The swing is computed in the parent's CURRENT local frame, so this is correct
// even when the parent bone has been driven away from rest this frame.
export function applyDirToBone(vrm, name, targetDir, restState, alpha = 1) {
  const s = restState[name];
  if (!s || !s.localRestFwd) return;
  const node = s.node;

  // Get parent's most-current world quaternion (call updateWorldMatrix first).
  node.parent.updateWorldMatrix(true, false);
  const parentCurWorldInv = node.parent.getWorldQuaternion(new THREE.Quaternion()).invert();

  // Target direction in parent's current local frame.
  const targetLocal = targetDir.clone().normalize().applyQuaternion(parentCurWorldInv);

  // Swing in local frame: from rest local forward to target local.
  const localSwing = quatFromUnitVectors(s.localRestFwd, targetLocal);

  // New local = localSwing ∘ restLocal  (premultiply: swing first, then orient).
  const newLocal = localSwing.multiply(s.restLocalQuat.clone());

  if (alpha >= 1) {
    node.quaternion.copy(newLocal);
  } else {
    node.quaternion.slerp(newLocal, alpha);
  }
  // Propagate so children see the updated matrix world immediately.
  node.updateWorldMatrix(false, true);
}

// Set a bone's rotation by specifying the desired WORLD quaternion directly.
// Used for hips/chest/head where the spine solver gives a full orientation.
export function setWorldRotation(vrm, name, worldQuat, alpha = 1) {
  const node = getBoneNode(vrm, name);
  if (!node) return;

  node.parent.updateWorldMatrix(true, false);
  const parentWorldInv = node.parent.getWorldQuaternion(new THREE.Quaternion()).invert();
  const local = parentWorldInv.multiply(worldQuat);

  if (alpha >= 1) {
    node.quaternion.copy(local);
  } else {
    node.quaternion.slerp(local, alpha);
  }
  node.updateWorldMatrix(false, true);
}

// Apply a delta quaternion on top of the bone's rest local rotation.
export function applyLocalDelta(vrm, name, deltaQuat, restState, alpha = 1) {
  const s = restState[name];
  if (!s) return;
  const target = s.restLocalQuat.clone().multiply(deltaQuat);
  if (alpha >= 1) {
    s.node.quaternion.copy(target);
  } else {
    s.node.quaternion.slerp(target, alpha);
  }
  s.node.updateWorldMatrix(false, true);
}

export function alignHandBasis(vrm, name, targetFwd, targetUp, restState, alpha = 1) {
  const s = restState[name];
  if (!s || !s.restPalmFwd) return;
  const node = s.node;
  node.parent.updateWorldMatrix(true, false);
  const parentWorldInv = node.parent.getWorldQuaternion(new THREE.Quaternion()).invert();

  const restRight = new THREE.Vector3().crossVectors(s.restPalmUp, s.restPalmFwd).normalize();
  const restQuat = quatFromBasis(restRight, s.restPalmUp, s.restPalmFwd);

  const targetRight = new THREE.Vector3().crossVectors(targetUp, targetFwd).normalize();
  const targetQuat = quatFromBasis(targetRight, targetUp, targetFwd);

  const rot = targetQuat.multiply(restQuat.invert());
  const targetWorld = rot.multiply(s.restWorldQuat);

  const local = parentWorldInv.multiply(targetWorld);
  if (alpha >= 1) {
    node.quaternion.copy(local);
  } else {
    node.quaternion.slerp(local, alpha);
  }
  node.updateWorldMatrix(false, true);
}

export function applyFingersWorld(vrm, side, fingerDirs, restState, alpha = 1) {
  const map = FINGER_BONES[side];
  if (!map || !fingerDirs) return;
  for (const [finger, dirs] of Object.entries(fingerDirs)) {
    const bones = map[finger];
    if (!bones) continue;
    for (let i = 0; i < bones.length && i < dirs.length; i++) {
      applyDirToBone(vrm, bones[i], dirs[i], restState, alpha);
    }
  }
}

export function resetToRest(restState) {
  for (const s of Object.values(restState)) {
    if (s?.node && s.restLocalQuat) s.node.quaternion.copy(s.restLocalQuat);
  }
}
