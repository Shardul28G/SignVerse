// pipeline.js — orchestrates filter -> solve -> retarget for one frame.
// Frame shape (matches fever2.json):
//   { pose3D, pose2D, faceLandmarks, leftHand, rightHand, leftHandWorld, rightHandWorld }
//
// MIRROR MODE (default, paired with MP_TO_THREE = makeScale(1,-1,-1) in coords.js):
//   Subject's LEFT body landmarks live on +X in Three -> drive avatar's RIGHT bones.
//   Subject's RIGHT body landmarks live on -X in Three -> drive avatar's LEFT bones.
//   Hand labels in MP Holistic are by image-position (camera POV):
//     frame.leftHandWorld  = subject's RIGHT hand  -> in mirror mode drives avatar LEFT
//     frame.rightHandWorld = subject's LEFT  hand  -> in mirror mode drives avatar RIGHT

import * as THREE from 'three';
import { solveSpine } from './solvers/spineSolver.js';
import { solveArm }   from './solvers/armSolver.js';
import { solveHand }  from './solvers/handSolver.js';
import {
  buildRestState, applyDirToBone, setWorldRotation,
  applyLocalDelta, applyFingersWorld, alignHandBasis, resetHandToRest
} from './retargeter.js';
import { makeFilterArray, filterLandmarks } from './filter.js';
import { quatFromUnitVectors } from './coords.js';
import { solveFace } from './solvers/faceSolver.js';

export class RetargetPipeline {
  constructor(vrm, opts = {}) {
    this.vrm = vrm;
    this.alpha = opts.alpha ?? 0.4;          // smoothing slerp factor per frame
    this.fingerAlpha = opts.fingerAlpha ?? 0.5;
    this.driveFingers = opts.driveFingers ?? true;
    this.restState = buildRestState(vrm);
    this.poseFilters = makeFilterArray(33, { minCutoff: 1.5, beta: 0.05 });
    this.lhFilters = makeFilterArray(21, { minCutoff: 2.0, beta: 0.1 });
    this.rhFilters = makeFilterArray(21, { minCutoff: 2.0, beta: 0.1 });
  }

  step(frame, t) {
    if (!frame) return;
    const pose = filterLandmarks(frame.pose3D, this.poseFilters, t);
    if (!pose || pose.length < 33) return;

    // ── Spine / chest / head ────────────────────────────────────────────────
    const spine = solveSpine(pose);
    
    // Stabilize hips: keep only Y-axis rotation (yaw) to prevent tilting/floating
    const hipEuler = new THREE.Euler().setFromQuaternion(spine.hipQuat, 'YXZ');
    hipEuler.x = 0; // Remove pitch
    hipEuler.z = 0; // Remove roll
    const stableHipQuat = new THREE.Quaternion().setFromEuler(hipEuler);
    setWorldRotation(this.vrm, 'hips', stableHipQuat, this.alpha);

    // chest (or spine fallback)
    if (this.restState.chest)      setWorldRotation(this.vrm, 'chest', spine.chestQuat, this.alpha);
    else if (this.restState.spine) setWorldRotation(this.vrm, 'spine', spine.chestQuat, this.alpha);

    if (this.restState.upperChest) setWorldRotation(this.vrm, 'upperChest', spine.chestQuat, this.alpha * 0.6);

    // neck — point neck rest dir along (shoulderMid -> earMid)
    if (this.restState.neck?.restDir) {
      applyDirToBone(this.vrm, 'neck', spine.neckDir, this.restState, this.alpha * 0.6);
    }

    // head
    if (this.restState.head) setWorldRotation(this.vrm, 'head', spine.headQuat, this.alpha);

    // ── Arms (mirror mode: subject-left -> avatar-right) ───────────────────
    const left  = solveArm('left',  pose);   // subject's left side, lives on +X in Three
    const right = solveArm('right', pose);   // subject's right side, lives on -X in Three

    if (left) {
      applyDirToBone(this.vrm, 'leftUpperArm', left.upperDir, this.restState, this.alpha);
      applyDirToBone(this.vrm, 'leftLowerArm', left.lowerDir, this.restState, this.alpha);
    }
    if (right) {
      applyDirToBone(this.vrm, 'rightUpperArm', right.upperDir, this.restState, this.alpha);
      applyDirToBone(this.vrm, 'rightLowerArm', right.lowerDir, this.restState, this.alpha);
    }

    // ── Hands (mirror mode) ─────────────────────────────────────────────────
    if (this.driveFingers) {
      // frame.leftHandWorld  = subject's LEFT hand (screen right) -> avatar LEFT (screen right)
      // frame.rightHandWorld = subject's RIGHT hand (screen left) -> avatar RIGHT (screen left)
      const subjectLeftHand  = frame.leftHandWorld;
      const subjectRightHand = frame.rightHandWorld;

      if (subjectRightHand?.length >= 21) {
        const filtered = filterLandmarks(subjectRightHand, this.rhFilters, t);
        const sol = solveHand(filtered, 'right');     // applied to avatar RIGHT
        if (sol) {
          alignHandBasis(this.vrm, 'rightHand', sol.palmFwd, sol.palmUp, this.restState, this.alpha);
          applyFingersWorld(this.vrm, 'right', sol.fingerDirs, this.restState, this.fingerAlpha);
        }
      } else {
        resetHandToRest(this.restState, 'right', this.fingerAlpha);
      }
      if (subjectLeftHand?.length >= 21) {
        const filtered = filterLandmarks(subjectLeftHand, this.lhFilters, t);
        const sol = solveHand(filtered, 'left');    // applied to avatar LEFT
        if (sol) {
          alignHandBasis(this.vrm, 'leftHand', sol.palmFwd, sol.palmUp, this.restState, this.alpha);
          applyFingersWorld(this.vrm, 'left', sol.fingerDirs, this.restState, this.fingerAlpha);
        }
      } else {
        resetHandToRest(this.restState, 'left', this.fingerAlpha);
      }
    }

    // ── Face Expressions ────────────────────────────────────────────────────
    if (frame.faceLandmarks) {
      const faceData = solveFace(frame.faceLandmarks);
      const em = this.vrm.expressionManager;
      if (em && faceData) {
        // Set VRM 1.0 and VRM 0.0 standard expression names to be safe
        em.setValue('aa', faceData.mouthA); em.setValue('a', faceData.mouthA);
        
        // In mirror mode, we swap the blinks so that the subject winking their right eye 
        // (which appears on the left side of the screen) makes the avatar's left eye wink
        em.setValue('blinkLeft', faceData.blinkRight);
        em.setValue('blinkRight', faceData.blinkLeft);
        
        // Zero out others for now
        em.setValue('ee', 0); em.setValue('e', 0);
        em.setValue('ih', 0); em.setValue('i', 0);
        em.setValue('oh', 0); em.setValue('o', 0);
        em.setValue('ou', 0); em.setValue('u', 0);
      }
    }

    this.vrm.update(1 / 25);
  }
}
