// skeleton.js — canonical rest-pose directions in Three world axes.
// Avatar faces -Z, head up +Y, avatar's right = +X, avatar's left = -X.
import * as THREE from 'three';

export const REST = {
  spine:         new THREE.Vector3( 0, 1, 0),
  chest:         new THREE.Vector3( 0, 1, 0),
  neck:          new THREE.Vector3( 0, 1, 0),
  head:          new THREE.Vector3( 0, 1, 0),
  leftUpperArm:  new THREE.Vector3(-1, 0, 0),
  leftLowerArm:  new THREE.Vector3(-1, 0, 0),
  rightUpperArm: new THREE.Vector3( 1, 0, 0),
  rightLowerArm: new THREE.Vector3( 1, 0, 0),
};
