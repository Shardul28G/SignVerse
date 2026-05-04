// solvers/faceSolver.js

export function solveFace(landmarks) {
  if (!landmarks || landmarks.length < 468) return null;

  // Helper to compute 3D Euclidean distance
  const dist = (p1, p2) => {
    if (!p1 || !p2) return 0;
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    const dz = p1.z - p2.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  };

  // Use face height (top of head to chin) for normalization
  const faceHeight = dist(landmarks[10], landmarks[152]);
  if (faceHeight <= 0.001) return null;

  // 1. Mouth Openness (maps to 'A' vowel)
  // Landmarks 13 (upper inner lip) and 14 (lower inner lip)
  const mouthDist = dist(landmarks[13], landmarks[14]);
  const mouthRatio = mouthDist / faceHeight;
  
  // Adjusted thresholds: 0.015 is closed, 0.07 is wide open
  let mouthA = (mouthRatio - 0.015) / 0.055;
  mouthA = Math.max(0, Math.min(1, mouthA));

  // 2. Eye Blinking
  // Left Eye: 159 (top), 145 (bottom)
  const leftEyeDist = dist(landmarks[159], landmarks[145]);
  const leftEyeRatio = leftEyeDist / faceHeight;
  
  // Right Eye: 386 (top), 374 (bottom)
  const rightEyeDist = dist(landmarks[386], landmarks[374]);
  const rightEyeRatio = rightEyeDist / faceHeight;

  // Thresholds for eyes: 0.01 is fully closed, 0.028 is fully open
  // We invert it because VRM blink is 1.0 for closed and 0.0 for open
  let blinkL = 1.0 - ((leftEyeRatio - 0.01) / 0.018);
  blinkL = Math.max(0, Math.min(1, blinkL));

  let blinkR = 1.0 - ((rightEyeRatio - 0.01) / 0.018);
  blinkR = Math.max(0, Math.min(1, blinkR));

  return {
    mouthA,
    blinkLeft: blinkL,
    blinkRight: blinkR
  };
}
