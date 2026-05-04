/**
 * Avatar-only viewer driven by the custom retargeting pipeline.
 * Loads anim.vrm + fever2.json, auto-fits the avatar to the viewport,
 * and plays back at 25 fps using the pipeline in src/retarget/.
 */

import React, { useEffect, useRef, useState, Suspense } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";
import * as THREE from "three";

import { RetargetPipeline } from "./retarget/pipeline.js";
import { StickFigure } from "./retarget/stickFigure.js";

const BG_COLOR = "#e8ecf2";       // light background

const FPS = 25;
const FRAME_TIME = 1 / FPS;
const VRM_URL = "./anim.vrm";
const JSON_URL = "./You.json";

// ─── VRM loader ──────────────────────────────────────────────────────────────
function useVRM(url) {
  const [vrm, setVrm] = useState(null);
  useEffect(() => {
    if (!url) return;
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));
    loader.load(
      url,
      (gltf) => {
        const model = gltf.userData.vrm;
        VRMUtils.removeUnnecessaryJoints(gltf.scene);
        VRMUtils.removeUnnecessaryVertices(gltf.scene);
        // rotateVRM0 is intentionally NOT called here.
        // With makeScale(-1,-1,1) in coords.js, the spine solver drives the avatar
        // to the correct world orientation. rotateVRM0 would double-flip it.
        setVrm(model);
      },
      undefined,
      (err) => console.error("VRM load error:", err),
    );
  }, [url]);
  return vrm;
}

// ─── Auto-fit camera around the avatar bounding box ──────────────────────────
// `extraWidth` widens the assumed scene width as a multiplier of the avatar's
// width — used to leave room for the stick figure beside the avatar.
function FitCameraToObject({ object, padding = 1.25, extraWidth = 1.0 }) {
  const { camera, size } = useThree();
  useEffect(() => {
    if (!object) return;
    const box = new THREE.Box3().setFromObject(object);
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const sizeVec = box.getSize(new THREE.Vector3());

    const effectiveWidth = sizeVec.x * extraWidth;
    const fov = THREE.MathUtils.degToRad(camera.fov);
    const aspect = size.width / size.height;
    const distV = (sizeVec.y / 2) / Math.tan(fov / 2);
    const distH = (effectiveWidth / 2) / (Math.tan(fov / 2) * aspect);
    const dist = Math.max(distV, distH) * padding;

    // Shift camera slightly to the right so both avatar and stick figure are framed.
    const shiftX = sizeVec.x * (extraWidth - 1) * 0.5;
    camera.position.set(center.x + shiftX, center.y, center.z + dist);
    camera.near = Math.max(0.01, dist / 100);
    camera.far  = dist * 100;
    camera.lookAt(center.x + shiftX, center.y, center.z);
    camera.updateProjectionMatrix();
  }, [object, camera, size, padding, extraWidth]);
  return null;
}

// ─── Animated VRM scene ──────────────────────────────────────────────────────
function VRMScene({ frames }) {
  const vrm = useVRM(VRM_URL);
  const groupRef = useRef();
  const stickGroupRef = useRef();
  const pipelineRef = useRef(null);
  const stickRef = useRef(null);
  const playback = useRef({ accumulator: 0, idx: 0 });

  useEffect(() => {
    if (!vrm || !groupRef.current) return;
    groupRef.current.add(vrm.scene);
    groupRef.current.updateMatrixWorld(true);
    pipelineRef.current = new RetargetPipeline(vrm, { alpha: 0.45, fingerAlpha: 0.5 });

    // Attach the stick figure — beside the avatar, anchored at avatar's hip height.
    if (stickGroupRef.current) {
      const box = new THREE.Box3().setFromObject(vrm.scene);
      const center = box.getCenter(new THREE.Vector3());
      const sizeVec = box.getSize(new THREE.Vector3());
      const sf = new StickFigure({
        scale: 1.0,
        offset: new THREE.Vector3(center.x + sizeVec.x * 1.1, center.y, center.z),
      });
      stickGroupRef.current.add(sf.root);
      stickRef.current = sf;
    }

    return () => {
      groupRef.current?.remove(vrm.scene);
      if (stickRef.current) {
        stickGroupRef.current?.remove(stickRef.current.root);
        stickRef.current.dispose();
        stickRef.current = null;
      }
      pipelineRef.current = null;
    };
  }, [vrm]);

  useFrame((_, delta) => {
    if (!vrm || !frames.length || !pipelineRef.current) return;
    const pb = playback.current;
    pb.accumulator += delta;
    while (pb.accumulator >= FRAME_TIME) {
      pb.accumulator -= FRAME_TIME;
      pb.idx = (pb.idx + 1) % frames.length;
    }
    const t = performance.now() / 1000;
    const frame = frames[pb.idx];
    pipelineRef.current.step(frame, t);
    stickRef.current?.update(frame);
  });

  return (
    <>
      <group ref={groupRef} />
      <group ref={stickGroupRef} />
      {vrm && <FitCameraToObject object={vrm.scene} padding={1.6} extraWidth={1.6} />}
    </>
  );
}

// ─── Lighting ────────────────────────────────────────────────────────────────
function Lights() {
  return (
    <>
      <ambientLight intensity={1.0} />
      <hemisphereLight args={["#ffffff", "#c8d0dc", 0.6]} />
      <directionalLight position={[3, 6, 4]} intensity={1.0} />
      <directionalLight position={[-3, 4, -4]} intensity={0.35} color="#aaccff" />
    </>
  );
}

// ─── App ─────────────────────────────────────────────────────────────────────
export default function App() {
  const [frames, setFrames] = useState([]);
  const [error, setError]   = useState(null);

  useEffect(() => {
    fetch(JSON_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status} loading ${JSON_URL}`);
        return r.json();
      })
      .then((data) => {
        const arr = Array.isArray(data) ? data : data.frames ?? [];
        arr.sort((a, b) => (a.frameIndex ?? 0) - (b.frameIndex ?? 0));
        setFrames(arr);
      })
      .catch((e) => setError(e.message));
  }, []);

  return (
    <div style={{ width: "100vw", height: "100vh", background: BG_COLOR, overflow: "hidden" }}>
      <Canvas
        camera={{ position: [0, 1.4, 3], fov: 35, near: 0.01, far: 100 }}
        gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}
      >
        <color attach="background" args={[BG_COLOR]} />
        <Lights />
        <Suspense fallback={null}>
          {frames.length > 0 && <VRMScene frames={frames} />}
        </Suspense>
      </Canvas>
      {error && (
        <div style={{
          position: "absolute", bottom: 12, left: 12,
          color: "#aa2222", fontFamily: "monospace", fontSize: 12,
        }}>⚠ {error}</div>
      )}
    </div>
  );
}
