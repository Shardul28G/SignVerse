/**
 * ISL Helper — split-layout redesign.
 *
 * Flow:
 *   1. User enters text / picks an image / records audio.
 *   2. Input is sent to a local llama.cpp server (http://127.0.0.1:8080)
 *      running a Gemma + ISL-gloss LoRA, which returns an ISL gloss line.
 *   3. Each gloss token is mapped to a landmark JSON in /public, and the
 *      clips are concatenated (with smooth transitions) and played on the
 *      VRM avatar at 25 fps.
 */

import React, { useEffect, useRef, useState, Suspense, useCallback } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";
import * as THREE from "three";
import { RetargetPipeline } from "./retarget/pipeline.js";
import "./App.css";

const FPS = 25;
const FRAME_TIME = 1 / FPS;
const VRM_URL = "./anim.vrm";

// Set VITE_LLAMA_URL on Vercel to your Modal endpoint, e.g.
//   https://<workspace>--isl-llama-server-serve.modal.run/v1/chat/completions
// Locally falls back to the Vite dev proxy in vite.config.js.
const LLAMA_URL =
  import.meta.env.VITE_LLAMA_URL || "/api/llama/v1/chat/completions";

// Set VITE_LANDMARKS_BASE_URL on Vercel to your public R2 URL, e.g.
//   https://pub-<id>.r2.dev   or   https://landmarks.your-domain.com
// In dev, vite.config.js serves data_tools/landmarks/*.json at /landmarks/*.json,
// and frontend/public/landmarks/ provides the 4 always-bundled fallbacks
// (Hello, Wait, Problem, Please) which also ship with the Vercel build.
const LANDMARKS_BASE_URL =
  import.meta.env.VITE_LANDMARKS_BASE_URL || "/landmarks";

// These four ship with the frontend bundle (frontend/public/landmarks/) so the
// initial UI states (greetings, loading/error fallbacks) never depend on R2.
const BUNDLED_LANDMARKS = new Set(["Hello", "Wait", "Problem", "Please"]);

function landmarkUrl(name) {
  if (BUNDLED_LANDMARKS.has(name)) return `/landmarks/${name}.json`;
  return `${LANDMARKS_BASE_URL}/${name}.json`;
}

// Sourced from data_tools/landmarks/*.json — every entry here has a matching
// landmark JSON either bundled (frontend/public/landmarks/) or on R2.
const DICTIONARY = [
  "After", "Age", "All", "Bad", "Beautiful", "Below", "Big", "Book", "Boy",
  "Capital", "Caution", "Child", "City", "Cold", "Come", "Country", "Deaf",
  "Dinner", "Doctor", "Drink", "Eat", "Evening", "Family", "Far", "Fast",
  "Father", "Fever", "Floor", "Food", "Girl", "Give", "Go", "Good", "Happy",
  "He", "Hello", "Help", "Here", "Hospital", "Hot", "Hungry", "India", "Know",
  "Language", "Large", "Learn", "Like", "Maharashtra", "Make", "Man", "More",
  "Morning", "Mother", "Mumbai", "Name", "New", "Night", "No", "One", "Play",
  "Please", "Population", "Problem", "Read", "Ready", "Remember", "Right",
  "Run", "Sad", "School", "See", "She", "Sick", "Sign", "Sister", "Slow",
  "Small", "Sorry", "State", "Stop", "Tablet", "Take", "Teach", "Teacher",
  "There", "They", "Thirsty", "Time", "Today", "Tomorrow", "Tonight", "Very",
  "Walk", "Water", "We", "Wet", "What", "Where", "Why", "Year", "Yes",
  "Yesterday", "You", "Your",
];

// Shared instruction block. We deliver this in the USER turn (not the system
// turn) because the f16 base model follows in-turn instructions much more
// reliably than system-role guidance — local testing showed system-role
// rules being ignored on image inputs (e.g. "WHAT YOU" for an image of a
// sign). Putting the rules adjacent to the actual input, and asking the
// model to think first, fixes that.
const GLOSS_INSTRUCTIONS =
  "Task: convert the input below into an Indian Sign Language (ISL) gloss line.\n" +
  "ISL gloss is a sequence of UPPERCASE keyword tokens in topic-comment order, " +
  "with no articles, no auxiliary verbs, and no punctuation.\n\n" +
  "First, think step-by-step inside a <think>…</think> block:\n" +
  "  • What is the MEANING of the input? (For an image: what does the scene/sign communicate? Not 'a photo of X'.)\n" +
  "  • Which content words carry that meaning?\n" +
  "  • Reorder them into ISL topic-comment order and uppercase them.\n" +
  "Then, AFTER closing </think>, output exactly ONE LINE containing ONLY the gloss tokens.\n\n" +
  "STRICT OUTPUT RULES:\n" +
  "1. The final line MUST be non-empty. Empty output is a FAILURE.\n" +
  "2. The final line MUST contain ONLY uppercase gloss tokens — no quotes, no labels " +
  "(no 'Answer:', 'Gloss:'), no markdown, no bullets, no explanation.\n" +
  "3. NEVER output the word HAVE. ISL gloss does not use 'have'.\n" +
  "4. NEVER output articles (a, an, the) or auxiliary verbs (is, am, are, was, were, do, does, did, will, would, can, could, should, may, might).\n" +
  "5. IMAGES: gloss the MEANING the image conveys, NOT the act of viewing it. " +
  "NEVER output tokens like SEE, PHOTO, PICTURE, IMAGE, LOOK, SHOW, READ, TEXT, SIGN, WHAT, YOU — " +
  "those describe the medium, not the message.\n" +
  "   Example: a Radiation Hazard sign → RADIATION HAZARD   (NOT 'WHAT YOU' or 'SEE SIGN RADIATION').\n" +
  "   Example: a photo of a wet floor warning → CAUTION WET FLOOR.\n\n" +
  "Example (text input 'What is the largest city in India?'):  NAME YOU WHAT INDIA STATE LARGE CAPITAL";

// ─── SVG icon helpers ─────────────────────────────────────────────────────────
const Ico = ({ d, size = 14, fill = "none", sw = 2 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={fill}
    stroke="currentColor" strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round">
    {Array.isArray(d) ? d.map((p, i) => <path key={i} d={p} />) : <path d={d} />}
  </svg>
);

const TextIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 7V4h16v3" /><path d="M9 20h6" /><path d="M12 4v16" />
  </svg>
);
const MicIcon = ({ size = 14 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z" />
    <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
    <path d="M12 19v3" />
  </svg>
);
const ImageIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2" />
    <circle cx="9" cy="9" r="2" />
    <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
  </svg>
);
const TrashIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 6h18" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
    <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
  </svg>
);
const InfoIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <path d="M12 16v-4" /><path d="M12 8h.01" />
  </svg>
);
const PlayIcon = ({ size = 14 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <path d="M8 5v14l11-7z" />
  </svg>
);
const PauseIcon = ({ size = 14 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <path d="M6 4h4v16H6zM14 4h4v16h-4z" />
  </svg>
);
const PrevIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
    <path d="M6 6h2v12H6zM9.5 12l8.5 6V6z" />
  </svg>
);
const NextIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
    <path d="M16 6h2v12h-2zM6 18l8.5-6L6 6z" />
  </svg>
);
const ReplayIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 12a9 9 0 1 0 3-6.7" /><path d="M3 4v5h5" />
  </svg>
);
const BookIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
    <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
  </svg>
);
const UploadIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="17 8 12 3 7 8" />
    <line x1="12" y1="3" x2="12" y2="15" />
  </svg>
);
const SettingsIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
  </svg>
);
const CaptionsIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="2" y="5" width="20" height="14" rx="2" />
    <path d="M7 15h2" /><path d="M11 15h6" /><path d="M7 11h10" />
  </svg>
);
const FullscreenIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 7V3h4" /><path d="M21 7V3h-4" />
    <path d="M3 17v4h4" /><path d="M21 17v4h-4" />
  </svg>
);
const ChevDownIcon = () => (
  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="6 9 12 15 18 9" />
  </svg>
);

// ─── VRM loader ───────────────────────────────────────────────────────────────
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
        setVrm(model);
      },
      undefined,
      (err) => console.error("VRM load error:", err),
    );
  }, [url]);
  return vrm;
}

// ─── Auto-fit camera ──────────────────────────────────────────────────────────
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
    const upperY = center.y + sizeVec.y * 0.25;
    const zoomDist = dist * 0.8;
    camera.position.set(center.x, upperY, center.z + zoomDist);
    camera.near = Math.max(0.01, zoomDist / 100);
    camera.far = zoomDist * 100;
    camera.lookAt(center.x, upperY, center.z);
    camera.updateProjectionMatrix();
  }, [object, camera, size, padding, extraWidth]);
  return null;
}

// ─── VRM scene (animated, with play/speed/progress/jump control) ──────────────
function VRMScene({ frames, wordRanges, onWordChange, playing, speed, onProgress, controlRef }) {
  const vrm = useVRM(VRM_URL);
  const groupRef = useRef();
  const pipelineRef = useRef(null);
  const playback = useRef({ accumulator: 0, idx: 0 });
  const lastWordIdx = useRef(-1);

  useEffect(() => {
    if (!vrm || !groupRef.current) return;
    groupRef.current.add(vrm.scene);
    groupRef.current.updateMatrixWorld(true);
    pipelineRef.current = new RetargetPipeline(vrm, { alpha: 0.45, fingerAlpha: 0.5 });
    return () => {
      groupRef.current?.remove(vrm.scene);
      pipelineRef.current = null;
    };
  }, [vrm]);

  useEffect(() => {
    playback.current.idx = 0;
    playback.current.accumulator = 0;
    lastWordIdx.current = -1;
  }, [frames]);

  useEffect(() => {
    if (!controlRef) return;
    controlRef.current = {
      jumpToWord: (wordIdx) => {
        if (wordRanges && wordRanges[wordIdx]) {
          playback.current.idx = wordRanges[wordIdx].start;
          playback.current.accumulator = 0;
        }
      },
    };
  }, [wordRanges, controlRef]);

  useFrame((_, delta) => {
    if (!vrm || !frames.length || !pipelineRef.current) return;
    const pb = playback.current;
    if (playing) {
      pb.accumulator += delta * (speed ?? 1.0);
      while (pb.accumulator >= FRAME_TIME) {
        pb.accumulator -= FRAME_TIME;
        pb.idx = (pb.idx + 1) % frames.length;
      }
    }
    const t = performance.now() / 1000;
    pipelineRef.current.step(frames[pb.idx], t);

    if (onProgress) onProgress(frames.length > 0 ? pb.idx / frames.length : 0);

    if (wordRanges && wordRanges.length && onWordChange) {
      let wIdx = -1;
      for (let i = 0; i < wordRanges.length; i++) {
        if (pb.idx >= wordRanges[i].start && pb.idx < wordRanges[i].end) {
          wIdx = i;
          break;
        }
      }
      if (wIdx !== lastWordIdx.current) {
        lastWordIdx.current = wIdx;
        onWordChange(wIdx);
      }
    }
  });

  return (
    <>
      <group ref={groupRef} />
      {vrm && <FitCameraToObject object={vrm.scene} padding={1.05} />}
    </>
  );
}

// ─── Lights ───────────────────────────────────────────────────────────────────
function Lights() {
  return (
    <>
      <ambientLight intensity={1.0} />
      <hemisphereLight args={["#ffffff", "#e8dece", 0.6]} />
      <directionalLight position={[3, 6, 4]} intensity={1.0} />
      <directionalLight position={[-3, 4, -4]} intensity={0.35} color="#aaccff" />
    </>
  );
}

// ─── Frame lerp (clip transitions) ───────────────────────────────────────────
function lerpFrame(frameA, frameB, t) {
  const lerpArr = (arrA, arrB) => {
    if (!arrA || !arrB) return arrA || arrB;
    if (arrA.length !== arrB.length) return arrA;
    return arrA.map((pt, i) => {
      const pb = arrB[i];
      if (!pt || !pb) return pt || pb;
      return {
        x: pt.x + (pb.x - pt.x) * t,
        y: pt.y + (pb.y - pt.y) * t,
        z: pt.z + (pb.z - pt.z) * t,
        visibility:
          pt.visibility !== undefined && pb.visibility !== undefined
            ? pt.visibility + (pb.visibility - pt.visibility) * t
            : pt.visibility,
      };
    });
  };
  return {
    pose3D: lerpArr(frameA.pose3D, frameB.pose3D),
    leftHandWorld: lerpArr(frameA.leftHandWorld, frameB.leftHandWorld),
    rightHandWorld: lerpArr(frameA.rightHandWorld, frameB.rightHandWorld),
    faceLandmarks: lerpArr(frameA.faceLandmarks, frameB.faceLandmarks),
  };
}

// Words the model may use that have no ISL clip — mapped to a signed synonym.
const GLOSS_SYNONYMS = {
  high: "large",
  highest: "large",
  most: "large",
  yours: "your",
};

// Tokens to drop entirely from the gloss before matching — words ISL gloss
// never uses (e.g. "have"). Even if the model emits them, we strip them.
const GLOSS_DROP = new Set(["have", "be"]);

// ─── Gloss → matched / skipped clip names ────────────────────────────────────
function glossToClipNames(gloss) {
  const tokens = gloss
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean)
    .map((t) => t.toLowerCase())
    .filter((t) => !GLOSS_DROP.has(t))
    .map((t) => GLOSS_SYNONYMS[t] ?? t);

  const dictLower = DICTIONARY.map((d) => ({
    name: d,
    tokens: d.toLowerCase().split("_"),
  })).sort((a, b) => b.tokens.length - a.tokens.length);

  const matched = [];
  const skipped = [];
  let i = 0;
  while (i < tokens.length) {
    let hit = null;
    for (const entry of dictLower) {
      const slice = tokens.slice(i, i + entry.tokens.length);
      if (
        slice.length === entry.tokens.length &&
        slice.every((s, k) => s === entry.tokens[k])
      ) {
        hit = entry;
        break;
      }
    }
    if (hit) {
      matched.push(hit.name);
      i += hit.tokens.length;
    } else {
      skipped.push(tokens[i]);
      i += 1;
    }
  }
  return { matched, skipped };
}

// ─── Load + concatenate landmark JSONs ───────────────────────────────────────
async function loadClipFrames(clipNames) {
  const TRANSITION_FRAMES = 15;
  const datas = await Promise.all(
    clipNames.map((name) =>
      fetch(landmarkUrl(name)).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status} loading ${name}.json`);
        return r.json();
      }),
    ),
  );
  let combined = [];
  const wordRanges = [];
  for (let i = 0; i < datas.length; i++) {
    const arr = (Array.isArray(datas[i]) ? datas[i] : datas[i].frames ?? [])
      .slice()
      .sort((a, b) => (a.frameIndex ?? 0) - (b.frameIndex ?? 0));
    const start = combined.length;
    if (i > 0 && combined.length > 0 && arr.length > 0) {
      const last = combined[combined.length - 1];
      const first = arr[0];
      for (let s = 1; s <= TRANSITION_FRAMES; s++) {
        combined.push(lerpFrame(last, first, s / (TRANSITION_FRAMES + 1)));
      }
    }
    combined = combined.concat(arr);
    wordRanges.push({ name: clipNames[i], start, end: combined.length });
  }
  return { frames: combined, wordRanges };
}

// ─── LLM call ─────────────────────────────────────────────────────────────────
async function fetchGloss({ text, imageDataUrl, audioDataUrl }) {
  let userContent;
  if (audioDataUrl) {
    const base64 = audioDataUrl.split(",")[1];
    userContent = [
      {
        type: "text",
        text:
          GLOSS_INSTRUCTIONS +
          "\n\nInput: the following audio clip. Transcribe it mentally inside <think>, then produce the gloss line." +
          (text ? `\nUser hint: ${text}` : ""),
      },
      { type: "input_audio", input_audio: { data: base64, format: "wav" } },
    ];
  } else if (imageDataUrl) {
    // llama-cpp-python's vision pipeline requires image_url FIRST — it binds
    // the image by position before reading the text prompt.
    userContent = [
      { type: "image_url", image_url: { url: imageDataUrl } },
      {
        type: "text",
        text:
          GLOSS_INSTRUCTIONS +
          "\n\nInput: the image above. Inside <think>, describe what the image actually communicates " +
          "(the message, not 'a photo of…'), then produce the gloss line." +
          (text ? `\nUser hint: ${text}` : ""),
      },
    ];
  } else {
    userContent = `${GLOSS_INSTRUCTIONS}\n\nInput: ${text}`;
  }

  const body = {
    model: "gemma",
    messages: [
      { role: "user", content: userContent },
    ],
    // budget_tokens activates Gemma 4's built-in thinking mode.
    // llama-cpp-python passes unknown fields through to the underlying
    // Llama object, so extra keys here are safe even on older builds.
    budget_tokens: 2048,
    // Thinking models need temperature >= 0.5 — below that the model
    // suppresses the reasoning chain and gives deterministic one-liners.
    temperature: 0.6,
    // Extra headroom for <think>…</think> tokens before the gloss line.
    max_tokens: 4096,
    stream: false,
  };

  console.log("[Gemma] → request body:", JSON.stringify(body, null, 2));
  const resp = await fetch(LLAMA_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const respText = await resp.text();
  console.log("[Gemma] ← HTTP", resp.status, "raw response:\n", respText);
  let data = null;
  try { data = JSON.parse(respText); } catch { /* leave as raw text */ }
  if (!resp.ok) {
    return { raw: "", cleaned: "", debug: respText, httpError: `HTTP ${resp.status}` };
  }
  const message  = data?.choices?.[0]?.message ?? {};
  const raw      = message.content ?? "";
  // Some llama.cpp builds put thinking in a separate field instead of inline.
  const reasoningField = message.reasoning_content ?? message.reasoning ?? "";

  // Strip <think>…</think> reasoning blocks the model may emit inline
  const afterThinking = raw.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  let cleaned = afterThinking.replace(/^["'`]+|["'`]+$/g, "").replace(/[.!?]+$/, "").trim();

  // Fallback: model returned empty / non-gloss content but the answer is in
  // its reasoning. Sources, in order:
  //   1. message.reasoning_content (separate API field)
  //   2. <think>…</think> inside content
  if (!cleaned) {
    const inlineThink = raw.match(/<think>([\s\S]*?)<\/think>/i)?.[1] ?? "";
    const reasoningText = reasoningField || inlineThink;
    if (reasoningText) {
      const recovered = extractGlossFromReasoning(reasoningText);
      if (recovered) {
        console.log("[Gemma] recovered gloss from reasoning:", recovered);
        cleaned = recovered;
      }
    }
  }

  console.log(
    "[Gemma] content:", raw,
    "→ reasoning_content len:", reasoningField.length,
    "→ after-think:", afterThinking,
    "→ cleaned:", cleaned,
  );
  return { raw: afterThinking || cleaned, cleaned, debug: respText, httpError: null };
}

// Extract an ISL gloss from a reasoning block. The gloss is generally the
// LAST line, so we walk lines bottom-up; for each line we scan tokens
// right-to-left and keep the trailing run of consecutive ALL-CAPS word
// tokens. This handles things like:
//   "So the final answer is: TONIGHT DINNER AFTER TABLET ONE TAKE PLEASE"
//   "Final: **TONIGHT DINNER AFTER TABLET ONE TAKE PLEASE**"
//   plain "TONIGHT DINNER AFTER TABLET ONE TAKE PLEASE"
function extractGlossFromReasoning(text) {
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  for (let li = lines.length - 1; li >= 0; li--) {
    const tokens = lines[li].split(/\s+/).filter(Boolean);
    const out = [];
    for (let ti = tokens.length - 1; ti >= 0; ti--) {
      // Strip surrounding punctuation/markdown (*, **, ", ', :, etc.)
      const t = tokens[ti].replace(/^[^\w]+|[^\w]+$/g, "");
      if (/^[A-Z][A-Z_]+$/.test(t)) {
        out.unshift(t);
      } else if (out.length > 0) {
        break; // run ended on this line
      }
    }
    if (out.length >= 1) return out.join(" ");
  }
  return "";
}

// ─── WAV encoder (PCM Float32 → 16-bit WAV ArrayBuffer) ──────────────────────
function encodeWav(samples, sampleRate) {
  const dataSize = samples.length * 2;
  const buf = new ArrayBuffer(44 + dataSize);
  const v = new DataView(buf);
  const ws = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  ws(0, "RIFF"); v.setUint32(4, 36 + dataSize, true);
  ws(8, "WAVE"); ws(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true); v.setUint16(32, 2, true);
  v.setUint16(34, 16, true); ws(36, "data"); v.setUint32(40, dataSize, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buf;
}

function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

// ─── Custom audio player ─────────────────────────────────────────────────────
function AudioPlayer({ src }) {
  const audioRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [current, setCurrent] = useState(0);
  const [duration, setDuration] = useState(0);

  const toggle = () => {
    const a = audioRef.current;
    if (!a) return;
    playing ? a.pause() : a.play();
    setPlaying(!playing);
  };

  const fmt = (s) =>
    `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  const progress = duration ? current / duration : 0;

  return (
    <div className="audio-player">
      <audio
        ref={audioRef}
        src={src}
        onTimeUpdate={(e) => setCurrent(e.target.currentTime)}
        onDurationChange={(e) => setDuration(e.target.duration)}
        onEnded={() => { setPlaying(false); setCurrent(0); }}
      />
      <button className="ap-btn" onClick={toggle} aria-label={playing ? "Pause" : "Play"}>
        {playing ? <PauseIcon size={16} /> : <PlayIcon size={16} />}
      </button>
      <div className="ap-track">
        <div className="ap-bar-wrap">
          <div className="ap-bar-fill" style={{ width: `${progress * 100}%` }} />
          <input
            type="range" min={0} max={duration || 1} step={0.01} value={current}
            className="ap-scrubber"
            onChange={(e) => {
              const t = +e.target.value;
              audioRef.current.currentTime = t;
              setCurrent(t);
            }}
          />
        </div>
        <div className="ap-time">{fmt(current)} / {fmt(duration)}</div>
      </div>
    </div>
  );
}

const SIGN_LANGUAGES = [
  { code: "ISL", label: "ISL", full: "Indian Sign Language",    available: true  },
  { code: "ASL", label: "ASL", full: "American Sign Language",  available: false },
  { code: "BSL", label: "BSL", full: "British Sign Language",   available: false },
  { code: "Auslan", label: "Auslan", full: "Australian Sign Language", available: false },
];

// ─── Topbar ───────────────────────────────────────────────────────────────────
function Topbar({ status }) {
  const modelOnline = status !== "error";
  const [langOpen, setLangOpen] = useState(false);
  const [selectedLang, setSelectedLang] = useState("ISL");
  const [tooltipCode, setTooltipCode] = useState(null);
  const dropRef = useRef(null);

  // Close on outside click
  useEffect(() => {
    if (!langOpen) return;
    const onDown = (e) => {
      if (!dropRef.current?.contains(e.target)) setLangOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [langOpen]);

  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark">i</div>
        <div>
          <div className="brand-name">SignVerse</div>
          <div className="brand-sub">Sign Language assistant</div>
        </div>
      </div>
      <div className="topbar-actions">
        {/* Language selector */}
        <div className="lang-selector" ref={dropRef}>
          <button
            className="pill-btn lang-pill"
            onClick={() => setLangOpen((o) => !o)}
            aria-haspopup="listbox"
            aria-expanded={langOpen}
          >
            <span className="lang-flag">🤟</span>
            {selectedLang}
            <ChevDownIcon />
          </button>

          {langOpen && (
            <div className="lang-dropdown" role="listbox">
              {SIGN_LANGUAGES.map((lang) => (
                <div key={lang.code} className="lang-option-wrap">
                  <button
                    role="option"
                    aria-selected={lang.code === selectedLang}
                    className={`lang-option${lang.code === selectedLang ? " selected" : ""}${!lang.available ? " disabled" : ""}`}
                    onClick={() => {
                      if (!lang.available) return;
                      setSelectedLang(lang.code);
                      setLangOpen(false);
                    }}
                    onMouseEnter={() => !lang.available && setTooltipCode(lang.code)}
                    onMouseLeave={() => setTooltipCode(null)}
                  >
                    <span className="lang-code">{lang.label}</span>
                    <span className="lang-full">{lang.full}</span>
                    {!lang.available && <span className="lang-soon">Soon</span>}
                    {lang.code === selectedLang && (
                      <svg className="lang-check" width="13" height="13" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12" />
                      </svg>
                    )}
                  </button>
                  {tooltipCode === lang.code && (
                    <div className="lang-tooltip">Coming soon</div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        <button className="pill-btn">
          <span
            className="dot"
            style={{ background: modelOnline ? "var(--ok)" : "var(--warn)" }}
          />
          {modelOnline ? "Model online" : "Model offline"}
        </button>

        <button className="pill-btn" aria-label="Settings">
          <SettingsIcon />
          Settings
        </button>
      </div>
    </header>
  );
}

// ─── Gloss output card (shows model response) ────────────────────────────────
function GlossCard({
  status, matched, skipped, activeWordIdx,
  gloss, error, modelDebug,
  onRetry, onDictPreview,
}) {
  const [dictOpen, setDictOpen] = useState(false);
  const glossCardState =
    status === "loading" ? "loading" : error ? "error" : "ready";

  const glossStatusText = (() => {
    if (status === "loading") return "Avatar signing PLEASE WAIT…";
    if (error) return "Translation failed";
    if (activeWordIdx >= 0 && matched.length > 0)
      return `Now signing · word ${activeWordIdx + 1} of ${matched.length}`;
    if (matched.length > 0) return "Signing…";
    return "Ready";
  })();

  return (
    <div className="gloss-card" data-state={glossCardState}>
      <div className="gloss-head">
        <span className="gloss-label">ISL Gloss · model output</span>
        <span className="gloss-status">
          <span className="pulse" />
          {glossStatusText}
        </span>
      </div>

      {/* Ready state */}
      <div className="gloss-state state-ready">
        <div className="gloss-display">
          {matched.length > 0
            ? matched.map((w) => w.toUpperCase()).join(" ")
            : <span className="placeholder">Type below and press Sign it</span>
          }
        </div>
        {matched.length > 0 && (
          <>
            <div className="gloss-chips">
              {matched.map((w, i) => (
                <span
                  key={`${w}-${i}`}
                  className={`chip${i === activeWordIdx ? " active" : ""}`}
                >
                  {w.toUpperCase()}
                </span>
              ))}
              {skipped.map((w, i) => (
                <span key={`s-${w}-${i}`} className="chip skipped">{w}</span>
              ))}
            </div>
            <div className="gloss-meta">
              <span><b>{matched.length}</b> matched</span>
              <span><b>{skipped.length}</b> skipped</span>
              <span>· model: <b>gemma-isl</b></span>
            </div>
          </>
        )}
      </div>

      {/* Loading state */}
      <div className="gloss-state state-loading">
        <div className="loading-row">
          <span className="loading-spinner" />
          <div>
            <div className="loading-title">Translating with Gemma…</div>
            <div className="loading-sub">
              Avatar is signing <b>PLEASE WAIT</b> while the model thinks.
            </div>
          </div>
        </div>
        <div className="skeleton-line w-80" />
        <div className="skeleton-chips">
          <span /><span /><span />
        </div>
      </div>

      {/* Error state */}
      <div className="gloss-state state-error">
        {gloss && (
          <div className="gloss-display" style={{ marginBottom: 2 }}>
            {gloss.toUpperCase()}
          </div>
        )}
        <div className="error-row">
          <span className="error-icon">!</span>
          <div>
            <div className="error-title">
              {error === "no_dict_match"
                ? "No words matched the dictionary"
                : "Model returned no usable gloss"}
            </div>
            <div className="error-sub">
              {error === "no_dict_match"
                ? "The model produced the gloss above but none of the words are in the dictionary. "
                : error
                ? error.replace(/\.$/, "") + ". "
                : "None of the words matched the dictionary. "}
              Avatar is signing <b>PROBLEM</b>. Try a phrase using the available words below.
            </div>
          </div>
        </div>
        <div className="error-actions">
          <button className="error-btn primary" onClick={onRetry}>↻ Retry</button>
          <button
            className="error-btn"
            onClick={() => {
              setDictOpen(true);
              setTimeout(() => {
                document.getElementById("dictPanel")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
              }, 50);
            }}
          >
            View dictionary
          </button>
        </div>
      </div>

      {/* Footer: dict toggle + raw HTTP */}
      <div className="gloss-footer">
        <button
          className="link-btn"
          onClick={() => setDictOpen((o) => !o)}
          aria-expanded={dictOpen}
          aria-controls="dictPanel"
        >
          <BookIcon />
          Available words
          <span className="pill-count">{DICTIONARY.length}</span>
          <span className="chev" style={{ transform: dictOpen ? "rotate(180deg)" : "none" }}>
            <ChevDownIcon />
          </span>
        </button>
        {modelDebug && (
          <details className="raw-resp">
            <summary>Raw HTTP</summary>
            <pre>{modelDebug}</pre>
          </details>
        )}
      </div>

      {/* Dictionary panel */}
      {dictOpen && (
        <div className="dict-panel" id="dictPanel">
          <div className="dict-head">
            <span className="dict-title">Words I can sign</span>
            <span className="dict-hint">
              Phrases must use these — model maps to closest match
            </span>
          </div>
          <div className="dict-grid">
            {DICTIONARY.map((word) => (
              <button
                key={word}
                className="dict-word"
                title={`Preview "${word}" sign`}
                onClick={() => onDictPreview(word)}
              >
                <span className="play-ico">▶</span>
                {word.toUpperCase()}
              </button>
            ))}
          </div>
          <div className="dict-foot">
            <span className="dict-hint">
              Try: "<b>You</b> have a <b>fever</b>" · "<b>What</b> is <b>your name</b>" · "<b>Caution</b> · <b>wet floor</b>"
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

function fmtRelativeTime(ts, now) {
  const diff = (now - ts) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return new Date(ts).toLocaleDateString();
}

// ─── History panel (recent local inputs) ─────────────────────────────────────
function HistoryPanel({ history, onPick, onClear }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 30000);
    return () => clearInterval(id);
  }, []);

  const kindIcon = (entry) => {
    if (entry.kind === "image") return <ImageIcon />;
    if (entry.kind === "audio") return <MicIcon size={13} />;
    return <TextIcon />;
  };

  return (
    <div className="history-panel">
      <div className="history-head">
        <span className="history-label">History · this session</span>
        {history.length > 0 && (
          <button className="link-btn" onClick={onClear} title="Clear history">
            <TrashIcon />
            Clear
          </button>
        )}
      </div>
      {history.length === 0 ? (
        <div className="history-empty">
          No history yet. Your inputs will appear here.
        </div>
      ) : (
        <ul className="history-list">
          {history.map((entry) => (
            <li key={entry.id}>
              <button
                className="history-item"
                onClick={() => onPick(entry)}
                title="Re-run this input"
              >
                <span className="history-icon">{kindIcon(entry)}</span>
                <span className="history-body">
                  <span className="history-text">
                    {entry.text || (entry.kind === "image" ? "Image input" : entry.kind === "audio" ? "Voice input" : "—")}
                  </span>
                  {entry.gloss && (
                    <span className="history-gloss">{entry.gloss.toUpperCase()}</span>
                  )}
                </span>
                <span className="history-time">{fmtRelativeTime(entry.ts, now)}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// ─── Unified input bar (replaces desktop input card + mobile bar) ────────────
function InputBar({ onPlay, status, pendingText, onConsumePendingText }) {
  const [text, setText] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [recording, setRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [audioDataUrl, setAudioDataUrl] = useState(null);
  const audioNodesRef = useRef(null);
  const samplesRef = useRef([]);
  const fileInpRef = useRef(null);
  const inputRef = useRef(null);

  const busy = status === "loading";

  // Allow parent (e.g. history click) to pre-fill the text box.
  useEffect(() => {
    if (pendingText != null) {
      setText(pendingText);
      onConsumePendingText?.();
      inputRef.current?.focus();
    }
  }, [pendingText, onConsumePendingText]);

  const handleImageFile = async (file) => {
    if (!file) { setImageFile(null); setImagePreview(null); return; }
    setImageFile(file);
    const dataUrl = await fileToDataURL(file);
    setImagePreview(dataUrl);
  };

  const clearAttachments = () => {
    setImageFile(null);
    setImagePreview(null);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);
    setAudioDataUrl(null);
  };

  const submit = async () => {
    if (busy) return;
    if (!text.trim() && !imageFile && !audioDataUrl) return;
    let imageDataUrl = null;
    if (imageFile) imageDataUrl = await fileToDataURL(imageFile);
    onPlay({ text: text.trim(), imageDataUrl, audioDataUrl });
    setText("");
    clearAttachments();
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ctx = new AudioContext();
      const source = ctx.createMediaStreamSource(stream);
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      samplesRef.current = [];
      processor.onaudioprocess = (e) => {
        samplesRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      };
      source.connect(processor);
      processor.connect(ctx.destination);
      audioNodesRef.current = { ctx, source, processor, stream };
      setRecording(true);
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
      setAudioDataUrl(null);
    } catch {
      alert("Microphone access denied. Please allow microphone access in your browser.");
    }
  };

  const stopRecording = async () => {
    const nodes = audioNodesRef.current;
    if (!nodes) return;
    const { ctx, source, processor, stream } = nodes;
    source.disconnect();
    processor.disconnect();
    stream.getTracks().forEach((t) => t.stop());
    const sampleRate = ctx.sampleRate;
    await ctx.close();
    audioNodesRef.current = null;
    setRecording(false);

    const total = samplesRef.current.reduce((n, c) => n + c.length, 0);
    const merged = new Float32Array(total);
    let off = 0;
    for (const chunk of samplesRef.current) { merged.set(chunk, off); off += chunk.length; }
    const wavBuf = encodeWav(merged, sampleRate);
    const blob = new Blob([wavBuf], { type: "audio/wav" });
    const objUrl = URL.createObjectURL(blob);
    setAudioUrl(objUrl);
    const dataUrl = await fileToDataURL(blob);
    setAudioDataUrl(dataUrl);
  };

  // Paste handler — always-on. Captures images from clipboard.
  useEffect(() => {
    const onPaste = (e) => {
      // Don't intercept paste into other inputs/textareas elsewhere.
      const tag = e.target?.tagName;
      const isOtherField =
        (tag === "INPUT" && e.target !== inputRef.current) || tag === "TEXTAREA";
      const items = [...(e.clipboardData?.items ?? [])];
      const imgItem = items.find((i) => i.type.startsWith("image/"));
      if (imgItem) {
        e.preventDefault();
        handleImageFile(imgItem.getAsFile());
        return;
      }
      if (isOtherField) return;
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, []);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const hasAttachment = imageFile || audioUrl;

  return (
    <div className="input-bar" role="group" aria-label="Compose ISL input">
      {hasAttachment && (
        <div className="input-attachments">
          {imageFile && (
            <div className="attachment-chip">
              {imagePreview && <img src={imagePreview} alt="" className="attachment-thumb" />}
              <span className="attachment-name">{imageFile.name}</span>
              <button
                type="button"
                className="attachment-clear"
                onClick={() => handleImageFile(null)}
                aria-label="Remove image"
              >×</button>
            </div>
          )}
          {audioUrl && (
            <div className="attachment-chip audio">
              <MicIcon size={13} />
              <AudioPlayer src={audioUrl} />
              <button
                type="button"
                className="attachment-clear"
                onClick={() => {
                  if (audioUrl) URL.revokeObjectURL(audioUrl);
                  setAudioUrl(null);
                  setAudioDataUrl(null);
                }}
                aria-label="Remove audio"
              >×</button>
            </div>
          )}
        </div>
      )}

      <div className="input-bar-row">
        <button
          type="button"
          className="ib-btn plus-btn"
          onClick={() => fileInpRef.current?.click()}
          aria-label="Attach image"
          disabled={busy}
          title="Attach image (or paste with Ctrl+V)"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 5v14M5 12h14" />
          </svg>
        </button>
        <input
          ref={fileInpRef}
          type="file"
          accept="image/*"
          className="sr-only"
          onChange={(e) => handleImageFile(e.target.files?.[0] ?? null)}
        />

        <input
          ref={inputRef}
          className="ib-input"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            recording
              ? "Listening… tap mic again to stop"
              : "Type a message, paste an image, or tap mic"
          }
          disabled={busy || recording}
          maxLength={500}
        />

        <div className="mic-wrap">
          <button
            type="button"
            className={`ib-btn mic-btn${recording ? " rec" : ""}`}
            onClick={recording ? stopRecording : startRecording}
            disabled={busy}
            aria-label={recording ? "Click again to stop recording" : "Record audio"}
            title={recording ? "Click again to stop recording" : "Click to record audio"}
          >
            <MicIcon size={18} />
          </button>
          {recording && (
            <span className="mic-hint">Tap to stop</span>
          )}
          <span className="mic-tooltip">
            {recording ? "Click again to stop recording" : "Click to record audio"}
          </span>
        </div>

        <button
          type="button"
          className="ib-send"
          onClick={submit}
          disabled={busy || (!text.trim() && !imageFile && !audioDataUrl)}
        >
          {busy ? <span className="spin" /> : <PlayIcon size={14} />}
          <span className="ib-send-label">Sign it</span>
        </button>
      </div>
    </div>
  );
}

// ─── Local + backend session history ─────────────────────────────────────────
const HISTORY_KEY = "isl-history";
const SESSION_ID_KEY = "isl-session-id";
const HISTORY_LIMIT = 30;

function getOrCreateSessionId() {
  let id = localStorage.getItem(SESSION_ID_KEY);
  if (!id) {
    id =
      (crypto?.randomUUID?.() ??
        `s-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`);
    localStorage.setItem(SESSION_ID_KEY, id);
  }
  return id;
}

// Best-effort backend session save (fire-and-forget). The same Modal proxy
// that fronts llama.cpp now also accepts /session/history. If the endpoint
// is missing or the call fails, we silently fall back to localStorage only.
function saveHistoryToBackend(sessionId, entry) {
  try {
    // The proxy lives at the same origin as LLAMA_URL, but with /session/history.
    // We derive it from LLAMA_URL by stripping /v1/chat/completions.
    const base = LLAMA_URL.replace(/\/v1\/chat\/completions\/?$/, "");
    fetch(`${base}/session/history`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, entry }),
      keepalive: true,
    }).catch(() => {});
  } catch {
    /* ignore */
  }
}

// ─── App ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [frames, setFrames]           = useState([]);
  const [wordRanges, setWordRanges]   = useState([]);
  const [matched, setMatched]         = useState([]);
  const [skipped, setSkipped]         = useState([]);
  const [activeWordIdx, setActiveWordIdx] = useState(-1);
  const [status, setStatus]           = useState("idle");
  const [gloss, setGloss]             = useState("");
  const [modelDebug, setModelDebug]   = useState("");
  const [hasSubmitted, setHasSubmitted] = useState(false);
  const [error, setError]             = useState(null);
  const [isPlaying, setIsPlaying]     = useState(true);
  const [speed, setSpeed]             = useState(1.0);
  const [history, setHistory] = useState(() => {
    try { return JSON.parse(localStorage.getItem(HISTORY_KEY) ?? "[]"); }
    catch { return []; }
  });
  const [pendingText, setPendingText] = useState(null);

  const sessionIdRef    = useRef(getOrCreateSessionId());
  const controlRef      = useRef(null);
  const progressFillRef = useRef(null);
  const lastSubmitRef   = useRef({ text: "", imageDataUrl: null });

  useEffect(() => {
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, HISTORY_LIMIT))); }
    catch { /* quota exceeded — drop silently */ }
  }, [history]);

  // Default animation on mount
  useEffect(() => {
    loadClipFrames(["Hello"])
      .then(({ frames: f, wordRanges: wr }) => {
        setFrames(f);
        setWordRanges(wr);
        setMatched(["Hello"]);
      })
      .catch((e) => console.warn("Default clip load failed:", e.message));
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
      if (e.code === "Space") { e.preventDefault(); setIsPlaying((p) => !p); }
      if (e.code === "KeyR")  { setIsPlaying(true); controlRef.current?.jumpToWord(0); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const handlePlay = useCallback(async ({ text, imageDataUrl, audioDataUrl }) => {
    lastSubmitRef.current = { text, imageDataUrl, audioDataUrl };
    const entryId = `h-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const kind = imageDataUrl ? "image" : audioDataUrl ? "audio" : "text";
    const baseEntry = { id: entryId, ts: Date.now(), text, kind, gloss: "" };
    setHistory((h) => [baseEntry, ...h].slice(0, HISTORY_LIMIT));
    saveHistoryToBackend(sessionIdRef.current, baseEntry);

    setStatus("loading");
    setError(null);
    setGloss("");
    setModelDebug("");
    setMatched([]);
    setSkipped([]);
    setActiveWordIdx(-1);
    setHasSubmitted(true);
    setIsPlaying(true);

    // ── Start the LLM call IMMEDIATELY (parallel with clip loading) ────────
    const llmPromise = fetchGloss({ text, imageDataUrl, audioDataUrl });

    // ── Load and show Please → Wait while the model thinks ─────────────────
    let waitStartTime = 0;
    try {
      const { frames: waitF, wordRanges: waitWr } =
        await loadClipFrames(["Please", "Wait"]);
      setFrames(waitF);
      setWordRanges(waitWr);
      setMatched(["Please", "Wait"]);
      waitStartTime = Date.now();
    } catch (e) {
      console.warn("Please/Wait clip load failed:", e.message);
    }

    // Please(97) + transition(15) + Wait(97) frames at 25fps ≈ 8.4s.
    // Guarantee the avatar gets to play through both clips at least once.
    const MIN_WAIT_MS = 8400;
    const ensureMinPlay = async () => {
      if (waitStartTime === 0) return;
      const elapsed = Date.now() - waitStartTime;
      if (elapsed < MIN_WAIT_MS) {
        await new Promise((r) => setTimeout(r, MIN_WAIT_MS - elapsed));
      }
    };

    let debugText = "";
    try {
      const { cleaned, debug, httpError } = await llmPromise;
      const glossText = cleaned;
      debugText = debug || "";
      // Surface the raw HTTP response immediately so it's available for
      // debugging even if we throw below (httpError, no-match, clip load).
      setModelDebug(debugText);
      if (httpError) throw new Error(`llama.cpp ${httpError}`);
      // Always store the raw gloss so it's visible in the error state too.
      setGloss(glossText);
      const { matched: m, skipped: s } = glossToClipNames(glossText);
      if (m.length === 0) {
        throw new Error("no_dict_match");
      }
      const { frames: newFrames, wordRanges: newRanges } = await loadClipFrames(m);

      // Hold the result until Please→Wait has had its full play
      await ensureMinPlay();

      setGloss(glossText);
      setMatched(m);
      setSkipped(s);
      setActiveWordIdx(-1);
      setFrames(newFrames);
      setWordRanges(newRanges);
      setStatus("ready");

      // Record the resolved gloss back onto the history entry.
      setHistory((h) =>
        h.map((e) => (e.id === entryId ? { ...e, gloss: glossText } : e))
      );
      saveHistoryToBackend(sessionIdRef.current, {
        ...baseEntry,
        gloss: glossText,
      });
    } catch (e) {
      const errorMsg = e.message;
      let errF = null, errWr = null;
      try {
        const r = await loadClipFrames(["Problem"]);
        errF = r.frames;
        errWr = r.wordRanges;
      } catch (e2) {
        console.warn("Problem clip load failed:", e2.message);
      }

      await ensureMinPlay();

      setError(errorMsg);
      if (debugText) setModelDebug(debugText);
      setStatus("idle");
      if (errF) {
        setFrames(errF);
        setWordRanges(errWr);
        setMatched(["Problem"]);
        setSkipped([]);
        setActiveWordIdx(-1);
      }
    }
  }, []);

  const handleRetry = useCallback(() => {
    const last = lastSubmitRef.current;
    if (last.text || last.imageDataUrl || last.audioDataUrl) handlePlay(last);
  }, [handlePlay]);

  const handleDictPreview = useCallback(async (word) => {
    try {
      const { frames: f, wordRanges: wr } = await loadClipFrames([word]);
      setFrames(f);
      setWordRanges(wr);
      setMatched([word]);
      setSkipped([]);
      setActiveWordIdx(-1);
      setIsPlaying(true);
    } catch (e) {
      console.warn("Dict preview failed:", e.message);
    }
  }, []);

  const handleProgress = useCallback((ratio) => {
    if (progressFillRef.current) {
      progressFillRef.current.style.transform = `scaleX(${ratio})`;
    }
  }, []);

  const nowSigningWord = (() => {
    if (status === "loading") return "PLEASE WAIT";
    if (error)               return "PROBLEM";
    if (activeWordIdx >= 0 && matched[activeWordIdx]) return matched[activeWordIdx].toUpperCase();
    if (matched.length > 0) return matched[0].toUpperCase();
    return "—";
  })();

  const totalSecs = matched.length > 0
    ? Math.round(frames.length / FPS)
    : 0;
  const formatTime = (s) =>
    `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

  return (
    <>
      <div className="bg-noise" aria-hidden="true" />
      <div className="shell">
        <Topbar status={status} />

        <main className="main">
          {/* ── Left: Gloss output (top) + History (bottom) ── */}
          <section className="col-left">
            <GlossCard
              status={status}
              matched={matched}
              skipped={skipped}
              activeWordIdx={activeWordIdx}
              gloss={gloss}
              error={error}
              modelDebug={modelDebug}
              onRetry={handleRetry}
              onDictPreview={handleDictPreview}
            />
            <HistoryPanel
              history={history}
              onPick={(entry) => {
                if (entry.text) setPendingText(entry.text);
              }}
              onClear={() => setHistory([])}
            />
          </section>

          {/* ── Right: Stage + Input bar below ── */}
          <section className="col-stage">
            <div className="stage">
              {/* Stage head */}
              <div className="stage-head">
                <div className="now-signing">
                  <span className="live-dot">●</span>
                  <div>
                    <div className="lbl">Now signing</div>
                    <div className="word">{nowSigningWord}</div>
                  </div>
                </div>
                <div className="stage-tools">
                  <button className="stage-tool" title="Captions"><CaptionsIcon /></button>
                  <button className="stage-tool" title="Fullscreen"><FullscreenIcon /></button>
                </div>
                <div className="mobile-brand-badge mobile-only">
                  <div className="mobile-brand-mark">i</div>
                  <span className="mobile-brand-name">SignVerse</span>
                </div>
              </div>

              {/* Avatar canvas */}
              <div className="stage-canvas">
                <div className="avatar-stage">
                  <div className="avatar-shadow" />
                  <Canvas
                    camera={{ position: [0, 1.4, 3], fov: 35, near: 0.01, far: 100 }}
                    gl={{
                      antialias: true,
                      alpha: true,
                      toneMapping: THREE.ACESFilmicToneMapping,
                    }}
                    style={{ width: "100%", height: "100%" }}
                  >
                    <Lights />
                    <Suspense fallback={null}>
                      {frames.length > 0 && (
                        <VRMScene
                          frames={frames}
                          wordRanges={wordRanges}
                          onWordChange={setActiveWordIdx}
                          playing={isPlaying}
                          speed={speed}
                          onProgress={handleProgress}
                          controlRef={controlRef}
                        />
                      )}
                    </Suspense>
                  </Canvas>
                </div>
              </div>

              {/* Progress bar */}
              <div className="stage-progress">
                <span className="time">00:00</span>
                <div className="progress-track">
                  <div className="progress-marks">
                    {matched.map((_, i) => (
                      <div key={i} className="seg" />
                    ))}
                  </div>
                  <div className="progress-fill" ref={progressFillRef} />
                </div>
                <span className="time">{formatTime(totalSecs)}</span>
              </div>

              {/* Stage controls */}
              <div className="stage-controls">
                <button
                  className="stage-ctrl"
                  title="Previous word"
                  onClick={() => {
                    const idx = Math.max(0, activeWordIdx - 1);
                    controlRef.current?.jumpToWord(idx);
                  }}
                >
                  <PrevIcon />
                </button>

                <button
                  className="stage-ctrl play"
                  onClick={() => setIsPlaying((p) => !p)}
                >
                  {isPlaying ? <><PauseIcon /> Pause</> : <><PlayIcon /> Play</>}
                </button>

                <button
                  className="stage-ctrl"
                  title="Replay"
                  onClick={() => { setIsPlaying(true); controlRef.current?.jumpToWord(0); }}
                >
                  <ReplayIcon />
                </button>

                <button
                  className="stage-ctrl"
                  title="Next word"
                  onClick={() => {
                    const idx = Math.min(matched.length - 1, activeWordIdx + 1);
                    controlRef.current?.jumpToWord(idx);
                  }}
                >
                  <NextIcon />
                </button>

                <span className="ctrl-div" />

                <div className="speed-toggle" role="group" aria-label="Playback speed">
                  {[0.5, 1.0, 1.5].map((s) => (
                    <button
                      key={s}
                      className={speed === s ? "on" : ""}
                      onClick={() => setSpeed(s)}
                    >
                      {s === 1.0 ? "1.0×" : `${s}×`}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Input bar (below avatar) */}
            <InputBar
              onPlay={handlePlay}
              status={status}
              pendingText={pendingText}
              onConsumePendingText={() => setPendingText(null)}
            />

            {/* Stage footer */}
            <div className="stage-footer">
              <span>
                Press <span className="kbd">⏎</span> to sign ·{" "}
                <span className="kbd">␣</span> to pause ·{" "}
                <span className="kbd">R</span> to replay
              </span>
              <span style={{ marginLeft: "auto" }}>
                Avatar: <b style={{ color: "var(--ink-2)" }}>remmy.vrm</b> · 25 fps
              </span>
            </div>
          </section>
        </main>
      </div>
    </>
  );
}
