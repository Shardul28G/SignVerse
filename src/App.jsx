/**
 * Avatar viewer driven by the custom retargeting pipeline.
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

const BG_COLOR = "#e8ecf2";
const FPS = 25;
const FRAME_TIME = 1 / FPS;
const VRM_URL = "./anim.vrm";

const LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions";

// Words available in /public as <Name>.json. Multi-word entries (longer first)
// are tried before single words so phrases like "What Is Your Name" prefer the
// dedicated clip over individual signs.
const DICTIONARY = [
  "Caution",
  "Fever",
  "Floor",
  "Name",
  "Please",
  "Wet",
  "What",
  "You",
  "Your",
  "hospital",
];

const SYSTEM_PROMPT =
  "You are an Indian Sign Language (ISL) gloss generator. " +
  "Convert the user's input into an ISL gloss line. " +
  "ISL gloss is a sequence of uppercase keyword tokens in topic-comment order, " +
  "with no articles, no auxiliary verbs, and no punctuation. " +
  "Reply with ONLY the gloss line — no quotes, no explanation, no extra text.";

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
        setVrm(model);
      },
      undefined,
      (err) => console.error("VRM load error:", err),
    );
  }, [url]);
  return vrm;
}

// ─── Auto-fit camera around the avatar bounding box ──────────────────────────
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
    camera.far  = zoomDist * 100;
    camera.lookAt(center.x, upperY, center.z);
    camera.updateProjectionMatrix();
  }, [object, camera, size, padding, extraWidth]);
  return null;
}

// ─── Animated VRM scene ──────────────────────────────────────────────────────
function VRMScene({ frames, wordRanges, onWordChange }) {
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

  // Reset playback to frame 0 whenever a new clip sequence is loaded.
  useEffect(() => {
    playback.current.idx = 0;
    playback.current.accumulator = 0;
    lastWordIdx.current = -1;
  }, [frames]);

  useFrame((_, delta) => {
    if (!vrm || !frames.length || !pipelineRef.current) return;
    const pb = playback.current;
    pb.accumulator += delta;
    while (pb.accumulator >= FRAME_TIME) {
      pb.accumulator -= FRAME_TIME;
      pb.idx = (pb.idx + 1) % frames.length;
    }
    const t = performance.now() / 1000;
    pipelineRef.current.step(frames[pb.idx], t);

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
      {vrm && <FitCameraToObject object={vrm.scene} padding={1.6} />}
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

// ─── Frame interpolation for clip transitions ────────────────────────────────
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
        visibility: pt.visibility !== undefined && pb.visibility !== undefined
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

// ─── Gloss → list of available dictionary clip names ─────────────────────────
// Greedy longest-match: scan tokens left-to-right and try the longest available
// phrase first (e.g. "What Is Your Name" → "What_Is_Your_Name").
function glossToClipNames(gloss) {
  const tokens = gloss
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean)
    .map((t) => t.toLowerCase());

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

// ─── Load + concat landmark JSONs with smooth transitions ────────────────────
// Returns { frames, wordRanges } where wordRanges[i] = { name, start, end }
// gives the half-open frame range belonging to clipNames[i] in the combined
// timeline (transition frames are attributed to the *next* word).
async function loadClipFrames(clipNames) {
  const TRANSITION_FRAMES = 15;
  const datas = await Promise.all(
    clipNames.map((name) =>
      fetch(`./${name}.json`).then((r) => {
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

// ─── llama.cpp call: text-only or multimodal (image) ─────────────────────────
async function fetchGloss({ text, imageDataUrl }) {
  const userContent = imageDataUrl
    ? [
        { type: "text", text: text || "Describe this image, then convert to ISL gloss." },
        { type: "image_url", image_url: { url: imageDataUrl } },
      ]
    : text;

  const body = {
    model: "gemma",
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userContent },
    ],
    temperature: 0.1,
    max_tokens: 2048,
    stream: false,
  };

  const resp = await fetch(LLAMA_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const respText = await resp.text();
  let data = null;
  try { data = JSON.parse(respText); } catch { /* leave as raw text */ }
  if (!resp.ok) {
    return { raw: "", cleaned: "", debug: respText, httpError: `HTTP ${resp.status}` };
  }
  const raw = data?.choices?.[0]?.message?.content ?? "";
  // Strip surrounding quotes/whitespace and any trailing punctuation.
  const cleaned = raw.trim().replace(/^["'`]+|["'`]+$/g, "").replace(/[.!?]+$/, "").trim();
  return { raw, cleaned, debug: respText, httpError: null };
}

// ─── Browser speech recognition (no server roundtrip) ────────────────────────
function getSpeechRecognition() {
  const Ctor = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Ctor) return null;
  const rec = new Ctor();
  rec.continuous = false;
  rec.interimResults = false;
  rec.lang = "en-IN";
  return rec;
}

function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

// ─── Input panel ─────────────────────────────────────────────────────────────
function InputPanel({ onPlay, status, gloss, modelRaw, modelDebug, hasSubmitted, matched, skipped, activeWordIdx, error }) {
  const [text, setText] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [recording, setRecording] = useState(false);
  const recRef = useRef(null);

  const submit = async () => {
    if (!text.trim() && !imageFile) return;
    let imageDataUrl = null;
    if (imageFile) imageDataUrl = await fileToDataURL(imageFile);
    onPlay({ text: text.trim(), imageDataUrl });
  };

  const startRecording = () => {
    const rec = getSpeechRecognition();
    if (!rec) {
      alert("SpeechRecognition is not supported in this browser. Try Chrome.");
      return;
    }
    recRef.current = rec;
    setRecording(true);
    rec.onresult = (e) => {
      const transcript = e.results[0][0].transcript;
      setText((prev) => (prev ? `${prev} ${transcript}` : transcript));
    };
    rec.onend = () => setRecording(false);
    rec.onerror = () => setRecording(false);
    rec.start();
  };

  const stopRecording = () => {
    recRef.current?.stop();
    setRecording(false);
  };

  const busy = status === "loading";

  return (
    <div style={panelStyle}>
      <div style={{ fontWeight: 600, marginBottom: 6 }}>Speak / type / show me something</div>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="e.g. What is your name?"
        rows={2}
        style={textareaStyle}
        disabled={busy}
      />
      <div style={{ display: "flex", gap: 8, marginTop: 8, flexWrap: "wrap", justifyContent: "center" }}>
        <label style={btnStyle(busy)}>
          {imageFile ? `📷 ${imageFile.name}` : "📷 Image"}
          <input
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={(e) => setImageFile(e.target.files?.[0] ?? null)}
            disabled={busy}
          />
        </label>
        {!recording ? (
          <button style={btnStyle(busy)} onClick={startRecording} disabled={busy}>🎤 Record</button>
        ) : (
          <button style={{ ...btnStyle(false), background: "#cc3344", color: "#fff" }} onClick={stopRecording}>
            ■ Stop
          </button>
        )}
        <button
          style={{ ...btnStyle(busy), background: "#2855aa", color: "#fff" }}
          onClick={submit}
          disabled={busy || (!text.trim() && !imageFile)}
        >
          {busy ? "Working…" : "▶ Sign it"}
        </button>
        {imageFile && (
          <button style={btnStyle(busy)} onClick={() => setImageFile(null)} disabled={busy}>
            Clear image
          </button>
        )}
      </div>
      {hasSubmitted && (
        <div style={glossBoxStyle}>
          <div style={{ fontSize: 11, color: "#5a6478", textTransform: "uppercase", letterSpacing: 0.6, marginBottom: 4 }}>
            ISL Gloss (model output)
          </div>
          <div style={{ fontFamily: "ui-monospace, Menlo, Consolas, monospace", fontSize: 18, fontWeight: 600, color: "#1a2030", letterSpacing: 1, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
            {gloss || modelRaw || <span style={{ color: "#aa2222" }}>(empty — model returned no content)</span>}
          </div>
          {modelRaw && gloss && modelRaw !== gloss && (
            <div style={{ marginTop: 4, fontSize: 11, color: "#5a6478" }}>
              raw: <span style={{ fontFamily: "monospace" }}>{JSON.stringify(modelRaw)}</span>
            </div>
          )}
          {modelDebug && (
            <details style={{ marginTop: 6, fontSize: 11, color: "#5a6478", textAlign: "left" }}>
              <summary style={{ cursor: "pointer", textAlign: "center" }}>raw HTTP response</summary>
              <pre style={{
                marginTop: 4, padding: 8, background: "#f7f8fb", border: "1px solid #dde2eb",
                borderRadius: 4, maxHeight: 180, overflow: "auto", whiteSpace: "pre-wrap",
                wordBreak: "break-word", fontSize: 11,
              }}>{modelDebug}</pre>
            </details>
          )}
          {matched && matched.length > 0 && (
            <div style={{ marginTop: 8, display: "flex", gap: 6, flexWrap: "wrap", justifyContent: "center" }}>
              {matched.map((w, i) => (
                <span
                  key={`${w}-${i}`}
                  style={{
                    padding: "3px 10px",
                    borderRadius: 999,
                    fontSize: 13,
                    fontWeight: 600,
                    fontFamily: "ui-monospace, Menlo, Consolas, monospace",
                    background: i === activeWordIdx ? "#2855aa" : "#dbe3f4",
                    color: i === activeWordIdx ? "#fff" : "#3a4a6b",
                    boxShadow: i === activeWordIdx ? "0 0 0 3px rgba(40,85,170,0.25)" : "none",
                    transition: "background 120ms, color 120ms, box-shadow 120ms",
                  }}
                >
                  {w}
                </span>
              ))}
            </div>
          )}
          {skipped && skipped.length > 0 && (
            <div style={{ marginTop: 6, fontSize: 12, color: "#8a6a1f" }}>
              Skipped (no clip): {skipped.join(", ")}
            </div>
          )}
        </div>
      )}
      {error && (
        <div style={{ marginTop: 6, fontSize: 12, color: "#aa2222" }}>⚠ {error}</div>
      )}
    </div>
  );
}

const panelStyle = {
  position: "absolute",
  top: 12,
  left: 12,
  width: "min(380px, calc(100vw - 24px))",
  maxHeight: "calc(100vh - 24px)",
  overflowY: "auto",
  background: "rgba(255,255,255,0.92)",
  backdropFilter: "blur(6px)",
  border: "1px solid #cdd2da",
  borderRadius: 10,
  padding: 12,
  fontFamily: "system-ui, sans-serif",
  fontSize: 14,
  boxShadow: "0 4px 16px rgba(0,0,0,0.08)",
  zIndex: 10,
  textAlign: "center",
};
const glossBoxStyle = {
  marginTop: 10,
  padding: "8px 12px",
  background: "#f0f4ff",
  border: "1px solid #b9c8ee",
  borderRadius: 8,
  textAlign: "center",
};
const textareaStyle = {
  width: "100%",
  resize: "vertical",
  border: "1px solid #cdd2da",
  borderRadius: 6,
  padding: 8,
  fontFamily: "inherit",
  fontSize: 14,
  boxSizing: "border-box",
};
const btnStyle = (disabled) => ({
  padding: "6px 12px",
  borderRadius: 6,
  border: "1px solid #cdd2da",
  background: "#f4f6fa",
  cursor: disabled ? "not-allowed" : "pointer",
  opacity: disabled ? 0.6 : 1,
  fontSize: 13,
  display: "inline-flex",
  alignItems: "center",
  gap: 4,
});

// ─── App ─────────────────────────────────────────────────────────────────────
export default function App() {
  const [frames, setFrames] = useState([]);
  const [wordRanges, setWordRanges] = useState([]);
  const [matched, setMatched] = useState([]);
  const [skipped, setSkipped] = useState([]);
  const [activeWordIdx, setActiveWordIdx] = useState(-1);
  const [status, setStatus] = useState("idle"); // idle | loading | ready
  const [gloss, setGloss] = useState("");      // cleaned gloss line
  const [modelRaw, setModelRaw] = useState(""); // raw LLM message content
  const [modelDebug, setModelDebug] = useState(""); // full HTTP response body
  const [hasSubmitted, setHasSubmitted] = useState(false);
  const [error, setError] = useState(null);

  // Load a default greeting clip on first mount so the avatar has something
  // to play before the user submits anything.
  useEffect(() => {
    loadClipFrames(["You", "Fever"])
      .then(({ frames: f, wordRanges: wr }) => {
        setFrames(f);
        setWordRanges(wr);
        setMatched(["You", "Fever"]);
      })
      .catch((e) => setError(e.message));
  }, []);

  const handlePlay = useCallback(async ({ text, imageDataUrl }) => {
    setStatus("loading");
    setError(null);
    setGloss("");
    setModelRaw("");
    setModelDebug("");
    setMatched([]);
    setSkipped([]);
    setActiveWordIdx(-1);
    setHasSubmitted(true);
    try {
      const { raw, cleaned, debug, httpError } = await fetchGloss({ text, imageDataUrl });
      // Always surface what the model returned, even if matching fails.
      setModelRaw(raw);
      setGloss(cleaned);
      setModelDebug(debug || "");
      if (httpError) throw new Error(`llama.cpp ${httpError} — see raw HTTP response below`);
      const { matched: m, skipped: s } = glossToClipNames(cleaned);
      setMatched(m);
      setSkipped(s);
      if (m.length === 0) {
        throw new Error(
          `No words in gloss matched the dictionary. Available: ${DICTIONARY.join(", ")}`,
        );
      }
      const { frames: newFrames, wordRanges: newRanges } = await loadClipFrames(m);
      setFrames(newFrames);
      setWordRanges(newRanges);
      setStatus("ready");
    } catch (e) {
      setError(e.message);
      setStatus("idle");
    }
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
          {frames.length > 0 && (
            <VRMScene
              frames={frames}
              wordRanges={wordRanges}
              onWordChange={setActiveWordIdx}
            />
          )}
        </Suspense>
      </Canvas>
      <InputPanel
        onPlay={handlePlay}
        status={status}
        gloss={gloss}
        modelRaw={modelRaw}
        modelDebug={modelDebug}
        hasSubmitted={hasSubmitted}
        matched={matched}
        skipped={skipped}
        activeWordIdx={activeWordIdx}
        error={error}
      />
    </div>
  );
}
