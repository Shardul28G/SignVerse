# ISL Translator

Speech / text / image → Indian Sign Language gloss → VRM avatar animation.

```
.
├── frontend/                # Vite + React app, deployed to Vercel
│   ├── public/
│   │   └── landmarks/       # 4 always-bundled fallbacks (Hello, Wait, Problem, Please)
│   ├── src/
│   ├── index.html
│   ├── vite.config.js       # dev middleware serves ../data_tools/landmarks
│   └── package.json
├── backend/                 # Modal.com llama.cpp server, T4 GPU
│   ├── app.py               # base.gguf + lora.gguf + mmproj.gguf  →  /v1/chat/completions
│   ├── requirements.txt     # local: just `modal` (image deps live inside app.py)
│   └── *.gguf               # NOT in git — you drop them here, then push to a Modal Volume
└── data_tools/              # local-only utilities (not deployed anywhere)
    ├── landmarks/           # NOT in git — all landmark JSONs except the 4 bundled ones
    └── upload_to_r2.py      # boto3 → Cloudflare R2
```

## Local development

You can develop with everything local, no R2 / no Modal needed:

```bash
cd frontend
npm install
npm run dev
```

Vite serves:
- `frontend/public/landmarks/*.json` (the 4 bundled fallbacks) at `/landmarks/*.json`
- `data_tools/landmarks/*.json` (everything else) at `/landmarks/*.json`
  via a small dev-only middleware in `vite.config.js`.

So `fetch('/landmarks/Mumbai.json')` works locally without ever touching R2.

For the LLM, run `llama-server` locally on port 8080 as before — `vite.config.js`
proxies `/api/llama/*` to `http://127.0.0.1:8080/*`.

## Production wiring

Set these environment variables in **Vercel → Project Settings → Environment Variables**:

| Variable                    | Example                                                                          |
| --------------------------- | -------------------------------------------------------------------------------- |
| `VITE_LANDMARKS_BASE_URL`   | `https://pub-xxxxxxxx.r2.dev` (your R2 public URL — no trailing slash)           |
| `VITE_LLAMA_URL`            | `https://your-workspace--isl-llama-server-serve.modal.run/v1/chat/completions`   |

In production the frontend calls:
- `${VITE_LANDMARKS_BASE_URL}/Mumbai.json` for non-bundled landmarks
- `/landmarks/Hello.json` (and the other 3) for bundled ones — these ship with the
  Vercel build from `frontend/public/landmarks/`
- `${VITE_LLAMA_URL}` for chat completions

## One-time data upload (R2)

```bash
pip install boto3 python-dotenv
# Put R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_BUCKET in
# data_tools/.env  (git-ignored)
python data_tools/upload_to_r2.py
```

The script skips the four files that ship with the frontend.

## One-time model upload (Modal)

```bash
pip install modal
modal setup

# Drop base.gguf / lora.gguf / mmproj.gguf into backend/, then:
modal volume create isl-models
modal volume put isl-models backend/base.gguf   /base.gguf
modal volume put isl-models backend/lora.gguf   /lora.gguf
modal volume put isl-models backend/mmproj.gguf /mmproj.gguf

modal deploy backend/app.py
```

Modal prints the public URL. Append `/v1/chat/completions` and set it on Vercel
as `VITE_LLAMA_URL`.

## Adding a new sign

1. Drop `<NewWord>.json` into `data_tools/landmarks/`.
2. Add `"NewWord"` to the `DICTIONARY` array in `frontend/src/App.jsx`.
3. Local: `npm run dev` — works immediately.
4. Production: `python data_tools/upload_to_r2.py` to push the new file to R2.
   No redeploy needed.
