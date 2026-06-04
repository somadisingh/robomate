# 🤖 Robomate — Physical Data Marketplace for Robot Learning

Robomate is an end-to-end physical data marketplace and review studio designed to solve the robotics data bottleneck. It connects **labs** looking for real-world demonstration data with **collectors** who capture these physical-world demonstrations using their iPhones. 

Raw multi-modal demonstration recordings are uploaded directly from iOS, processed on cloud-based GPU pipelines to produce rich, robot-readable annotations (including **3D Gaussian Splats**, **MediaPipe joint tracking**, **YOLO object detection**, and **temporal action segments**), and verified by **AI evaluators (Gemini Vision)**.

---

## 🏗️ Repository Architecture

Robomate is structured as a lightweight monorepo:

```
robomate/
├── iosApp/              # SwiftUI Data Collector (ARKit, LiDAR, CoreMotion, GPS)
├── web/                 # Next.js 16 Marketplace & 3D Review Studio (Webpack dev server)
├── backend/             # Serverless Python backend on Modal for GPU-accelerated AI pipelines
├── supabase/            # Database schema migrations, RLS policies, and Edge Functions
├── playground/          # Sensor verification and ML playground experiments
└── docs/                # Architectural specs and pipeline design documents
```

---

## 🔄 End-to-End Data Pipeline

```mermaid
sequenceDiagram
    autonumber
    actor Lab as Lab User
    actor Collector as iPhone Collector
    participant Supabase as Supabase Storage/DB
    participant EdgeFn as Supabase Edge Function
    participant Modal as Modal GPU Cloud
    participant Gemini as Gemini Vision API

    Lab->>Supabase: 1. Create a task (e.g., "Plug in a laptop charger", prompts: ["laptop", "charger"])
    Collector->>Supabase: 2. View open tasks on iPhone app
    Collector->>Collector: 3. Record physical demo (video + LiDAR + camera pose + IMU)
    Collector->>Supabase: 4. Stream upload raw data bundle to 'recordings' bucket (signed URL)
    Collector->>EdgeFn: 5. Finalize upload (trigger Edge Function)
    EdgeFn->>Supabase: 6. Create recording / submission rows & seed analysis jobs in DB
    EdgeFn-->>Modal: 7. Kick off pipeline (authenticated HTTP trigger)
    par Parallel GPU Analysis on Modal
        Modal->>Gemini: 8. Ask Gemini to score video against task (0-10, pass/fail)
        Modal->>Modal: 9. Run YOLO to detect & track prompt objects (e.g., laptop, charger)
        Modal->>Modal: 10. Run MediaPipe to track 3D hand joints & knuckles
        Modal->>Modal: 11. Segment temporal phases ("reaching", "grasping", "plugging")
        Modal->>Modal: 12. Train 3D Gaussian Splat (7k iterations) using LiDAR depth + pose
    end
    Modal->>Supabase: 13. Upload analysis files (manifest.json, splat.spz, tracking JSONs)
    Modal->>Supabase: 14. Write Gemini scores, summaries, and success/fail flags to DB
    Lab->>web/: 15. View beautiful interactive 3D review studio & download files for robot training
```

---

## 🚀 Getting Started

### Prerequisites

- **macOS** with Node.js 18+ installed.
- **Python 3.12** with `uv` or `pip` package manager.
- **Xcode 15+** (for iOS compilation) and **XcodeGen** (`brew install xcodegen`).
- **Supabase CLI** (for local development or migrations).
- **Modal Account** (for serverless GPU deployments).

---

### 1. Web Frontend (Next.js)

The web client serves as the marketplace front, lab dashboard, and 3D visualization studio.

```bash
cd web
npm install

# Clear quarantine if Gatekeeper blocks native bundles on macOS:
xattr -dr com.apple.quarantine node_modules

# Run the development server (Webpack fallback port 3001)
npm run dev:next -- --webpack --port 3001
```

- **URL**: `http://localhost:3001`
- **Collector Test Logins**: `test-collector@example.com` / `TestCollector123!` or `flints.freaky_8c@icloud.com`
- **Lab Test Logins**: `test-lab@example.com` / `TestLab123!`

---

### 2. GPU Analysis Backend (Modal)

The backend runs onserverless cloud containers (Modal App: `copilot-hackathon-backend-analysis`) with automatic GPU scaling.

```bash
cd backend
# Create local virtual environment and authenticate Modal
uv venv && source .venv/bin/activate
uv run --python 3.12 modal token new

# Upload backend secrets from .env to Modal vault
uv run --python 3.12 modal secret create copilot-hackathon-backend-secrets --from-dotenv .env --force

# Deploy backend to Modal
uv run --python 3.12 modal deploy modal_app.py
```

- **Analysis API Endpoint**: `https://somadisingh--copilot-hackathon-backend-analysis-submit-analysis.modal.run`

---

### 3. iOS Data Collector (SwiftUI / ARKit)

The iOS application captures the demo and uploads raw bundles.

```bash
cd iosApp
# Generate the Xcode project from project.yml
xcodegen generate

# Open in Xcode
open DataCollector.xcodeproj
```

*Note: Ensure you configure your own Apple Developer Account Team under "Signing & Capabilities" in Xcode and enable "Developer Mode" on your test iPhone.*

---

### 4. Database & Orchestration (Supabase)

To provision or push database changes:

```bash
cd supabase
# Link local CLI to your project
supabase link --project-ref uiminnwdvqjkrrjoyylx

# Push base schema & analysis migrations through session pooler
supabase db push --db-url "postgresql://postgres.uiminnwdvqjkrrjoyylx:[PASSWORD]@aws-1-us-west-2.pooler.supabase.com:5432/postgres"

# Set secrets for the Edge Function
supabase secrets set --project-ref uiminnwdvqjkrrjoyylx \
  MODAL_ANALYSIS_URL="https://somadisingh--copilot-hackathon-backend-analysis-submit-analysis.modal.run" \
  MODAL_ANALYSIS_SECRET="[SHARED_SECRET]" \
  COPILOT_HACKATHON_ENABLE_RESOURCE_INTENSIVE_AI_TASKS="1"

# Deploy Edge Function
supabase functions deploy submit-recording --project-ref uiminnwdvqjkrrjoyylx --use-api
```

---

## 📂 Robot-Learning Artifacts (What the Robot Sees)

All processed outputs are saved privately in Supabase Storage under the `recordings/<id>/analysis/` prefix.

1. **3D Reconstruction (`gaussian_splat/splat.spz`)**: A compressed 3D Gaussian Splat scene containing spatial coordinates, color, opacity, and shape of every reconstructed point. Robots learn physical geometry and spatial affordances directly from this interactive 3D model.
2. **Object Affordances (`yolo-detections.json`)**: Tracks 2D bounding boxes and frame-by-frame locations of prompt-specified objects (`['can', 'dustbin']`, `['laptop', 'charger']`).
3. **Manipulation Demonstrations (`mediapipe-hands.json`)**: Contains coordinate timelines for 21 3D joint and fingertip positions. Useful for training hand-eye coordination policies.
4. **Task Phases (`temporal-actions.json`)**: Segmented stages of action progression over time.
5. **AI Quality Gate (`gemini-eval.json`)**: Deep judgment scoring (0-10), task summaries, and detailed explanations of whether the human demonstrated a successful demonstration.

---

## 🔍 How to View and Download Artifacts

### 1. In-Browser 3D Studio (Recommended)
Open the Studio page in the web app:
`http://localhost:3001/studio/[recording_id]` (e.g., `http://localhost:3001/studio/d2167e59-0431-4d9c-b821-2c393bce966e`).
*Log in as `test-lab@example.com` or your collector profile first to bypass RLS filters.*

### 2. Supabase Storage Browser (Direct)
Navigate to **Storage -> Buckets -> `recordings`** in your Supabase project dashboard:
`https://supabase.com/dashboard/project/uiminnwdvqjkrrjoyylx/storage/buckets/recordings`
Navigate into any `<recording_id>/analysis/` folder to manually inspect or download the JSON labels or 3D files.

### 3. Dedicated 3D Splat Editor
Download `splat.spz` or `seed_points.ply` from Storage and drag-and-drop it into **PlayCanvas SuperSplat** (https://superspl.at/editor) to view, clean, or rotate the 3D scene.
