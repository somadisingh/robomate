# CLAUDE.md

Guidance for working in this repository.

## Project: "Airtasker for physical data" (working name)

A marketplace where everyday people get **paid to record real-world physical
behaviours** (e.g. picking up a fruit, opening a door, using a tool). The
recordings — video plus rich sensor metadata — are sold to research labs that
need real-world data to train robotics / embodied-AI models.

Three pieces make up the full product:

1. **iOS app** (this person's responsibility) — capture video + sensor data on
   an iPhone and upload it to the cloud.
2. **Web app** (Next.js + Vercel + Supabase) — bounty marketplace + collected-data
   management dashboard for both contributors and labs.
3. **AI / research layer** — hand-pose estimation, depth estimation, object
   detection, action labelling, and (stretch) natural-language search over the
   video corpus.

The flow: a lab posts a **bounty** ("record yourself doing X, $Y") → a contributor
records it with the iOS app → data lands in cloud storage → the lab reviews and
pays.

## My scope: the iOS data-collection app

**Platform:** iOS, target device **iPhone 14 Pro**, built with **Swift + SwiftUI**.

**The app does one thing well: capture richly-annotated recordings and upload
them to the cloud.** No marketplace UI, no AI inference on-device — that lives in
the web app / research layer.

### Core requirements

- **Record video** to a bounty's spec (the task tells the user what behaviour to
  capture).
- **Local-first capture:** record → save locally → user taps **Upload** to push
  to the cloud. Recording must not depend on a live network connection.
- **Capture sensor metadata alongside each recording:**
  - GPS location
  - Timestamp
  - IMU: accelerometer, gyroscope, magnetometer
- **Upload to Supabase Storage buckets** (cloud object storage). Each upload
  carries its metadata so the web dashboard can display and query it.

### Sensors — priority order

| Sensor | Priority | Notes |
| --- | --- | --- |
| Single rear camera (video) | **Must have** | Core of every recording |
| GPS + timestamp | **Must have** | Metadata on every clip |
| IMU (accel / gyro / magnetometer) | **Must have** | Motion context |
| Camera 6DoF pose (ARKit VIO) | **Must have** | The differentiator — ground-truth camera trajectory web-scraped video can't provide |
| LiDAR depth | Stretch | iPhone 14 Pro has a LiDAR scanner; metric depth |
| Stereo / dual-camera simultaneous capture | Stretch | For stereo vision |

Ship the must-haves first; treat LiDAR and stereo capture as stretch goals.

### Local storage: the recording bundle

**The data is NOT stored inside the video file.** An `.mp4` only holds pixels
(and audio) — it can't naturally hold depth maps, 100Hz IMU rows, or per-frame
camera poses, and forcing them in would make agents decode the whole video just
to read the IMU.

Instead, each recording is a **bundle**: one folder containing the video *next
to* a separate file per sensor stream. They stay aligned because **every sample
carries a timestamp from one shared clock** (the ARKit frame timestamp) — not
because they share a file.

```
Documents/recordings/
  <recordingId>/              # one folder = one recording ("episode")
    video.mp4                 # RGB pixels only (HEVC)
    depth.bin                 # per-frame LiDAR depth (stretch)
    poses.parquet             # camera 6DoF pose per frame
    imu.parquet               # accel/gyro/mag rows (~100Hz)
    intrinsics.json           # camera lens matrix (fx, fy, cx, cy)
    metadata.json             # manifest: ids, gps, timestamps, which streams
                              #   exist, upload status
```

`metadata.json` is the **self-describing manifest** so a cloud agent can read
one small file and know exactly what the bundle contains and where each stream
lives. Track each recording's lifecycle in a local **SwiftData** row:

```
recording → local → uploading → uploaded
```

**Capture loop:** append each stream to its own file in real time (never buffer
in memory). On stop, finalise the video + write `metadata.json`. On upload, push
the whole folder to a Supabase bucket at `recordings/<recordingId>/...` and
insert a DB row pointing to it. Aligning the streams by timestamp is the cloud's
job, not the app's.

### Data sizes (per minute, rough)

The metadata streams you'd worry about are actually tiny — the cost is pixels.

| Stream | ~Size / min | Note |
| --- | --- | --- |
| RGB video (1080p HEVC) | ~60 MB | hardware-compressed |
| LiDAR depth | the monster — throttle to ~10–15fps + compress | raw is ~690 MB/min |
| Camera pose | ~130 KB | negligible |
| IMU | ~260 KB | negligible |
| GPS | ~0 | negligible |

Keep clips short (~10–60s, matches the bounty model), default to 1080p + HEVC.

The data is kept in Private Sandbox, not Photos/Files. 

## Tech & conventions

- **Language/UI:** Swift, SwiftUI.
- **Cloud:** Supabase (Storage buckets for media; metadata in Supabase as
  needed). The web app shares this Supabase backend.
- **Frameworks to reach for:**
  - `ARKit` — **primary capture source.** One `ARFrame` callback gives RGB +
    LiDAR depth + 6DoF camera pose + intrinsics, all time-synced. Provides the
    camera-pose differentiator (visual-inertial odometry).
  - `AVFoundation` (`AVAssetWriter`) — hardware-encodes the RGB frames to `.mp4`.
  - `CoreMotion` — IMU (accelerometer, gyroscope, magnetometer, ~100Hz).
  - `CoreLocation` — GPS.
  - `SwiftData` — local DB tracking each recording's upload status.
  - `URLSession` (background) — resilient uploads that survive backgrounding.

### Project setup & building

The Xcode project is **generated by [XcodeGen](https://github.com/yonaskolb/XcodeGen)**
from `project.yml` — the `.xcodeproj` is git-ignored and disposable.

- **Regenerate after adding/removing files or editing `project.yml`:**
  `xcodegen generate`
- **Source lives in** `DataCollector/` (globbed automatically — new files are
  picked up on regenerate, no manual target membership needed).
- **Declare SwiftPM dependencies in `project.yml`**, not via Xcode's GUI (GUI
  changes are wiped on regenerate). Supabase is already wired in.
- **Build check (CLI):**
  `xcodebuild -project DataCollector.xcodeproj -scheme DataCollector -destination 'generic/platform=iOS Simulator' build`
- **Real testing needs a physical iPhone** — the simulator has no camera, ARKit,
  LiDAR, or IMU. Set your signing Team in `project.yml` (`DEVELOPMENT_TEAM`) so it
  survives regeneration.

## Context / references

- This is a hackathon project (Copilot Hackathon). Bias toward a **working demo** over
  completeness.
- Research that motivates the data: NVIDIA EgoScale
  (https://research.nvidia.com/labs/gear/egoscale/) and Microsoft VITRA
  (https://microsoft.github.io/VITRA/).
- Stretch for the whole team: real deployment on an **SO-101 robot arm**.

## Glossary — what each library / model does

**Mental model:** the libraries below run *in the iOS app* (you build with them).
The AI models run *in the cloud, later* — the app never executes them; it only
captures the raw data they consume.

### Libraries in the iOS app (Apple's built-in, free toolkits)

| Library | What it does, in plain terms |
| --- | --- |
| **SwiftUI** | Builds the screens (buttons, the record button, the recordings list). |
| **AVFoundation** | The camcorder — records the camera to an `.mp4` using the hardware video encoder. |
| **ARKit** | Tracks **where the phone is in 3D space** as it moves and reads **LiDAR depth**. Hands you camera + depth + position + timestamp together. |
| **CoreMotion** | Reads the **IMU** — the phone's "inner ear" (movement, rotation, compass), ~100×/sec. |
| **CoreLocation** | Reads **GPS** (latitude/longitude). |
| **SwiftData** | Local database — remembers your recordings and whether each is uploaded. |
| **URLSession** | Networking — uploads files to the cloud, even when the app is backgrounded. |

### Cloud backend

| Tool | What it does |
| --- | --- |
| **Supabase** | Ready-made backend. **Storage buckets** = cloud folder for the video files; **Postgres database** = table listing each recording's info + where its files live. Has a Swift SDK. |

### AI models (cloud-side — NOT in the app)

| Model / tool | What it extracts from the uploaded data |
| --- | --- |
| **MediaPipe** | **Hand pose** — 21 hand joints per frame; the "action" a robot would copy. |
| **Depth estimation** | Guesses distance from a flat image — a fallback when there's no LiDAR. |
| **Object detection / segmentation** | Finds and outlines objects ("cup here, hand there"). |
| **Action labelling** | Names the activity ("picking up a fruit"). |
| **Embeddings + vector DB** | Turns clips into meaning-vectors for natural-language search (powers "show me clips of picking up a fruit"; Supabase `pgvector`). |

### Robotics endgame (concepts, not code you write)

| Term | Meaning |
| --- | --- |
| **VLA model** ("Vision-Language-Action") | AI that watches video + reads an instruction → outputs robot movements. Your data trains these (e.g. EgoScale, VITRA). |
| **LeRobot** | HuggingFace's robotics library + a standard dataset format (video + sensor tables + metadata). Worth matching our bundle to. |
| **SO-101** | A cheap robot arm LeRobot supports — the team's stretch deployment target. |

**Jargon decoder:** **LiDAR** = laser depth scanner (distances in metres, Pro
phones only). **IMU** = accelerometer + gyroscope + magnetometer together.
**6DoF pose** = the phone's full position (x, y, z) *and* orientation
(tilt/turn/roll) in space.
