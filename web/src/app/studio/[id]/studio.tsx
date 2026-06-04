'use client'

import Link from 'next/link'
import dynamic from 'next/dynamic'
import { useRouter } from 'next/navigation'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './studio.css'
import SceneControls from './scene-controls'
import type { SceneLayers } from './scene-view'
import { useCameraSync, type CameraPath, type SceneViewHandle } from './use-camera-sync'

const SceneView = dynamic(() => import('./scene-view'), { ssr: false })

/* ═════════════════════════════════════════════════════════════════════
 *  TYPES — from page.tsx
 * ═════════════════════════════════════════════════════════════════════ */

export type AnalysisJobKind =
  | 'gemini_eval'
  | 'mediapipe_hands'
  | 'yolo_objects'
  | 'sam_segments'
  | 'temporal_actions'
  | 'gaussian_splat'

export type AnalysisJobStatus = 'pending' | 'running' | 'succeeded' | 'failed'

export type AnalysisJobPayload = {
  kind: AnalysisJobKind
  status: AnalysisJobStatus
  artifact_path: string | null
  summary: Record<string, unknown> | null
  error: string | null
  started_at: string | null
  finished_at: string | null
  signedUrl: string | null
}

export type SiblingRecording = {
  id: string
  takeNumber: number
  score: number | null
  success: boolean | null
  status: 'uploaded' | 'analyzing' | 'analyzed' | 'analysis_failed'
  createdAt: string
  durationMs: number | null
  isCurrent: boolean
}

export type DetectedObjectRow = {
  class_name?: string
  count?: number
  max_confidence?: number
  representative_frame?: number
}

export type StudioData = {
  recording: {
    id: string
    status: 'uploaded' | 'analyzing' | 'analyzed' | 'analysis_failed'
    isScoring: boolean
    summary: string | null
    success: boolean | null
    successReasoning: string | null
    score: number | null
    scoreReasoning: string | null
    detectedObjects: DetectedObjectRow[]
    createdAt: string
    deviceModel: string | null
    durationMs: number | null
    sizeBytes: number | null
    gpsLat: number | null
    gpsLon: number | null
    gpsAccuracyM: number | null
    streams: string[]
    collectorId: string | null
    collectorName: string | null
    depthWidth: number | null
    depthHeight: number | null
    depthFrameCount: number | null
  }
  task: {
    id: string
    title: string | null
    description: string | null
    objects: string[]
    requiredCapabilities: string[]
    bountyAmount: number | null
    dataType: string | null
  } | null
  siblings: SiblingRecording[]
  jobs: AnalysisJobPayload[]
  streamUrls: Record<'video.mp4' | 'imu.jsonl' | 'poses.jsonl' | 'intrinsics.json' | 'depth.bin' | 'transcript.json', string | null>
  splatScene: SplatScenePayload | null
}

export type SplatScenePayload = {
  splatUrl: string
  cameraPathUrl: string
  seedPointsUrl: string | null
  numGaussians: number
  frameCount: number
  fps: number
}

/* ═════════════════════════════════════════════════════════════════════
 *  PARSED ARTIFACT SHAPES
 * ═════════════════════════════════════════════════════════════════════ */

type HandData = {
  handedness: 'Left' | 'Right' | string
  confidence: number
  landmarks: { x: number; y: number; z: number }[] // 21, normalized 0-1
}
type HandFrame = { t: number; hands: HandData[] }

type YoloInst = {
  box: [number, number, number, number] // pixel xyxy
  conf: number
  class_id: number
  class_name: string
}
type YoloFrame = { t: number; instances: YoloInst[] }
type YoloMeta = { origW: number; origH: number; sourceFps: number; frameCount: number }

type SamInst = {
  polygon: [number, number][] // normalized xyn
  class_name: string
  conf: number
}
type SamFrame = { t: number; instances: SamInst[] }

type ActionSeg = {
  start: number
  end: number
  caption: string
  hand: 'left' | 'right' | 'unknown'
  object: string | null
  confidence: number
  meaningful: boolean
  reason: string
}

/** A speech transcript phrase, timed in seconds from recording start. */
type TranscriptSeg = { start: number; end: number; text: string }

type ImuSample = { t: number; ax: number; ay: number; az: number; gx: number; gy: number; gz: number }
type PoseSample = {
  t: number
  px: number
  py: number
  pz: number
  qw: number
  qx: number
  qy: number
  qz: number
  yaw: number // degrees, derived
  pitch: number
  roll: number
}

/** A single LiDAR depth frame. `data` is W*H float32 metres (row-major). */
type DepthFrame = { t: number; data: Float32Array }
type DepthArtifact = {
  frames: DepthFrame[]
  width: number
  height: number
  /** Robust vmin/vmax used for colourmap normalization (5th / 95th pct). */
  vmin: number
  vmax: number
}

type ArtifactState = {
  hands: { samples: HandFrame[]; sourceFps: number; sampleFps: number; stride: number } | null
  yolo: { frames: YoloFrame[]; meta: YoloMeta } | null
  sam: { frames: SamFrame[]; meta: YoloMeta; textPrompts: string[] } | null
  actions: ActionSeg[] | null
  imu: ImuSample[] | null
  poses: PoseSample[] | null
  depth: DepthArtifact | null
  transcript: TranscriptSeg[] | null
}

const EMPTY_ARTIFACTS: ArtifactState = {
  hands: null,
  yolo: null,
  sam: null,
  actions: null,
  imu: null,
  poses: null,
  depth: null,
  transcript: null,
}

/* ═════════════════════════════════════════════════════════════════════
 *  LAYERS
 * ═════════════════════════════════════════════════════════════════════ */

type LayerKey = 'rgb' | 'depth' | 'hands' | 'objects' | 'masks'

type LayerSpec = {
  key: LayerKey
  label: string
  source: string
  shortcut: string
  swatch: string
  jobKind?: AnalysisJobKind
}

const LAYER_SPECS: LayerSpec[] = [
  { key: 'rgb', label: 'RGB Source', source: 'iPhone wrist-mounted', shortcut: '1', swatch: '#e8e8ea' },
  { key: 'depth', label: 'Depth Field', source: 'LiDAR · projected', shortcut: '2', swatch: '#5b8def' },
  { key: 'hands', label: 'Hand Landmarks', source: 'MediaPipe Hands', shortcut: '3', swatch: '#f0a35e', jobKind: 'mediapipe_hands' },
  { key: 'objects', label: 'Object Detection', source: 'YOLOv26', shortcut: '4', swatch: '#c2e02b', jobKind: 'yolo_objects' },
  { key: 'masks', label: 'Instance Mask', source: 'SAM 3.1', shortcut: '5', swatch: '#d96b9d', jobKind: 'sam_segments' },
]

type LayerState = Record<LayerKey, { visible: boolean; opacity: number }>

const INITIAL_LAYERS: LayerState = {
  rgb: { visible: true, opacity: 1 },
  depth: { visible: false, opacity: 0.6 },
  hands: { visible: true, opacity: 1 },
  objects: { visible: true, opacity: 1 },
  masks: { visible: false, opacity: 0.5 },
}

const JOB_LABEL: Record<AnalysisJobKind, string> = {
  gemini_eval: 'Gemini scoring',
  mediapipe_hands: 'MediaPipe hands',
  yolo_objects: 'YOLOv26 detection',
  sam_segments: 'SAM 3.1 masks',
  temporal_actions: 'Action segmentation',
  gaussian_splat: '3D Gaussian splat',
}

const CLASS_COLORS = [
  '#c2e02b', '#5b8def', '#f0a35e', '#d96b9d', '#5edcc6',
  '#e8e8ea', '#b3a4ff', '#ffd166', '#9af7a0', '#f87171',
]

function colorForClass(name: string): string {
  let h = 0
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) | 0
  return CLASS_COLORS[Math.abs(h) % CLASS_COLORS.length]
}

/* ═════════════════════════════════════════════════════════════════════
 *  HELPERS
 * ═════════════════════════════════════════════════════════════════════ */

function fmtTime(t: number) {
  if (!isFinite(t) || t < 0) t = 0
  const m = Math.floor(t / 60)
  const s = Math.floor(t % 60)
  const cs = Math.floor((t * 100) % 100)
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${cs.toString().padStart(2, '0')}`
}

function fmtBytes(n: number | null) {
  if (n == null) return '—'
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`
}

function fmtDate(iso: string | null) {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString(undefined, {
      year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
    })
  } catch { return iso }
}

function shortId(id: string) {
  return id.slice(0, 8).toUpperCase()
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t
}

/** Binary search for largest item where `keyOf(item) <= t`. Returns -1 if none. */
function searchLowerBound<T>(arr: T[], t: number, keyOf: (x: T) => number): number {
  if (arr.length === 0) return -1
  let lo = 0, hi = arr.length - 1, best = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (keyOf(arr[mid]) <= t) { best = mid; lo = mid + 1 }
    else hi = mid - 1
  }
  return best
}

/** Convert Hamilton quaternion (w,x,y,z) → Euler degrees (yaw, pitch, roll, intrinsic ZYX). */
function quatToEulerDeg(qw: number, qx: number, qy: number, qz: number) {
  // roll (x)
  const sinr = 2 * (qw * qx + qy * qz)
  const cosr = 1 - 2 * (qx * qx + qy * qy)
  const roll = Math.atan2(sinr, cosr)
  // pitch (y)
  const sinp = 2 * (qw * qy - qz * qx)
  const pitch = Math.abs(sinp) >= 1 ? Math.sign(sinp) * Math.PI / 2 : Math.asin(sinp)
  // yaw (z)
  const siny = 2 * (qw * qz + qx * qy)
  const cosy = 1 - 2 * (qy * qy + qz * qz)
  const yaw = Math.atan2(siny, cosy)
  const R = 180 / Math.PI
  return { yaw: yaw * R, pitch: pitch * R, roll: roll * R }
}

/* ═════════════════════════════════════════════════════════════════════
 *  ARTIFACT LOADER
 * ═════════════════════════════════════════════════════════════════════ */

function useArtifacts(data: StudioData, samWanted: boolean, depthWanted: boolean): {
  artifacts: ArtifactState
  loading: Record<string, boolean>
  errors: Record<string, string | null>
  depthProgress: number | null
} {
  const [depthProgress, setDepthProgress] = useState<number | null>(null)
  const [artifacts, setArtifacts] = useState<ArtifactState>(EMPTY_ARTIFACTS)
  const [loading, setLoading] = useState<Record<string, boolean>>({})
  const [errors, setErrors] = useState<Record<string, string | null>>({})

  const handsUrl = data.jobs.find((j) => j.kind === 'mediapipe_hands' && j.status === 'succeeded')?.signedUrl ?? null
  const yoloUrl = data.jobs.find((j) => j.kind === 'yolo_objects' && j.status === 'succeeded')?.signedUrl ?? null
  const samUrl = data.jobs.find((j) => j.kind === 'sam_segments' && j.status === 'succeeded')?.signedUrl ?? null
  const actionsUrl = data.jobs.find((j) => j.kind === 'temporal_actions' && j.status === 'succeeded')?.signedUrl ?? null
  const imuUrl = data.streamUrls['imu.jsonl']
  const posesUrl = data.streamUrls['poses.jsonl']
  const transcriptUrl = data.streamUrls['transcript.json']

  // hands
  useEffect(() => {
    if (!handsUrl) return
    let cancelled = false
    setLoading((p) => ({ ...p, hands: true }))
    fetch(handsUrl)
      .then((r) => r.json())
      .then((j: { source_fps: number; sample_fps: number; stride: number; frames: Array<{ source_frame: number; time_sec: number; hands: HandData[] }> }) => {
        if (cancelled) return
        const samples: HandFrame[] = j.frames
          .filter((f) => f.hands && f.hands.length > 0)
          .map((f) => ({ t: f.time_sec, hands: f.hands }))
        setArtifacts((p) => ({
          ...p,
          hands: { samples, sourceFps: j.source_fps, sampleFps: j.sample_fps, stride: j.stride },
        }))
        setLoading((p) => ({ ...p, hands: false }))
      })
      .catch((e) => {
        if (cancelled) return
        setErrors((p) => ({ ...p, hands: String(e) }))
        setLoading((p) => ({ ...p, hands: false }))
      })
    return () => { cancelled = true }
  }, [handsUrl])

  // yolo
  useEffect(() => {
    if (!yoloUrl) return
    let cancelled = false
    setLoading((p) => ({ ...p, yolo: true }))
    fetch(yoloUrl)
      .then((r) => r.json())
      .then((j: { frame_count: number; frames: Array<{ frame_index: number; orig_shape: [number, number]; instances: Array<{ box_xyxy: [number, number, number, number]; confidence: number; class_id: number; class_name: string }> }> }) => {
        if (cancelled) return
        const origH = j.frames[0]?.orig_shape?.[0] ?? 720
        const origW = j.frames[0]?.orig_shape?.[1] ?? 1280
        const sourceFps = (data.recording.durationMs && j.frame_count)
          ? j.frame_count / (data.recording.durationMs / 1000)
          : 30
        const frames: YoloFrame[] = j.frames
          .filter((f) => f.instances && f.instances.length > 0)
          .map((f) => ({
            t: f.frame_index / sourceFps,
            instances: f.instances.map((i) => ({
              box: i.box_xyxy, conf: i.confidence, class_id: i.class_id, class_name: i.class_name,
            })),
          }))
        setArtifacts((p) => ({ ...p, yolo: { frames, meta: { origW, origH, sourceFps, frameCount: j.frame_count } } }))
        setLoading((p) => ({ ...p, yolo: false }))
      })
      .catch((e) => {
        if (cancelled) return
        setErrors((p) => ({ ...p, yolo: String(e) }))
        setLoading((p) => ({ ...p, yolo: false }))
      })
    return () => { cancelled = true }
  }, [yoloUrl, data.recording.durationMs])

  // sam — lazy
  useEffect(() => {
    if (!samUrl || !samWanted) return
    let cancelled = false
    setLoading((p) => ({ ...p, sam: true }))
    fetch(samUrl)
      .then((r) => r.json())
      .then((j: { frame_count: number; text_prompts: string[]; frames: Array<{ frame_index: number; orig_shape: [number, number]; instances: Array<{ box_xyxy: [number, number, number, number]; confidence: number; class_id: number; class_name: string; mask_polygon_xyn: [number, number][] }> }> }) => {
        if (cancelled) return
        const origH = j.frames[0]?.orig_shape?.[0] ?? 720
        const origW = j.frames[0]?.orig_shape?.[1] ?? 1280
        const sourceFps = (data.recording.durationMs && j.frame_count)
          ? j.frame_count / (data.recording.durationMs / 1000)
          : 30
        const frames: SamFrame[] = j.frames
          .filter((f) => f.instances && f.instances.length > 0)
          .map((f) => ({
            t: f.frame_index / sourceFps,
            instances: f.instances.map((i) => ({
              polygon: i.mask_polygon_xyn, class_name: i.class_name, conf: i.confidence,
            })),
          }))
        setArtifacts((p) => ({ ...p, sam: { frames, meta: { origW, origH, sourceFps, frameCount: j.frame_count }, textPrompts: j.text_prompts ?? [] } }))
        setLoading((p) => ({ ...p, sam: false }))
      })
      .catch((e) => {
        if (cancelled) return
        setErrors((p) => ({ ...p, sam: String(e) }))
        setLoading((p) => ({ ...p, sam: false }))
      })
    return () => { cancelled = true }
  }, [samUrl, samWanted, data.recording.durationMs])

  // actions
  useEffect(() => {
    if (!actionsUrl) return
    let cancelled = false
    setLoading((p) => ({ ...p, actions: true }))
    fetch(actionsUrl)
      .then((r) => r.json())
      .then((j: { segments: Array<{ start_sec: number; end_sec: number; caption: string; hand: string; object: string | null; confidence: number; meaningful_manipulation: boolean; reason: string }> }) => {
        if (cancelled) return
        const segments: ActionSeg[] = (j.segments ?? []).map((s) => {
          const hand: ActionSeg['hand'] =
            s.hand === 'left' ? 'left' : s.hand === 'right' ? 'right' : 'unknown'
          return {
            start: s.start_sec,
            end: s.end_sec,
            caption: s.caption,
            hand,
            object: s.object,
            confidence: s.confidence,
            meaningful: !!s.meaningful_manipulation,
            reason: s.reason,
          }
        }).sort((a, b) => a.start - b.start)
        setArtifacts((p) => ({ ...p, actions: segments }))
        setLoading((p) => ({ ...p, actions: false }))
      })
      .catch((e) => {
        if (cancelled) return
        setErrors((p) => ({ ...p, actions: String(e) }))
        setLoading((p) => ({ ...p, actions: false }))
      })
    return () => { cancelled = true }
  }, [actionsUrl])

  // imu
  useEffect(() => {
    if (!imuUrl) return
    let cancelled = false
    setLoading((p) => ({ ...p, imu: true }))
    fetch(imuUrl)
      .then((r) => r.text())
      .then((text) => {
        if (cancelled) return
        const rows: ImuSample[] = []
        let t0: number | null = null
        for (const line of text.split('\n')) {
          if (!line.trim()) continue
          try {
            const o = JSON.parse(line)
            if (typeof o.t !== 'number') continue
            if (t0 == null) t0 = o.t
            rows.push({
              t: o.t - (t0 as number),
              ax: o.ax ?? 0, ay: o.ay ?? 0, az: o.az ?? 0,
              gx: o.gx ?? 0, gy: o.gy ?? 0, gz: o.gz ?? 0,
            })
          } catch { /* skip */ }
        }
        setArtifacts((p) => ({ ...p, imu: rows }))
        setLoading((p) => ({ ...p, imu: false }))
      })
      .catch((e) => {
        if (cancelled) return
        setErrors((p) => ({ ...p, imu: String(e) }))
        setLoading((p) => ({ ...p, imu: false }))
      })
    return () => { cancelled = true }
  }, [imuUrl])

  // transcript (speech-to-text phrases; times already 0-based from recording start)
  useEffect(() => {
    if (!transcriptUrl) return
    let cancelled = false
    setLoading((p) => ({ ...p, transcript: true }))
    fetch(transcriptUrl)
      .then((r) => r.json())
      .then((j: { segments?: Array<{ start_time: number; end_time: number; text: string }> }) => {
        if (cancelled) return
        const segs: TranscriptSeg[] = (j.segments ?? [])
          .filter((s) => typeof s.start_time === 'number' && (s.text ?? '').trim().length > 0)
          .map((s) => ({ start: s.start_time, end: s.end_time, text: s.text }))
          .sort((a, b) => a.start - b.start)
        setArtifacts((p) => ({ ...p, transcript: segs }))
        setLoading((p) => ({ ...p, transcript: false }))
      })
      .catch((e) => {
        if (cancelled) return
        setErrors((p) => ({ ...p, transcript: String(e) }))
        setLoading((p) => ({ ...p, transcript: false }))
      })
    return () => { cancelled = true }
  }, [transcriptUrl])

  // poses
  useEffect(() => {
    if (!posesUrl) return
    let cancelled = false
    setLoading((p) => ({ ...p, poses: true }))
    fetch(posesUrl)
      .then((r) => r.text())
      .then((text) => {
        if (cancelled) return
        const rows: PoseSample[] = []
        let t0: number | null = null
        for (const line of text.split('\n')) {
          if (!line.trim()) continue
          try {
            const o = JSON.parse(line)
            if (typeof o.t !== 'number') continue
            if (t0 == null) t0 = o.t
            const qw = o.qw ?? 1, qx = o.qx ?? 0, qy = o.qy ?? 0, qz = o.qz ?? 0
            const { yaw, pitch, roll } = quatToEulerDeg(qw, qx, qy, qz)
            rows.push({
              t: o.t - (t0 as number),
              px: o.px ?? 0, py: o.py ?? 0, pz: o.pz ?? 0,
              qw, qx, qy, qz, yaw, pitch, roll,
            })
          } catch { /* skip */ }
        }
        setArtifacts((p) => ({ ...p, poses: rows }))
        setLoading((p) => ({ ...p, poses: false }))
      })
      .catch((e) => {
        if (cancelled) return
        setErrors((p) => ({ ...p, poses: String(e) }))
        setLoading((p) => ({ ...p, poses: false }))
      })
    return () => { cancelled = true }
  }, [posesUrl])

  // depth.bin — lazy. Stream + parse into per-frame Float32Array, then derive
  // robust vmin/vmax (5th/95th pct of all valid samples) for colour normalization.
  const depthUrl = data.streamUrls['depth.bin']
  const depthW = data.recording.depthWidth
  const depthH = data.recording.depthHeight
  const depthFC = data.recording.depthFrameCount
  useEffect(() => {
    if (!depthWanted || !depthUrl || !depthW || !depthH || !depthFC) return
    if (artifacts.depth) return // already loaded
    let cancelled = false
    setLoading((p) => ({ ...p, depth: true }))
    setDepthProgress(0)

    ;(async () => {
      try {
        const res = await fetch(depthUrl)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const totalLen = Number(res.headers.get('content-length') || 0)
        const reader = res.body!.getReader()
        const chunks: Uint8Array[] = []
        let received = 0
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          if (cancelled) { reader.cancel(); return }
          chunks.push(value)
          received += value.byteLength
          if (totalLen > 0) setDepthProgress(received / totalLen)
        }
        // Concatenate
        const buf = new Uint8Array(received)
        let off = 0
        for (const c of chunks) { buf.set(c, off); off += c.byteLength }
        if (cancelled) return

        const px = depthW * depthH
        const recordSize = 8 + px * 4
        const actualFrames = Math.floor(buf.byteLength / recordSize)
        const frames: DepthFrame[] = new Array(actualFrames)
        const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
        let t0: number | null = null
        // Cumulative histogram for percentile-based vmin/vmax (1cm bins, 0..10m → 1000 bins)
        const BINS = 1000
        const BIN_M = 0.01
        const hist = new Uint32Array(BINS)
        let totalSamples = 0
        for (let i = 0; i < actualFrames; i++) {
          const base = i * recordSize
          const t = dv.getFloat64(base, true)
          if (t0 == null) t0 = t
          const data = new Float32Array(buf.buffer, buf.byteOffset + base + 8, px).slice()
          frames[i] = { t: t - t0, data }
          // sample every 16th pixel for the histogram (still ~2300 samples per frame)
          for (let j = 0; j < px; j += 16) {
            const d = data[j]
            if (isFinite(d) && d > 0 && d < 10) {
              const b = Math.min(BINS - 1, Math.floor(d / BIN_M))
              hist[b]++
              totalSamples++
            }
          }
        }
        // Robust min/max via percentiles
        const lo = totalSamples * 0.05
        const hi = totalSamples * 0.95
        let cum = 0, vmin = 0.4, vmax = 4.0
        for (let b = 0; b < BINS; b++) {
          cum += hist[b]
          if (vmin === 0.4 && cum >= lo) vmin = b * BIN_M
          if (cum >= hi) { vmax = b * BIN_M; break }
        }
        if (vmax <= vmin) vmax = vmin + 0.5

        if (cancelled) return
        setArtifacts((p) => ({
          ...p,
          depth: { frames, width: depthW, height: depthH, vmin, vmax },
        }))
        setLoading((p) => ({ ...p, depth: false }))
        setDepthProgress(null)
      } catch (e) {
        if (cancelled) return
        setErrors((p) => ({ ...p, depth: String(e) }))
        setLoading((p) => ({ ...p, depth: false }))
        setDepthProgress(null)
      }
    })()
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [depthUrl, depthW, depthH, depthFC, depthWanted])

  return { artifacts, loading, errors, depthProgress }
}

/* ═════════════════════════════════════════════════════════════════════
 *  STUDIO ROOT
 * ═════════════════════════════════════════════════════════════════════ */

export default function Studio({ data }: { data: StudioData }) {
  const router = useRouter()
  const videoRef = useRef<HTMLVideoElement>(null)

  const durationS = useMemo(() => {
    if (data.recording.durationMs && data.recording.durationMs > 0) {
      return data.recording.durationMs / 1000
    }
    return 30
  }, [data.recording.durationMs])

  const [layers, setLayers] = useState<LayerState>(INITIAL_LAYERS)
  const [currentTime, setCurrentTime] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [activeTab, setActiveTab] = useState<'eval' | 'task' | 'jobs' | 'meta'>('eval')
  const [scrubHover, setScrubHover] = useState<number | null>(null)
  const [selectedLayer, setSelectedLayer] = useState<LayerKey>('objects')
  const [siblingOpen, setSiblingOpen] = useState(false)
  const [videoSize, setVideoSize] = useState<{ vw: number; vh: number } | null>(null)
  const [stageSize, setStageSize] = useState<{ w: number; h: number } | null>(null)

  // 3D scene state — only relevant when data.splatScene is present.
  const [viewMode, setViewMode] = useState<'2d' | '3d'>('2d')
  const [sceneLayers, setSceneLayers] = useState<SceneLayers>({
    splats: true,
    trajectory: true,
    seedPoints: false,
  })
  const [cameraMode, setCameraMode] = useState<'synced' | 'free'>('synced')
  const sceneHandleRef = useRef<SceneViewHandle | null>(null)
  const [cameraPath, setCameraPath] = useState<CameraPath | null>(null)
  const sceneAvailable = data.splatScene !== null

  // Fetch camera path JSON once when the splat scene becomes available.
  useEffect(() => {
    if (!data.splatScene) return
    let cancelled = false
    fetch(data.splatScene.cameraPathUrl)
      .then((r) => r.json())
      .then((path: CameraPath) => {
        if (!cancelled) setCameraPath(path)
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [data.splatScene])

  // Camera sync is only ENGAGED when the user has chosen synced AND the video
  // is actually playing. When paused (or in free mode), the user gets manual
  // control via WASD/drag; pressing play snaps the camera back to the path.
  const syncEngaged = viewMode === '3d' && cameraMode === 'synced' && isPlaying
  const manualControl = viewMode === '3d' && !syncEngaged
  useCameraSync({
    videoRef,
    cameraPath,
    sceneHandle: sceneHandleRef.current,
    enabled: syncEngaged,
  })

  const jobByKind = useMemo(() => {
    const m = new Map<AnalysisJobKind, AnalysisJobPayload>()
    for (const j of data.jobs) m.set(j.kind, j)
    return m
  }, [data.jobs])

  const effectiveLayers = useMemo<LayerState>(() => {
    let next = layers
    for (const spec of LAYER_SPECS) {
      if (!spec.jobKind) continue
      const job = jobByKind.get(spec.jobKind)
      if (job && job.status !== 'succeeded' && next[spec.key].visible) {
        if (next === layers) next = { ...layers }
        next[spec.key] = { ...next[spec.key], visible: false }
      }
    }
    return next
  }, [jobByKind, layers])

  const { artifacts, loading, depthProgress } = useArtifacts(data, effectiveLayers.masks.visible, effectiveLayers.depth.visible)

  // Keep a ref of the latest currentTime so we can restore playback position
  // when the <video> element remounts on viewMode flips (2D ↔ 3D) without
  // having to add currentTime to the rVFC effect's deps.
  const currentTimeRef = useRef(currentTime)
  useEffect(() => { currentTimeRef.current = currentTime }, [currentTime])

  // rVFC time sync — tight, native frame rate.
  // Re-runs on viewMode change so we re-bind listeners to the new <video>
  // element (it gets remounted under a different parent in 3D mode).
  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    let handle = 0
    let cancelled = false

    // Set the video's playbackRate when speed changes
    v.playbackRate = speed

    // Restore playback position if the <video> remounted (currentTime=0 on a
    // fresh element but our state remembers where we were).
    const lastT = currentTimeRef.current
    if (lastT > 0.01 && Math.abs(v.currentTime - lastT) > 0.25) {
      try { v.currentTime = lastT } catch { /* metadata not ready; will seek when 'loadedmetadata' fires */ }
    }

    type RVFCMeta = { mediaTime: number }
    type VideoEl = HTMLVideoElement & {
      requestVideoFrameCallback?: (cb: (now: number, meta: RVFCMeta) => void) => number
      cancelVideoFrameCallback?: (h: number) => void
    }
    const ve = v as VideoEl
    const supported = typeof ve.requestVideoFrameCallback === 'function'

    if (supported) {
      const tick = (_now: number, meta: RVFCMeta) => {
        if (cancelled) return
        setCurrentTime(meta.mediaTime)
        handle = ve.requestVideoFrameCallback!(tick)
      }
      handle = ve.requestVideoFrameCallback!(tick)
    }

    // Always subscribe to play/pause + a fallback `timeupdate` for when paused/seeked
    const onPlay = () => setIsPlaying(true)
    const onPause = () => setIsPlaying(false)
    const onEnd = () => setIsPlaying(false)
    const onTime = () => { if (!supported || v.paused) setCurrentTime(v.currentTime) }
    const onSeeked = () => setCurrentTime(v.currentTime)
    const onLoaded = () => {
      if (v.videoWidth > 0 && v.videoHeight > 0) {
        setVideoSize({ vw: v.videoWidth, vh: v.videoHeight })
      }
    }
    v.addEventListener('play', onPlay)
    v.addEventListener('pause', onPause)
    v.addEventListener('ended', onEnd)
    v.addEventListener('timeupdate', onTime)
    v.addEventListener('seeked', onSeeked)
    v.addEventListener('loadedmetadata', onLoaded)
    // If already loaded
    if (v.videoWidth > 0) onLoaded()
    // Sync our isPlaying state to the new element's actual state (e.g., on
    // 2D ↔ 3D remount the new <video> starts paused even if we were playing).
    setIsPlaying(!v.paused)

    return () => {
      cancelled = true
      if (supported && handle && ve.cancelVideoFrameCallback) ve.cancelVideoFrameCallback(handle)
      v.removeEventListener('play', onPlay)
      v.removeEventListener('pause', onPause)
      v.removeEventListener('ended', onEnd)
      v.removeEventListener('timeupdate', onTime)
      v.removeEventListener('seeked', onSeeked)
      v.removeEventListener('loadedmetadata', onLoaded)
    }
  }, [speed, viewMode])

  // Keep playbackRate in sync if speed changes mid-life
  useEffect(() => {
    const v = videoRef.current
    if (v) v.playbackRate = speed
  }, [speed])

  // Keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.code === 'Space') {
        e.preventDefault()
        togglePlay()
      } else if (e.code === 'ArrowLeft') {
        seek(currentTime - (e.shiftKey ? 1 : 1 / 30))
      } else if (e.code === 'ArrowRight') {
        seek(currentTime + (e.shiftKey ? 1 : 1 / 30))
      } else if (e.key === 'Home') {
        seek(0)
      } else if (e.key === 'End') {
        seek(durationS)
      } else if (e.key === '[' && data.siblings.length > 1) {
        navigateSibling(-1)
      } else if (e.key === ']' && data.siblings.length > 1) {
        navigateSibling(1)
      } else {
        const layer = LAYER_SPECS.find((l) => l.shortcut === e.key)
        if (layer) toggleLayer(layer.key)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentTime, durationS, data.siblings])

  function togglePlay() {
    const v = videoRef.current
    if (!v) return
    if (v.paused) v.play().catch(() => {})
    else v.pause()
  }

  function seek(t: number) {
    const v = videoRef.current
    const clamped = Math.max(0, Math.min(durationS, t))
    setCurrentTime(clamped)
    if (v) v.currentTime = clamped
  }

  function toggleLayer(key: LayerKey) {
    setLayers((prev) => ({ ...prev, [key]: { ...prev[key], visible: !prev[key].visible } }))
  }

  function setOpacity(key: LayerKey, opacity: number) {
    setLayers((prev) => ({ ...prev, [key]: { ...prev[key], opacity } }))
  }

  function navigateSibling(dir: -1 | 1) {
    const idx = data.siblings.findIndex((s) => s.isCurrent)
    if (idx === -1) return
    const next = data.siblings[idx + dir]
    if (!next) return
    router.push(`/studio/${next.id}`)
  }

  const currentSibling = data.siblings.find((s) => s.isCurrent)
  const currentTakeIdx = data.siblings.findIndex((s) => s.isCurrent)

  // Active action segment lookup
  const currentAction = useMemo(() => {
    if (!artifacts.actions || artifacts.actions.length === 0) return null
    // overlapping segments per hand are possible — return the most-confident currently active
    const active = artifacts.actions.filter((s) => currentTime >= s.start && currentTime < s.end)
    if (active.length === 0) return null
    return active.reduce((best, s) => (s.confidence > best.confidence ? s : best))
  }, [artifacts.actions, currentTime])

  // Active transcript phrase for the subtitle bar under the video.
  const currentCaption = useMemo(() => {
    if (!artifacts.transcript) return null
    return artifacts.transcript.find((s) => currentTime >= s.start && currentTime < s.end)?.text ?? null
  }, [artifacts.transcript, currentTime])

  const frame = Math.floor(currentTime * 30)

  return (
    <div className={`studio-scope ${sceneAvailable ? 'studio-scope--has-scene' : ''}`}>
      <TopBar
        data={data}
        currentSibling={currentSibling ?? null}
        totalTakes={data.siblings.length}
        currentAction={currentAction?.caption ?? '—'}
        frame={frame}
        currentTime={currentTime}
        onPrev={() => navigateSibling(-1)}
        onNext={() => navigateSibling(1)}
        canPrev={currentTakeIdx > 0}
        canNext={currentTakeIdx >= 0 && currentTakeIdx < data.siblings.length - 1}
        siblingOpen={siblingOpen}
        onSiblingOpen={setSiblingOpen}
        onPickSibling={(id) => { setSiblingOpen(false); router.push(`/studio/${id}`) }}
      />

      {sceneAvailable && (
        <div className="studio__viewmode">
          <button
            className={`studio__viewmode-button ${viewMode === '2d' ? 'studio__viewmode-button--active' : ''}`}
            onClick={() => setViewMode('2d')}
          >
            2D layered
          </button>
          <button
            className={`studio__viewmode-button ${viewMode === '3d' ? 'studio__viewmode-button--active' : ''}`}
            onClick={() => setViewMode('3d')}
          >
            3D scene
          </button>
          {data.splatScene && (
            <span className="studio__viewmode-meta">
              {data.splatScene.numGaussians.toLocaleString()} gaussians · {data.splatScene.frameCount} frames
            </span>
          )}
        </div>
      )}

      <div className="studio__body">
        {viewMode === '2d' && (
          <>
            <LeftRail
              layers={effectiveLayers}
              jobByKind={jobByKind}
              selected={selectedLayer}
              onSelect={setSelectedLayer}
              onToggle={toggleLayer}
              onOpacity={setOpacity}
              poses={artifacts.poses}
              currentTime={currentTime}
              loadingHands={loading.hands}
              loadingYolo={loading.yolo}
              loadingSam={loading.sam}
              loadingDepth={loading.depth}
              depthProgress={depthProgress}
              streams={data.recording.streams}
            />

            <main className="viewport viewport--2d">
              <Viewport
                videoRef={videoRef}
                videoUrl={data.streamUrls['video.mp4']}
                videoSize={videoSize}
                onStageSizeChange={setStageSize}
                stageSize={stageSize}
                layers={effectiveLayers}
                artifacts={artifacts}
                currentTime={currentTime}
                durationS={durationS}
                currentAction={currentAction?.caption ?? '—'}
                recording={data.recording}
              />
              {artifacts.transcript && artifacts.transcript.length > 0 && (
                <div className="vp-caption">
                  {currentCaption
                    ? <span>{currentCaption}</span>
                    : <span className="vp-caption__idle">· · ·</span>}
                </div>
              )}
            </main>
          </>
        )}

        {viewMode === '3d' && data.splatScene && (
          <>
            <aside className="rail rail--scene">
              <SceneControls
                layers={sceneLayers}
                onLayersChange={setSceneLayers}
                cameraMode={cameraMode}
                onCameraModeChange={setCameraMode}
                summary={{
                  numGaussians: data.splatScene.numGaussians,
                  frameCount: data.splatScene.frameCount,
                  fps: data.splatScene.fps,
                }}
              />
            </aside>
            <main className="viewport viewport--scene">
              {/* Hidden but mounted: video.currentTime drives the camera-sync hook
                  while in 3D mode. The timeline scrubber still controls it. */}
              {data.streamUrls['video.mp4'] && (
                <video
                  ref={videoRef}
                  src={data.streamUrls['video.mp4']}
                  style={{ display: 'none' }}
                  playsInline
                  preload="metadata"
                  crossOrigin="anonymous"
                />
              )}
              <SceneView
                ref={sceneHandleRef}
                scene={data.splatScene}
                layers={sceneLayers}
                cameraPath={cameraPath}
                manualControl={manualControl}
              />
            </main>
          </>
        )}

        <Inspector
          data={data}
          activeTab={activeTab}
          onTab={setActiveTab}
          currentAction={currentAction}
          currentTime={currentTime}
          artifacts={artifacts}
        />
      </div>

      <Timeline
        durationS={durationS}
        currentTime={currentTime}
        onSeek={seek}
        isPlaying={isPlaying}
        onPlay={togglePlay}
        speed={speed}
        onSpeed={setSpeed}
        scrubHover={scrubHover}
        onScrubHover={setScrubHover}
        artifacts={artifacts}
        detectedObjects={data.recording.detectedObjects}
      />
    </div>
  )
}

/* ═════════════════════════════════════════════════════════════════════
 *  TOP BAR
 * ═════════════════════════════════════════════════════════════════════ */

function TopBar({
  data, currentSibling, totalTakes, currentAction, frame, currentTime,
  onPrev, onNext, canPrev, canNext, siblingOpen, onSiblingOpen, onPickSibling,
}: {
  data: StudioData
  currentSibling: SiblingRecording | null
  totalTakes: number
  currentAction: string
  frame: number
  currentTime: number
  onPrev: () => void
  onNext: () => void
  canPrev: boolean
  canNext: boolean
  siblingOpen: boolean
  onSiblingOpen: (b: boolean) => void
  onPickSibling: (id: string) => void
}) {
  const closeHref = data.task ? `/lab/tasks/${data.task.id}` : '/lab/dashboard'
  return (
    <header className="topbar">
      <div className="topbar__left">
        <Link href={closeHref} className="topbar__close" aria-label="Close studio">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
          </svg>
        </Link>
        <div className="topbar__crumb">
          <span>recordings</span>
          <span className="topbar__crumb-sep">/</span>
          <span className="topbar__crumb-id">{shortId(data.recording.id)}</span>
          {currentSibling && totalTakes > 1 && (
            <>
              <span className="topbar__crumb-sep">·</span>
              <span className="topbar__crumb-take">take {currentSibling.takeNumber} / {totalTakes}</span>
            </>
          )}
        </div>
        <h1 className="topbar__title">{data.task?.title ?? 'Untitled task'}</h1>
      </div>

      <div className="topbar__center">
        <div className="liveband">
          <span className={`liveband__dot ${data.recording.isScoring ? 'liveband__dot--warm' : ''}`} />
          <span className="liveband__action">{currentAction}</span>
          <span className="liveband__sep" />
          <span className="liveband__meta">f<i>{frame.toString().padStart(5, '0')}</i></span>
          <span className="liveband__meta">t<i>{currentTime.toFixed(2)}s</i></span>
        </div>
      </div>

      <div className="topbar__right">
        {totalTakes > 1 && (
          <div className="takeswitcher">
            <button className="iconbtn iconbtn--ghost" onClick={onPrev} disabled={!canPrev} title="Previous take [">
              <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor"><path d="M8 2L3.5 6 8 10z" /></svg>
            </button>
            <button className="iconbtn takeswitcher__main" onClick={() => onSiblingOpen(!siblingOpen)} aria-expanded={siblingOpen}>
              <span className="takeswitcher__label">
                take <strong>{currentSibling?.takeNumber ?? '?'}</strong>
                <span className="takeswitcher__divider">/</span>
                <span className="takeswitcher__total">{totalTakes}</span>
              </span>
              <svg width="8" height="8" viewBox="0 0 8 8" fill="currentColor">
                <path d={siblingOpen ? 'M1 5l3-3 3 3z' : 'M1 3l3 3 3-3z'} />
              </svg>
            </button>
            <button className="iconbtn iconbtn--ghost" onClick={onNext} disabled={!canNext} title="Next take ]">
              <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor"><path d="M4 2l4.5 4L4 10z" /></svg>
            </button>
            {siblingOpen && (
              <SiblingPopover siblings={data.siblings} onPick={onPickSibling} onClose={() => onSiblingOpen(false)} />
            )}
          </div>
        )}

        <RecordingStatusChip recording={data.recording} />

        <div className="topbar__actions">
          <button className="iconbtn" title="Compare takes">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <rect x="1.5" y="2.5" width="4.5" height="9" rx="0.5" stroke="currentColor" strokeWidth="1.2" />
              <rect x="8" y="2.5" width="4.5" height="9" rx="0.5" stroke="currentColor" strokeWidth="1.2" />
            </svg>
          </button>
          <button className="iconbtn iconbtn--primary" title="Export">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M7 1.5v8M3.5 6L7 9.5 10.5 6M2 12h10" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span>Export</span>
          </button>
        </div>
      </div>
    </header>
  )
}

function RecordingStatusChip({ recording }: { recording: StudioData['recording'] }) {
  if (recording.isScoring) return <button className="chip chip--warn"><span className="chip__dot" /> scoring</button>
  if (recording.score !== null) {
    return (
      <button className="chip chip--score">
        <strong className="chip__score">{recording.score}</strong>
        <span className="chip__score-of">/10</span>
        {recording.success != null && (
          <span className={`chip__pill ${recording.success ? 'chip__pill--ok' : 'chip__pill--miss'}`}>
            {recording.success ? '✓' : '✕'}
          </span>
        )}
      </button>
    )
  }
  if (recording.status === 'analyzing') return <button className="chip chip--warn"><span className="chip__dot" /> analyzing</button>
  if (recording.status === 'analysis_failed') return <button className="chip chip--error"><span className="chip__dot" /> failed</button>
  return <button className="chip"><span className="chip__dot" /> {recording.status}</button>
}

function SiblingPopover({ siblings, onPick, onClose }: {
  siblings: SiblingRecording[]
  onPick: (id: string) => void
  onClose: () => void
}) {
  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      const target = e.target as HTMLElement
      if (!target.closest('.siblingpop') && !target.closest('.takeswitcher__main')) onClose()
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [onClose])

  return (
    <div className="siblingpop">
      <div className="siblingpop__head">
        <span>all takes · {siblings.length}</span>
        <span className="siblingpop__hint">[ / ] to switch</span>
      </div>
      <ul className="siblingpop__list">
        {siblings.map((s) => (
          <li
            key={s.id}
            className={`siblingrow ${s.isCurrent ? 'siblingrow--current' : ''}`}
            onClick={() => !s.isCurrent && onPick(s.id)}
          >
            <span className="siblingrow__take">#{s.takeNumber.toString().padStart(2, '0')}</span>
            <span className="siblingrow__id">{shortId(s.id)}</span>
            <span className="siblingrow__dur">{s.durationMs ? `${(s.durationMs / 1000).toFixed(1)}s` : '—'}</span>
            <span className="siblingrow__score">
              {s.score != null ? <><strong>{s.score}</strong><i>/10</i></> : <span className="siblingrow__dim">—</span>}
            </span>
            <span className={`siblingrow__status siblingrow__status--${s.status}`}>{s.status}</span>
            {s.success != null && (
              <span className={`siblingrow__flag ${s.success ? 'siblingrow__flag--ok' : 'siblingrow__flag--miss'}`}>
                {s.success ? '✓' : '✕'}
              </span>
            )}
          </li>
        ))}
      </ul>
    </div>
  )
}

/* ═════════════════════════════════════════════════════════════════════
 *  LEFT RAIL
 * ═════════════════════════════════════════════════════════════════════ */

function LeftRail({
  layers, jobByKind, selected, onSelect, onToggle, onOpacity,
  poses, currentTime, loadingHands, loadingYolo, loadingSam, loadingDepth, depthProgress, streams,
}: {
  layers: LayerState
  jobByKind: Map<AnalysisJobKind, AnalysisJobPayload>
  selected: LayerKey
  onSelect: (k: LayerKey) => void
  onToggle: (k: LayerKey) => void
  onOpacity: (k: LayerKey, v: number) => void
  poses: PoseSample[] | null
  currentTime: number
  loadingHands: boolean | undefined
  loadingYolo: boolean | undefined
  loadingSam: boolean | undefined
  loadingDepth: boolean | undefined
  depthProgress: number | null
  streams: string[]
}) {
  return (
    <aside className="rail">
      <div className="rail__section">
        <div className="rail__heading">
          <span>Layers</span>
          <span className="rail__count">
            {Object.values(layers).filter((l) => l.visible).length} / {LAYER_SPECS.length}
          </span>
        </div>

        <ul className="layerlist">
          {LAYER_SPECS.map((layer) => {
            const s = layers[layer.key]
            const isSelected = selected === layer.key
            const job = layer.jobKind ? jobByKind.get(layer.jobKind) : null
            const unavailable = layer.jobKind && (!job || job.status !== 'succeeded')
            const isLoading =
              (layer.key === 'hands' && loadingHands) ||
              (layer.key === 'objects' && loadingYolo) ||
              (layer.key === 'masks' && loadingSam && s.visible) ||
              (layer.key === 'depth' && loadingDepth && s.visible)
            return (
              <li
                key={layer.key}
                className={`layer ${isSelected ? 'layer--selected' : ''} ${!s.visible ? 'layer--off' : ''} ${unavailable ? 'layer--unavailable' : ''}`}
                onClick={() => onSelect(layer.key)}
              >
                <button
                  className="layer__eye"
                  onClick={(e) => { e.stopPropagation(); if (!unavailable) onToggle(layer.key) }}
                  disabled={!!unavailable}
                  aria-label={s.visible ? 'Hide' : 'Show'}
                >
                  {s.visible ? (
                    <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                      <path d="M1 6s2-3.5 5-3.5S11 6 11 6s-2 3.5-5 3.5S1 6 1 6z" stroke="currentColor" strokeWidth="1" />
                      <circle cx="6" cy="6" r="1.4" fill="currentColor" />
                    </svg>
                  ) : (
                    <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                      <path d="M2 2l8 8" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
                      <path d="M1 6s2-3.5 5-3.5c.7 0 1.3.2 1.9.4M11 6s-2 3.5-5 3.5c-.7 0-1.3-.2-1.9-.4" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
                    </svg>
                  )}
                </button>
                <span className="layer__swatch" style={{ background: layer.swatch }} />
                <div className="layer__body">
                  <div className="layer__name">{layer.label}</div>
                  <div className="layer__source">
                    {unavailable ? (
                      <span className="layer__nodata">no artifact</span>
                    ) : isLoading ? (
                      layer.key === 'depth' && depthProgress != null
                        ? <span className="layer__loading">decoding · {(depthProgress * 100).toFixed(0)}%</span>
                        : <span className="layer__loading">loading…</span>
                    ) : layer.source}
                  </div>
                </div>
                <span className="layer__shortcut">{layer.shortcut}</span>
                {isSelected && !unavailable && (
                  <div className="layer__opacity" onClick={(e) => e.stopPropagation()}>
                    <span className="layer__opacity-label">Opacity</span>
                    <input
                      type="range" min={0} max={1} step={0.01} value={s.opacity}
                      onChange={(e) => onOpacity(layer.key, parseFloat(e.target.value))}
                    />
                    <span className="layer__opacity-value">{Math.round(s.opacity * 100)}%</span>
                  </div>
                )}
              </li>
            )
          })}
        </ul>
      </div>

      <div className="rail__section">
        <div className="rail__heading">
          <span>Raw streams</span>
          <span className="rail__count">{streams.length}</span>
        </div>
        <ul className="streamlist">
          {streams.map((s) => (
            <li key={s} className="streamlist__item">
              <span className="streamlist__dot" />
              <span className="streamlist__name">{s}</span>
            </li>
          ))}
        </ul>
      </div>

      <div className="rail__section rail__section--bottom">
        <div className="rail__heading"><span>6DOF · ARKit</span></div>
        <MiniViewPose currentTime={currentTime} poses={poses} />
      </div>
    </aside>
  )
}

function MiniViewPose({ currentTime, poses }: { currentTime: number; poses: PoseSample[] | null }) {
  let yaw = 0, pitch = 0, roll = 0, hasReal = false
  if (poses && poses.length > 0) {
    const idx = searchLowerBound(poses, currentTime, (p) => p.t)
    const pick = idx >= 0 ? poses[idx] : poses[0]
    yaw = pick.yaw; pitch = pick.pitch; roll = pick.roll
    hasReal = true
  }
  return (
    <div className="minicard">
      <div className="minicard__head">
        <span>orientation</span>
        <span className="minicard__unit">{hasReal ? 'live' : '—'}</span>
      </div>
      <div className="posewidget">
        <div
          className="phone3d"
          style={{ transform: `rotateX(${pitch}deg) rotateY(${yaw}deg) rotateZ(${roll}deg)` }}
        >
          {/* SCREEN side (FaceID camera, faces user) */}
          <div className="phone3d__face phone3d__face--front">
            <span className="phone3d__screen" />
          </div>
          {/* CAMERA side — the rear lens that captures the scene */}
          <div className="phone3d__face phone3d__face--back">
            <span className="phone3d__lens" />
            <span className="phone3d__lens-ring" />
          </div>
          <div className="phone3d__face phone3d__face--top" />
          <div className="phone3d__face phone3d__face--bottom" />
          <div className="phone3d__face phone3d__face--left" />
          <div className="phone3d__face phone3d__face--right" />
        </div>
      </div>
      <div className="minicard__foot minicard__foot--cols">
        <span>y <i>{yaw.toFixed(1)}</i></span>
        <span>p <i>{pitch.toFixed(1)}</i></span>
        <span>r <i>{roll.toFixed(1)}</i></span>
      </div>
    </div>
  )
}

/* ═════════════════════════════════════════════════════════════════════
 *  VIEWPORT + OVERLAYS (real data)
 * ═════════════════════════════════════════════════════════════════════ */

/** Compute the letterboxed content rect of a video with object-fit: contain inside a stage. */
function videoContentRect(stageW: number, stageH: number, vw: number, vh: number) {
  if (stageW <= 0 || stageH <= 0 || vw <= 0 || vh <= 0) {
    return { x: 0, y: 0, width: stageW, height: stageH }
  }
  const sa = stageW / stageH
  const va = vw / vh
  if (va > sa) {
    const h = stageW / va
    return { x: 0, y: (stageH - h) / 2, width: stageW, height: h }
  } else {
    const w = stageH * va
    return { x: (stageW - w) / 2, y: 0, width: w, height: stageH }
  }
}

function Viewport({
  videoRef, videoUrl, videoSize, stageSize, onStageSizeChange,
  layers, artifacts, currentTime, durationS, currentAction, recording,
}: {
  videoRef: React.RefObject<HTMLVideoElement | null>
  videoUrl: string | null
  videoSize: { vw: number; vh: number } | null
  stageSize: { w: number; h: number } | null
  onStageSizeChange: (s: { w: number; h: number }) => void
  layers: LayerState
  artifacts: ArtifactState
  currentTime: number
  durationS: number
  currentAction: string
  recording: StudioData['recording']
}) {
  const stageRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = stageRef.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect()
      onStageSizeChange({ w: r.width, h: r.height })
    })
    ro.observe(el)
    const r = el.getBoundingClientRect()
    onStageSizeChange({ w: r.width, h: r.height })
    return () => ro.disconnect()
  }, [onStageSizeChange])

  const contentRect = useMemo(() => {
    if (!stageSize || !videoSize) return null
    return videoContentRect(stageSize.w, stageSize.h, videoSize.vw, videoSize.vh)
  }, [stageSize, videoSize])

  return (
    <div className="vp">
      <div className="vp__chrome vp__chrome--tl">
        <span>SCENE</span>
        <span className="vp__chrome-meta">
          {recording.deviceModel ?? 'unknown'} · {recording.durationMs ? `${(recording.durationMs / 1000).toFixed(1)}s` : '—'}
          {videoSize ? ` · ${videoSize.vw}×${videoSize.vh}` : ''}
        </span>
      </div>
      <div className="vp__chrome vp__chrome--tr">
        <span className="vp__chrome-meta">recording <i>{shortId(recording.id)}</i></span>
      </div>
      <div className="vp__chrome vp__chrome--bl">
        <span className="vp__chrome-meta">action <i>{currentAction}</i></span>
      </div>
      <div className="vp__chrome vp__chrome--br">
        <span className="vp__chrome-meta">{fmtTime(currentTime)} / {fmtTime(durationS)}</span>
      </div>

      <CornerBracket pos="tl" />
      <CornerBracket pos="tr" />
      <CornerBracket pos="bl" />
      <CornerBracket pos="br" />

      <div ref={stageRef} className="vp__stage">
        {videoUrl ? (
          <video
            ref={videoRef}
            src={videoUrl}
            className="vp__video"
            style={{ opacity: layers.rgb.visible ? layers.rgb.opacity : 0 }}
            playsInline
            preload="metadata"
            crossOrigin="anonymous"
          />
        ) : (
          <div className="vp__novideo"><span>video unavailable</span></div>
        )}

        {/* All overlays positioned to match the letterboxed content rect */}
        {contentRect && (
          <div
            className="vp__overlay-frame"
            style={{
              position: 'absolute',
              left: contentRect.x,
              top: contentRect.y,
              width: contentRect.width,
              height: contentRect.height,
              pointerEvents: 'none',
            }}
          >
            {/* depth sits beneath ML overlays */}
            {layers.depth.visible && artifacts.depth && (
              <DepthOverlay depth={artifacts.depth} currentTime={currentTime} opacity={layers.depth.opacity} />
            )}
            {layers.objects.visible && artifacts.yolo && (
              <YoloOverlay yolo={artifacts.yolo} currentTime={currentTime} opacity={layers.objects.opacity} />
            )}
            {layers.masks.visible && artifacts.sam && videoSize && (
              <SamOverlay sam={artifacts.sam} videoSize={videoSize} currentTime={currentTime} opacity={layers.masks.opacity} />
            )}
            {layers.hands.visible && artifacts.hands && videoSize && (
              <HandOverlay hands={artifacts.hands} videoSize={videoSize} currentTime={currentTime} opacity={layers.hands.opacity} />
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function CornerBracket({ pos }: { pos: 'tl' | 'tr' | 'bl' | 'br' }) {
  return <span className={`vp__bracket vp__bracket--${pos}`} />
}

/* ─── Turbo colour-map LUT (256 entries) ─── */
// Polynomial approximation of Google's turbo colormap (Mikhail Polyanskiy 2019).
// Pre-baked once at module load → 256 × 3 bytes for O(1) lookup.
const TURBO_LUT = (() => {
  const lut = new Uint8Array(256 * 3)
  for (let i = 0; i < 256; i++) {
    const x = i / 255
    const r = 34.61 + x * (1172.33 - x * (10793.56 - x * (33300.12 - x * (38394.49 - x * 14825.05))))
    const g = 23.31 + x * (557.33 + x * (1225.33 - x * (3574.96 - x * (1073.77 + x * 707.56))))
    const b = 27.2 + x * (3211.1 - x * (15327.97 - x * (27814.0 - x * (22569.18 - x * 6838.66))))
    lut[i * 3 + 0] = Math.max(0, Math.min(255, Math.round(r)))
    lut[i * 3 + 1] = Math.max(0, Math.min(255, Math.round(g)))
    lut[i * 3 + 2] = Math.max(0, Math.min(255, Math.round(b)))
  }
  return lut
})()

/* ─── Depth overlay (real LiDAR projected, Turbo colormap on canvas) ─── */
function DepthOverlay({ depth, currentTime, opacity }: {
  depth: DepthArtifact
  currentTime: number
  opacity: number
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const { width: w, height: h, vmin, vmax } = depth

  // Find nearest depth frame; if too stale, skip
  const sample = useMemo(() => {
    const idx = searchLowerBound(depth.frames, currentTime, (f) => f.t)
    if (idx < 0) return null
    const cur = depth.frames[idx]
    if (currentTime - cur.t > 0.5) return null
    return cur
  }, [depth.frames, currentTime])

  useEffect(() => {
    const c = canvasRef.current
    if (!c || !sample) return
    const ctx = c.getContext('2d')
    if (!ctx) return
    const img = ctx.createImageData(w, h)
    const px = img.data
    const data = sample.data
    const range = Math.max(1e-6, vmax - vmin)
    const n = w * h
    // Note: iPhone capture stores depth in camera (landscape) orientation;
    // displayed video may be rotated. Here we trust that the post-encode video
    // matches the depth orientation. If misaligned for some captures, this is
    // where we'd insert a rotation. For now, direct mapping.
    for (let i = 0; i < n; i++) {
      const d = data[i]
      const j = i * 4
      if (!isFinite(d) || d <= 0) {
        // invalid samples are transparent
        px[j + 3] = 0
        continue
      }
      let t = (d - vmin) / range
      if (t < 0) t = 0
      else if (t > 1) t = 1
      const k = Math.floor(t * 255) * 3
      px[j + 0] = TURBO_LUT[k]
      px[j + 1] = TURBO_LUT[k + 1]
      px[j + 2] = TURBO_LUT[k + 2]
      px[j + 3] = 255
    }
    ctx.putImageData(img, 0, 0)
  }, [sample, w, h, vmin, vmax])

  return (
    <div
      className="vp__overlay-svg"
      style={{ position: 'absolute', inset: 0, opacity }}
    >
      <canvas
        ref={canvasRef}
        width={w}
        height={h}
        style={{
          width: '100%',
          height: '100%',
          imageRendering: 'auto',
          mixBlendMode: 'screen',
          display: 'block',
        }}
      />
      {/* small colorbar legend in the bottom-right corner of the overlay frame */}
      <DepthLegend vmin={vmin} vmax={vmax} />
    </div>
  )
}

function DepthLegend({ vmin, vmax }: { vmin: number; vmax: number }) {
  return (
    <div
      style={{
        position: 'absolute',
        right: 10,
        bottom: 10,
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        background: 'rgba(10,10,12,0.7)',
        padding: '4px 8px',
        border: '1px solid rgba(255,255,255,0.12)',
        fontFamily: 'var(--font-geist-mono), monospace',
        fontSize: 10,
        color: '#e8e8ea',
        letterSpacing: '0.04em',
      }}
    >
      <span>{vmin.toFixed(2)}m</span>
      <div
        style={{
          width: 96,
          height: 8,
          background:
            'linear-gradient(90deg,' +
            ' rgb(48,18,59), rgb(70,107,227), rgb(33,170,213),' +
            ' rgb(96,212,150), rgb(180,222,44), rgb(248,160,29), rgb(170,3,16))',
          border: '1px solid rgba(255,255,255,0.15)',
        }}
      />
      <span>{vmax.toFixed(2)}m</span>
    </div>
  )
}

/* ─── YOLO overlay (real per-frame bboxes) ─── */
function YoloOverlay({ yolo, currentTime, opacity }: {
  yolo: { frames: YoloFrame[]; meta: YoloMeta }
  currentTime: number
  opacity: number
}) {
  const idx = searchLowerBound(yolo.frames, currentTime, (f) => f.t)
  // If too far from this sample (no detections nearby), skip
  if (idx < 0) return null
  const sample = yolo.frames[idx]
  if (currentTime - sample.t > 0.2) return null // stale > 200ms = no current detection

  const { origW, origH } = yolo.meta
  return (
    <svg
      className="vp__overlay-svg"
      viewBox={`0 0 ${origW} ${origH}`}
      preserveAspectRatio="none"
      style={{ opacity, position: 'absolute', inset: 0, width: '100%', height: '100%' }}
    >
      {sample.instances.map((inst, i) => {
        const color = colorForClass(inst.class_name)
        const [x1, y1, x2, y2] = inst.box
        const w = x2 - x1, h = y2 - y1
        const tagW = inst.class_name.length * 14 + 90
        return (
          <g key={i}>
            <rect x={x1} y={y1} width={w} height={h} fill="none" stroke={color} strokeWidth={3} />
            <path d={`M${x1} ${y1 + 18} L${x1} ${y1} L${x1 + 18} ${y1}`} stroke={color} strokeWidth={4} fill="none" />
            <path d={`M${x2 - 18} ${y1} L${x2} ${y1} L${x2} ${y1 + 18}`} stroke={color} strokeWidth={4} fill="none" />
            <path d={`M${x1} ${y2 - 18} L${x1} ${y2} L${x1 + 18} ${y2}`} stroke={color} strokeWidth={4} fill="none" />
            <path d={`M${x2 - 18} ${y2} L${x2} ${y2} L${x2} ${y2 - 18}`} stroke={color} strokeWidth={4} fill="none" />
            <g transform={`translate(${x1}, ${Math.max(0, y1 - 24)})`}>
              <rect width={tagW} height="22" fill={color} />
              <text x="6" y="16" fontSize="14" fontFamily="var(--font-geist-mono)" fontWeight="600" fill="#0a0a0c" letterSpacing="0.04em">
                {inst.class_name.toUpperCase()}
              </text>
              <text x={tagW - 50} y="16" fontSize="13" fontFamily="var(--font-geist-mono)" fill="#0a0a0c" letterSpacing="0.02em">
                {(inst.conf * 100).toFixed(0)}%
              </text>
            </g>
          </g>
        )
      })}
    </svg>
  )
}

/* ─── Hand overlay (real 21-landmark skeletons, interpolated between samples) ─── */
const HAND_BONES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
]

// Fingertip indices in MediaPipe's 21-point hand model
const FINGERTIPS = new Set([4, 8, 12, 16, 20])

function HandOverlay({ hands, videoSize, currentTime, opacity }: {
  hands: { samples: HandFrame[]; sourceFps: number; sampleFps: number; stride: number }
  videoSize: { vw: number; vh: number }
  currentTime: number
  opacity: number
}) {
  const idx = searchLowerBound(hands.samples, currentTime, (s) => s.t)
  if (idx < 0) return null
  const cur = hands.samples[idx]
  const next = hands.samples[idx + 1]
  const samplePeriod = 1 / hands.sampleFps
  let frame: HandFrame
  if (next && next.t - cur.t < samplePeriod * 3) {
    const a = (currentTime - cur.t) / (next.t - cur.t)
    const aa = Math.max(0, Math.min(1, a))
    const hands_ = cur.hands.map((h, hi) => {
      const nh = next.hands[hi]
      if (!nh || nh.handedness !== h.handedness) return h
      return {
        ...h,
        landmarks: h.landmarks.map((lm, li) => {
          const nlm = nh.landmarks[li]
          return { x: lerp(lm.x, nlm.x, aa), y: lerp(lm.y, nlm.y, aa), z: lerp(lm.z, nlm.z, aa) }
        }),
      }
    })
    frame = { t: cur.t, hands: hands_ }
  } else {
    frame = cur
  }
  if (currentTime - cur.t > samplePeriod * 4) return null

  const { vw, vh } = videoSize
  // Stroke widths in pixels (vector-effect makes them resolution-independent)
  const BONE_W = 1.5
  const JOINT_R = 3.5       // px in non-scaling coords
  const TIP_R = 5
  const WRIST_R = 7

  // Bone color: studio accent
  const ACCENT = '#c2e02b'

  return (
    <svg
      className="vp__overlay-svg"
      viewBox={`0 0 ${vw} ${vh}`}
      preserveAspectRatio="none"
      style={{ opacity, position: 'absolute', inset: 0, width: '100%', height: '100%' }}
    >
      {frame.hands.map((hand, hi) => (
        <g key={hi}>
          {/* bones — hairline accent, butt-capped square ends look cleaner at 1.5px */}
          {HAND_BONES.map(([a, b], bi) => {
            const la = hand.landmarks[a]
            const lb = hand.landmarks[b]
            if (!la || !lb) return null
            return (
              <line
                key={bi}
                x1={la.x * vw} y1={la.y * vh}
                x2={lb.x * vw} y2={lb.y * vh}
                stroke={ACCENT}
                strokeWidth={BONE_W}
                strokeLinecap="round"
                vectorEffect="non-scaling-stroke"
              />
            )
          })}
          {/* landmarks */}
          {hand.landmarks.map((lm, li) => {
            const cx = lm.x * vw
            const cy = lm.y * vh
            if (li === 0) {
              // Wrist: outlined ring, dark center → reads as the anchor
              return (
                <g key={li}>
                  <circle cx={cx} cy={cy} r={WRIST_R} fill="#0a0a0c" stroke={ACCENT} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
                  <circle cx={cx} cy={cy} r={1.8} fill={ACCENT} vectorEffect="non-scaling-stroke" />
                </g>
              )
            }
            if (FINGERTIPS.has(li)) {
              // Fingertips: solid accent, larger
              return (
                <circle key={li} cx={cx} cy={cy} r={TIP_R} fill={ACCENT} vectorEffect="non-scaling-stroke" />
              )
            }
            // Inner joints: white-filled with accent stroke
            return (
              <circle
                key={li}
                cx={cx} cy={cy}
                r={JOINT_R}
                fill="#fafafa"
                stroke={ACCENT}
                strokeWidth={1}
                vectorEffect="non-scaling-stroke"
              />
            )
          })}
        </g>
      ))}
    </svg>
  )
}

/* ─── SAM mask overlay (real polygons) ─── */
function SamOverlay({ sam, videoSize, currentTime, opacity }: {
  sam: { frames: SamFrame[]; meta: YoloMeta; textPrompts: string[] }
  videoSize: { vw: number; vh: number }
  currentTime: number
  opacity: number
}) {
  const idx = searchLowerBound(sam.frames, currentTime, (f) => f.t)
  if (idx < 0) return null
  const sample = sam.frames[idx]
  if (currentTime - sample.t > 0.2) return null

  const { vw, vh } = videoSize

  return (
    <svg
      className="vp__overlay-svg"
      viewBox={`0 0 ${vw} ${vh}`}
      preserveAspectRatio="none"
      style={{ opacity, position: 'absolute', inset: 0, width: '100%', height: '100%' }}
    >
      {sample.instances.map((inst, i) => {
        const color = colorForClass(inst.class_name)
        if (!inst.polygon || inst.polygon.length < 3) return null
        const d = inst.polygon
          .map(([x, y], idx2) => `${idx2 === 0 ? 'M' : 'L'}${x * vw} ${y * vh}`)
          .join(' ') + ' Z'
        return (
          <g key={i}>
            {/* solid mask fill — strongly visible */}
            <path
              d={d}
              fill={color}
              fillOpacity={0.55}
              stroke={color}
              strokeWidth={2.5}
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
            />
            {/* a brighter contour on top for definition */}
            <path
              d={d}
              fill="none"
              stroke="#fafafa"
              strokeOpacity={0.85}
              strokeWidth={1}
              strokeLinejoin="round"
              strokeDasharray="6 4"
              vectorEffect="non-scaling-stroke"
            />
          </g>
        )
      })}
    </svg>
  )
}

/* ═════════════════════════════════════════════════════════════════════
 *  INSPECTOR
 * ═════════════════════════════════════════════════════════════════════ */

function Inspector({
  data, activeTab, onTab, currentAction, currentTime, artifacts,
}: {
  data: StudioData
  activeTab: 'eval' | 'task' | 'jobs' | 'meta'
  onTab: (t: 'eval' | 'task' | 'jobs' | 'meta') => void
  currentAction: ActionSeg | null
  currentTime: number
  artifacts: ArtifactState
}) {
  return (
    <aside className="inspector">
      <div className="inspector__tabs">
        {(['eval', 'task', 'jobs', 'meta'] as const).map((t) => (
          <button key={t} className={`tab ${activeTab === t ? 'tab--active' : ''}`} onClick={() => onTab(t)}>
            {t === 'task' ? 'Task' : t === 'eval' ? 'Eval' : t === 'jobs' ? 'Pipeline' : 'Meta'}
          </button>
        ))}
      </div>
      <div className="inspector__body">
        {activeTab === 'eval' && <EvalPanel data={data} currentAction={currentAction} currentTime={currentTime} artifacts={artifacts} />}
        {activeTab === 'task' && <TaskPanel data={data} />}
        {activeTab === 'jobs' && <JobsPanel data={data} />}
        {activeTab === 'meta' && <MetaPanel data={data} />}
      </div>
    </aside>
  )
}

function EvalPanel({ data, currentAction, currentTime, artifacts }: {
  data: StudioData
  currentAction: ActionSeg | null
  currentTime: number
  artifacts: ArtifactState
}) {
  const r = data.recording
  const scorePct = r.score != null ? r.score / 10 : 0

  // Current per-time YOLO sample for "live detections at cursor"
  const yoloLive = useMemo(() => {
    if (!artifacts.yolo) return []
    const idx = searchLowerBound(artifacts.yolo.frames, currentTime, (f) => f.t)
    if (idx < 0) return []
    const sample = artifacts.yolo.frames[idx]
    if (currentTime - sample.t > 0.3) return []
    return sample.instances
  }, [artifacts.yolo, currentTime])

  return (
    <div className="panel">
      <section className="panelblock">
        <div className="panelblock__head">gemini evaluation</div>
        {r.score == null && r.isScoring && (
          <div className="bigstate bigstate--warn">
            <span className="bigstate__pulse" />
            <div><strong>Scoring in progress…</strong><p>Gemini is reviewing this take.</p></div>
          </div>
        )}
        {r.score == null && !r.isScoring && (
          <div className="bigstate"><strong>Not scored</strong><p>This recording has no Gemini score yet.</p></div>
        )}
        {r.score != null && (
          <div className="score">
            <div className="score__ring">
              <svg viewBox="0 0 80 80">
                <circle cx="40" cy="40" r="34" stroke="rgba(255,255,255,0.08)" strokeWidth="3" fill="none" />
                <circle
                  cx="40" cy="40" r="34" stroke="#c2e02b" strokeWidth="3" fill="none"
                  strokeDasharray={`${2 * Math.PI * 34 * scorePct} ${2 * Math.PI * 34}`}
                  strokeLinecap="round" transform="rotate(-90 40 40)"
                />
              </svg>
              <div className="score__num"><strong>{r.score}</strong><span>/10</span></div>
            </div>
            <div className="score__meta">
              {r.success != null && (
                <span className={`chip ${r.success ? 'chip--success' : 'chip--error'}`}>
                  <span className="chip__dot" /> {r.success ? 'success' : 'failure'}
                </span>
              )}
              {r.summary && <p className="score__summary">{r.summary}</p>}
            </div>
          </div>
        )}
      </section>

      {(r.successReasoning || r.scoreReasoning) && (
        <section className="panelblock">
          <div className="panelblock__head">justification</div>
          {r.successReasoning && (
            <div className="just">
              <div className="just__label">success reasoning</div>
              <p className="just__body">{r.successReasoning}</p>
            </div>
          )}
          {r.scoreReasoning && (
            <div className="just">
              <div className="just__label">score reasoning</div>
              <p className="just__body">{r.scoreReasoning}</p>
            </div>
          )}
        </section>
      )}

      {currentAction && (
        <section className="panelblock">
          <div className="panelblock__head">current action · t={currentTime.toFixed(2)}s</div>
          <div className="actioncard">
            <div className="actioncard__head">
              <span className={`actioncard__hand actioncard__hand--${currentAction.hand}`}>
                {currentAction.hand}
              </span>
              <span className="actioncard__caption">{currentAction.caption}</span>
              <span className="actioncard__conf">{(currentAction.confidence * 100).toFixed(0)}%</span>
            </div>
            <p className="actioncard__reason">{currentAction.reason}</p>
            <div className="actioncard__meta">
              <span>{currentAction.start.toFixed(2)}s → {currentAction.end.toFixed(2)}s</span>
              {currentAction.object && <span>obj: <i>{currentAction.object}</i></span>}
              {currentAction.meaningful && <span className="actioncard__flag">meaningful</span>}
            </div>
          </div>
        </section>
      )}

      <section className="panelblock">
        <div className="panelblock__head">
          {yoloLive.length > 0
            ? `live detections · ${yoloLive.length} at cursor`
            : `summary detections · ${r.detectedObjects.length}`}
        </div>
        {yoloLive.length > 0 ? (
          <ul className="detected">
            {yoloLive.slice(0, 8).map((inst, i) => {
              const color = colorForClass(inst.class_name)
              return (
                <li key={i} className="detected__item">
                  <span className="detected__dot" style={{ background: color }} />
                  <span className="detected__name">{inst.class_name}</span>
                  <span className="detected__count">live</span>
                  <span className="detected__bar">
                    <span className="detected__bar-fill" style={{ width: `${inst.conf * 100}%`, background: color }} />
                  </span>
                  <span className="detected__conf">{(inst.conf * 100).toFixed(0)}%</span>
                </li>
              )
            })}
          </ul>
        ) : r.detectedObjects.length === 0 ? (
          <p className="dim">No objects detected.</p>
        ) : (
          <ul className="detected">
            {[...r.detectedObjects]
              .sort((a, b) => (b.max_confidence ?? 0) - (a.max_confidence ?? 0))
              .slice(0, 8)
              .map((d, i) => {
                const conf = d.max_confidence ?? 0
                const color = colorForClass(d.class_name ?? 'unknown')
                return (
                  <li key={`${d.class_name}-${i}`} className="detected__item">
                    <span className="detected__dot" style={{ background: color }} />
                    <span className="detected__name">{d.class_name ?? 'unknown'}</span>
                    <span className="detected__count">×{d.count ?? 0}</span>
                    <span className="detected__bar">
                      <span className="detected__bar-fill" style={{ width: `${conf * 100}%`, background: color }} />
                    </span>
                    <span className="detected__conf">{(conf * 100).toFixed(0)}%</span>
                  </li>
                )
              })}
          </ul>
        )}
      </section>
    </div>
  )
}

function TaskPanel({ data }: { data: StudioData }) {
  const t = data.task
  return (
    <div className="panel">
      <section className="panelblock">
        <div className="panelblock__head">task</div>
        {t ? (
          <>
            <p className="panelblock__lede">{t.title ?? 'Untitled'}</p>
            {t.description && <p className="panelblock__body">{t.description}</p>}
          </>
        ) : <p className="dim">No task linked to this recording.</p>}
      </section>
      {t && t.requiredCapabilities.length > 0 && (
        <section className="panelblock">
          <div className="panelblock__head">required capabilities</div>
          <div className="tagrow">
            {t.requiredCapabilities.map((cap) => (
              <span key={cap} className="tag tag--cap">{cap}</span>
            ))}
          </div>
        </section>
      )}
      {t && t.objects.length > 0 && (
        <section className="panelblock">
          <div className="panelblock__head">target objects</div>
          <div className="tagrow">
            {t.objects.map((o) => <span key={o} className="tag tag--ware">{o}</span>)}
          </div>
        </section>
      )}
      {t && t.bountyAmount != null && (
        <section className="panelblock">
          <div className="panelblock__head">payout</div>
          <div className="payout">
            <div className="payout__amount">
              <strong>${t.bountyAmount.toFixed(2)}</strong>
              <span>per submission · {t.dataType ?? 'data'}</span>
            </div>
          </div>
        </section>
      )}
      {t && (
        <section className="panelblock">
          <Link href={`/lab/tasks/${t.id}`} className="ghostlink">← Back to task board</Link>
        </section>
      )}
    </div>
  )
}

function JobsPanel({ data }: { data: StudioData }) {
  const ordered: AnalysisJobKind[] = ['gemini_eval', 'mediapipe_hands', 'yolo_objects', 'sam_segments', 'temporal_actions']
  return (
    <div className="panel">
      <section className="panelblock">
        <div className="panelblock__head">analyzer pipeline</div>
        <ul className="pipeline">
          {ordered.map((kind) => {
            const job = data.jobs.find((j) => j.kind === kind)
            const label = JOB_LABEL[kind]
            if (!job) return (
              <li key={kind} className="pipeline__row pipeline__row--missing">
                <span className="pipeline__dot pipeline__dot--pending" />
                <span className="pipeline__name">{label}</span>
                <span className="pipeline__time">not run</span>
              </li>
            )
            const duration = job.started_at && job.finished_at
              ? `${((new Date(job.finished_at).getTime() - new Date(job.started_at).getTime()) / 1000).toFixed(1)}s`
              : job.status
            return (
              <li key={kind} className="pipeline__row">
                <span className={`pipeline__dot pipeline__dot--${job.status}`} />
                <div className="pipeline__main">
                  <div className="pipeline__name">{label}</div>
                  {job.error && <div className="pipeline__error">{job.error}</div>}
                </div>
                <span className="pipeline__time">{duration}</span>
                {job.signedUrl && (
                  <a href={job.signedUrl} target="_blank" rel="noreferrer" className="pipeline__link" title="Download artifact">↓</a>
                )}
              </li>
            )
          })}
        </ul>
      </section>
    </div>
  )
}

function MetaPanel({ data }: { data: StudioData }) {
  const r = data.recording
  return (
    <div className="panel">
      <section className="panelblock">
        <div className="panelblock__head">recording</div>
        <dl className="kv">
          <div><dt>id</dt><dd className="mono">{r.id}</dd></div>
          <div><dt>status</dt><dd>{r.status}</dd></div>
          <div><dt>collected by</dt><dd>{r.collectorName ?? (r.collectorId ? shortId(r.collectorId) : '—')}</dd></div>
          <div><dt>captured</dt><dd>{fmtDate(r.createdAt)}</dd></div>
          <div><dt>duration</dt><dd>{r.durationMs != null ? `${(r.durationMs / 1000).toFixed(2)} s` : '—'}</dd></div>
          <div><dt>file size</dt><dd>{fmtBytes(r.sizeBytes)}</dd></div>
        </dl>
      </section>
      <section className="panelblock">
        <div className="panelblock__head">device</div>
        <dl className="kv">
          <div><dt>model</dt><dd>{r.deviceModel ?? '—'}</dd></div>
          <div><dt>streams</dt><dd>{r.streams.length > 0 ? r.streams.join(', ') : '—'}</dd></div>
        </dl>
      </section>
      {(r.gpsLat != null && r.gpsLon != null) && (
        <section className="panelblock">
          <div className="panelblock__head">location</div>
          <dl className="kv">
            <div><dt>lat</dt><dd className="mono">{r.gpsLat.toFixed(6)}</dd></div>
            <div><dt>lon</dt><dd className="mono">{r.gpsLon.toFixed(6)}</dd></div>
            {r.gpsAccuracyM != null && <div><dt>accuracy</dt><dd>± {r.gpsAccuracyM.toFixed(1)} m</dd></div>}
          </dl>
          <a className="ghostlink"
            href={`https://www.openstreetmap.org/?mlat=${r.gpsLat}&mlon=${r.gpsLon}#map=17/${r.gpsLat}/${r.gpsLon}`}
            target="_blank" rel="noreferrer"
          >open in map ↗</a>
        </section>
      )}
    </div>
  )
}

/* ═════════════════════════════════════════════════════════════════════
 *  TIMELINE
 * ═════════════════════════════════════════════════════════════════════ */

function Timeline({
  durationS, currentTime, onSeek, isPlaying, onPlay, speed, onSpeed,
  scrubHover, onScrubHover, artifacts, detectedObjects,
}: {
  durationS: number
  currentTime: number
  onSeek: (t: number) => void
  isPlaying: boolean
  onPlay: () => void
  speed: number
  onSpeed: (s: number) => void
  scrubHover: number | null
  onScrubHover: (t: number | null) => void
  artifacts: ArtifactState
  detectedObjects: DetectedObjectRow[]
}) {
  const trackRef = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)

  const LABEL_W = 96
  const seekFromX = useCallback((clientX: number) => {
    const el = trackRef.current
    if (!el) return
    const r = el.getBoundingClientRect()
    const bodyLeft = r.left + LABEL_W
    const bodyWidth = r.width - LABEL_W
    if (bodyWidth <= 0) return
    const ratio = Math.max(0, Math.min(1, (clientX - bodyLeft) / bodyWidth))
    onSeek(ratio * durationS)
  }, [onSeek, durationS])

  useEffect(() => {
    const onMove = (e: MouseEvent) => { if (dragging.current) seekFromX(e.clientX) }
    const onUp = () => { dragging.current = false }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [seekFromX])

  const pct = (currentTime / durationS) * 100
  const tickSpacing = durationS <= 10 ? 1 : durationS <= 30 ? 2 : durationS <= 60 ? 5 : 10

  // Per-class presence ribbons computed from the full per-frame YOLO artifact.
  // For each top-N class, find time spans where the class is detected and
  // group consecutive frames (closing gaps ≤ FRAME_GAP_S) into segments.
  const detectionRibbons = useMemo(() => {
    if (!artifacts.yolo) {
      // fall back to the summary detected_objects (peak markers per class)
      return null
    }
    const N_CLASSES = 4
    const FRAME_GAP_S = 0.35 // bridge brief gaps so a class flashing on/off becomes one segment

    // Total detections per class
    const totals = new Map<string, number>()
    for (const f of artifacts.yolo.frames) {
      for (const inst of f.instances) {
        totals.set(inst.class_name, (totals.get(inst.class_name) ?? 0) + 1)
      }
    }
    const topClasses = [...totals.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, N_CLASSES)
      .map(([n]) => n)

    const ribbons: { className: string; color: string; segments: { start: number; end: number }[] }[] = []
    for (const className of topClasses) {
      const times: number[] = []
      for (const f of artifacts.yolo.frames) {
        if (f.instances.some((i) => i.class_name === className)) times.push(f.t)
      }
      times.sort((a, b) => a - b)
      const segments: { start: number; end: number }[] = []
      if (times.length > 0) {
        let segStart = times[0]
        let segEnd = times[0]
        const tStep = 1 / artifacts.yolo.meta.sourceFps
        for (let i = 1; i < times.length; i++) {
          if (times[i] - segEnd < FRAME_GAP_S) {
            segEnd = times[i]
          } else {
            segments.push({ start: segStart, end: segEnd + tStep })
            segStart = times[i]
            segEnd = times[i]
          }
        }
        segments.push({ start: segStart, end: segEnd + tStep })
      }
      ribbons.push({ className, color: colorForClass(className), segments })
    }
    return ribbons
  }, [artifacts.yolo])

  // Legacy "peak per class" markers from the summary column — used only when
  // the per-frame YOLO artifact hasn't loaded yet.
  const detectionMarkers = useMemo(() => {
    if (artifacts.yolo) return [] // ribbons take over
    return detectedObjects
      .slice(0, 10)
      .map((d) => {
        const frame = typeof d.representative_frame === 'number' ? d.representative_frame : null
        if (frame == null) return null
        const t = frame / 30
        if (t < 0 || t > durationS) return null
        return { t, label: d.class_name ?? '' }
      })
      .filter((x): x is { t: number; label: string } => x != null)
  }, [detectedObjects, durationS, artifacts.yolo])

  // Build IMU signal arrays once
  const imuLanes = useMemo(() => {
    if (!artifacts.imu || artifacts.imu.length === 0) return null
    const tMax = artifacts.imu[artifacts.imu.length - 1].t || 1
    const stride = Math.max(1, Math.floor(artifacts.imu.length / 360))
    const sample = (key: 'ax' | 'ay' | 'az') =>
      artifacts.imu!.filter((_, i) => i % stride === 0).map((r) => ({ t: r.t / tMax, v: r[key] }))
    const ax = sample('ax'), ay = sample('ay'), az = sample('az')
    // Normalize each axis to fill the lane: use the 95th-percentile absolute value
    // so a few large spikes don't squash the rest of the signal flat.
    const allAbs = [
      ...ax.map((p) => Math.abs(p.v)),
      ...ay.map((p) => Math.abs(p.v)),
      ...az.map((p) => Math.abs(p.v)),
    ].sort((a, b) => a - b)
    const p95 = allAbs[Math.floor(allAbs.length * 0.95)] || 0.001
    const scale = 0.92 / p95 // leaves a tiny headroom in the lane
    return {
      ax: ax.map((p) => ({ t: p.t, v: p.v * scale })),
      ay: ay.map((p) => ({ t: p.t, v: p.v * scale })),
      az: az.map((p) => ({ t: p.t, v: p.v * scale })),
    }
  }, [artifacts.imu])

  // Angular speed |ω| from IMU gyro — highlights when the wrist is rotating.
  // Plotted as a single line on a 0..1 lane (no negative values; this is a magnitude).
  const omegaLane = useMemo(() => {
    if (!artifacts.imu || artifacts.imu.length === 0) return null
    const tMax = artifacts.imu[artifacts.imu.length - 1].t || 1
    const stride = Math.max(1, Math.floor(artifacts.imu.length / 360))
    const mags: { t: number; v: number }[] = []
    for (let i = 0; i < artifacts.imu.length; i += stride) {
      const r = artifacts.imu[i]
      const w = Math.sqrt(r.gx * r.gx + r.gy * r.gy + r.gz * r.gz)
      mags.push({ t: r.t / tMax, v: w })
    }
    // Robust scale: use the 95th-percentile so a couple of spikes don't squash everything
    const sorted = mags.map((p) => p.v).sort((a, b) => a - b)
    const p95 = sorted[Math.floor(sorted.length * 0.95)] || 0.001
    const scale = 0.92 / p95
    // Map to [-1,+1] band-space the SignalLine expects; magnitude is positive, so
    // we shift it down so 0 sits at the lane bottom and the peak fills the lane.
    return mags.map((p) => ({ t: p.t, v: p.v * scale * 2 - 1 }))
  }, [artifacts.imu])

  return (
    <footer className="timeline">
      <div className="timeline__controls">
        <button className="ctrlbtn" onClick={() => onSeek(0)} aria-label="Restart">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor"><path d="M2 2v8h1.5V6.5L9.5 10V2L3.5 5.5V2z" /></svg>
        </button>
        <button className="ctrlbtn" onClick={() => onSeek(currentTime - 1 / 30)} aria-label="Prev frame">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor"><path d="M3 2v8h1V6.5L9 10V2L4 5.5V2z" /></svg>
        </button>
        <button className="ctrlbtn ctrlbtn--play" onClick={onPlay} aria-label={isPlaying ? 'Pause' : 'Play'}>
          {isPlaying ? (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor"><rect x="3" y="2.5" width="2.5" height="9" /><rect x="8.5" y="2.5" width="2.5" height="9" /></svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor"><path d="M3 2l9 5-9 5z" /></svg>
          )}
        </button>
        <button className="ctrlbtn" onClick={() => onSeek(currentTime + 1 / 30)} aria-label="Next frame">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor"><path d="M9 2v8H8V6.5L3 10V2L8 5.5V2z" transform="scale(-1,1) translate(-12,0)" /></svg>
        </button>

        <div className="timecode">
          <span className="timecode__current">{fmtTime(currentTime)}</span>
          <span className="timecode__sep">/</span>
          <span className="timecode__total">{fmtTime(durationS)}</span>
        </div>

        <div className="speed">
          {[0.25, 0.5, 1, 2].map((s) => (
            <button key={s} className={`speed__btn ${speed === s ? 'speed__btn--on' : ''}`} onClick={() => onSpeed(s)}>
              {s}×
            </button>
          ))}
        </div>

        <div className="timeline__spacer" />
      </div>

      <div className="timeline__tracks">
        <div className="timeline__ruler">
          {Array.from({ length: Math.ceil(durationS / tickSpacing) + 1 }).map((_, i) => {
            const t = i * tickSpacing
            if (t > durationS) return null
            return (
              <div key={i} className="timeline__sec timeline__sec--major" style={{ left: `${(t / durationS) * 100}%` }}>
                <span>{t}s</span>
              </div>
            )
          })}
        </div>

        <div
          ref={trackRef}
          className="timeline__surface"
          onMouseDown={(e) => {
            // ignore clicks on the label column
            const r = (e.currentTarget as HTMLDivElement).getBoundingClientRect()
            if (e.clientX < r.left + LABEL_W) return
            dragging.current = true
            seekFromX(e.clientX)
          }}
          onMouseMove={(e) => {
            const r = (e.currentTarget as HTMLDivElement).getBoundingClientRect()
            if (e.clientX < r.left + LABEL_W) {
              onScrubHover(null)
              return
            }
            const t = ((e.clientX - r.left - LABEL_W) / (r.width - LABEL_W)) * durationS
            onScrubHover(Math.max(0, Math.min(durationS, t)))
          }}
          onMouseLeave={() => onScrubHover(null)}
        >
          {/* Action segments — split into hand lanes */}
          {artifacts.actions && artifacts.actions.length > 0 ? (
            <>
              <Lane label="action · R" color="#5edcc6">
                {artifacts.actions.filter((s) => s.hand === 'right').map((seg, i) => (
                  <ActionSegBlock key={`r-${i}`} seg={seg} durationS={durationS} />
                ))}
              </Lane>
              <Lane label="action · L" color="#f0a35e">
                {artifacts.actions.filter((s) => s.hand === 'left').map((seg, i) => (
                  <ActionSegBlock key={`l-${i}`} seg={seg} durationS={durationS} />
                ))}
              </Lane>
            </>
          ) : (
            <Lane label="action" color="#c2e02b" muted>
              <span className="lane__empty">no action segments</span>
            </Lane>
          )}

          {artifacts.transcript && artifacts.transcript.length > 0 ? (
            <Lane label="transcript" color="#e6b800">
              {artifacts.transcript.map((seg, i) => (
                <TranscriptSegBlock key={i} seg={seg} durationS={durationS} />
              ))}
            </Lane>
          ) : (
            <Lane label="transcript" color="#e6b800" muted>
              <span className="lane__empty">no transcript</span>
            </Lane>
          )}

          {detectionRibbons && detectionRibbons.length > 0 ? (
            <Lane label="detections" color="#c2e02b" tall>
              {detectionRibbons.map((ribbon, ri) => (
                <div key={ribbon.className} className="ribbonrow" style={{ top: `${4 + ri * 9}px` }}>
                  <span className="ribbonrow__label" style={{ color: ribbon.color }}>
                    {ribbon.className}
                  </span>
                  {ribbon.segments.map((seg, si) => (
                    <div
                      key={si}
                      className="ribbonseg"
                      style={{
                        left: `${(seg.start / durationS) * 100}%`,
                        width: `${Math.max(0.15, ((seg.end - seg.start) / durationS) * 100)}%`,
                        background: ribbon.color,
                      }}
                      title={`${ribbon.className} · ${seg.start.toFixed(2)}–${seg.end.toFixed(2)}s`}
                    />
                  ))}
                </div>
              ))}
            </Lane>
          ) : detectionMarkers.length > 0 && (
            <Lane label="detections" color="#c2e02b">
              {detectionMarkers.map((m, i) => (
                <span
                  key={i}
                  className="detmark"
                  style={{ left: `${(m.t / durationS) * 100}%` }}
                  title={`${m.label} @ ${m.t.toFixed(2)}s`}
                />
              ))}
            </Lane>
          )}

          {imuLanes ? (
            <Lane label="imu · |a|" color="#5b8def">
              <SignalLine series={imuLanes.ax} color="#5b8def" />
              <SignalLine series={imuLanes.ay} color="#d96b9d" />
              <SignalLine series={imuLanes.az} color="#5edcc6" />
            </Lane>
          ) : (
            <Lane label="imu · |a|" color="#5b8def" muted>
              <span className="lane__empty">no imu data</span>
            </Lane>
          )}

          {omegaLane && (
            <Lane label="ω · |rate|" color="#5edcc6">
              <SignalLine series={omegaLane} color="#5edcc6" />
            </Lane>
          )}

          <div
            className="playhead"
            style={{ left: `calc(${LABEL_W}px + (100% - ${LABEL_W}px) * ${pct / 100})` }}
          >
            <span className="playhead__bulb" />
          </div>
          {scrubHover !== null && scrubHover >= 0 && scrubHover <= durationS && (
            <div
              className="hoverhead"
              style={{ left: `calc(${LABEL_W}px + (100% - ${LABEL_W}px) * ${scrubHover / durationS})` }}
            >
              <span className="hoverhead__time">{fmtTime(scrubHover)}</span>
            </div>
          )}
        </div>
      </div>
    </footer>
  )
}

function TranscriptSegBlock({ seg, durationS }: { seg: TranscriptSeg; durationS: number }) {
  const left = (seg.start / durationS) * 100
  const width = Math.max(0.5, ((seg.end - seg.start) / durationS) * 100)
  return (
    <div
      className="actionseg"
      style={{ left: `${left}%`, width: `${width}%`, background: 'rgba(230,184,0,0.28)' }}
      title={`"${seg.text}"\n${seg.start.toFixed(2)}–${seg.end.toFixed(2)}s`}
    >
      <span>{seg.text}</span>
    </div>
  )
}

function ActionSegBlock({ seg, durationS }: { seg: ActionSeg; durationS: number }) {
  const left = (seg.start / durationS) * 100
  const width = ((seg.end - seg.start) / durationS) * 100
  const tone = seg.hand === 'right'
    ? 'rgba(94,220,198,0.32)'
    : seg.hand === 'left'
      ? 'rgba(240,163,94,0.32)'
      : 'rgba(255,255,255,0.12)'
  return (
    <div
      className={`actionseg actionseg--${seg.hand} ${!seg.meaningful ? 'actionseg--idle' : ''}`}
      style={{ left: `${left}%`, width: `${width}%`, background: tone }}
      title={`${seg.caption}\n${seg.reason}\n${(seg.confidence * 100).toFixed(0)}% conf`}
    >
      <span>{seg.caption}</span>
    </div>
  )
}

function Lane({ label, color, muted, tall, children }: {
  label: string
  color: string
  muted?: boolean
  tall?: boolean
  children: React.ReactNode
}) {
  return (
    <div className={`lane ${muted ? 'lane--muted' : ''} ${tall ? 'lane--tall' : ''}`}>
      <div className="lane__label" style={{ color }}>
        <span className="lane__dot" style={{ background: color }} />
        <span>{label}</span>
      </div>
      <div className="lane__body">{children}</div>
    </div>
  )
}

function SignalLine({ series, color }: { series: { t: number; v: number }[]; color: string }) {
  if (series.length === 0) return null
  const points = series.map((p) => `${p.t * 100},${25 - Math.max(-1, Math.min(1, p.v)) * 22}`).join(' ')
  return (
    <svg viewBox="0 0 100 50" className="signal" preserveAspectRatio="none">
      <polyline
        points={points}
        stroke={color}
        strokeWidth={1}
        fill="none"
        opacity={0.85}
        vectorEffect="non-scaling-stroke"
        strokeLinejoin="round"
      />
    </svg>
  )
}
