import { createClient } from '@/lib/supabase/server'
import { notFound, redirect } from 'next/navigation'
import Studio, { type StudioData, type SiblingRecording, type AnalysisJobPayload, type SplatScenePayload } from './studio'

type AnalysisJobKind =
  | 'gemini_eval'
  | 'mediapipe_hands'
  | 'yolo_objects'
  | 'sam_segments'
  | 'temporal_actions'
  | 'gaussian_splat'

type RecordingRow = {
  id: string
  bounty_id: string | null
  collector_id: string | null
  storage_path: string
  status: 'uploaded' | 'analyzing' | 'analyzed' | 'analysis_failed'
  is_scoring: boolean
  summary: string | null
  success: boolean | null
  success_reasoning: string | null
  score: number | null
  score_reasoning: string | null
  detected_objects: unknown
  analysis_artifacts: unknown
  created_at: string
  device_model: string | null
  duration_ms: number | null
  size_bytes: number | null
  gps_lat: number | null
  gps_lon: number | null
  gps_accuracy_m: number | null
  streams: string[] | null
  depth_width: number | null
  depth_height: number | null
  depth_frame_count: number | null
}

const STREAM_NAMES = ['video.mp4', 'imu.jsonl', 'poses.jsonl', 'intrinsics.json', 'depth.bin', 'transcript.json'] as const

export default async function StudioPage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params
  const supabase = await createClient()

  // Auth required — needed for storage signed URLs and RLS-scoped reads.
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) redirect(`/login?next=/studio/${id}`)

  const { data: recording, error } = await supabase
    .from('recordings')
    .select(
      'id, bounty_id, collector_id, storage_path, status, is_scoring, summary, success, success_reasoning, score, score_reasoning, detected_objects, analysis_artifacts, created_at, device_model, duration_ms, size_bytes, gps_lat, gps_lon, gps_accuracy_m, streams, depth_width, depth_height, depth_frame_count'
    )
    .eq('id', id)
    .single<RecordingRow>()

  if (error || !recording) notFound()

  const storagePath = recording.storage_path.replace(/\/$/, '')

  // Parallel fetch task, jobs, siblings, collector profile, and signed URLs
  const [
    taskResult,
    jobsResult,
    siblingsResult,
    collectorResult,
    streamUrlEntries,
  ] = await Promise.all([
    recording.bounty_id
      ? supabase
          .from('tasks')
          .select('id, title, description, objects, required_capabilities, bounty_amount, data_type')
          .eq('id', recording.bounty_id)
          .maybeSingle()
      : Promise.resolve({ data: null }),

    supabase
      .from('recording_analysis_jobs')
      .select('kind, status, artifact_path, summary, error, started_at, finished_at')
      .eq('recording_id', id),

    recording.bounty_id
      ? supabase
          .from('recordings')
          .select('id, score, success, status, created_at, duration_ms')
          .eq('bounty_id', recording.bounty_id)
          .order('created_at', { ascending: true })
      : Promise.resolve({ data: [] }),

    recording.collector_id
      ? supabase
          .from('profiles')
          .select('display_name')
          .eq('id', recording.collector_id)
          .maybeSingle()
      : Promise.resolve({ data: null }),

    Promise.all(
      STREAM_NAMES.map(async (name) => {
        if (!recording.streams?.includes(name)) return [name, null] as const
        const { data } = await supabase.storage
          .from('recordings')
          .createSignedUrl(`${storagePath}/${name}`, 3600)
        return [name, data?.signedUrl ?? null] as const
      })
    ),
  ])

  const task = (taskResult.data as {
    id: string
    title: string | null
    description: string | null
    objects: string[] | null
    required_capabilities: string[] | null
    bounty_amount: number | null
    data_type: string | null
  } | null) ?? null

  const jobs = (jobsResult.data ?? []) as Array<{
    kind: AnalysisJobKind
    status: 'pending' | 'running' | 'succeeded' | 'failed'
    artifact_path: string | null
    summary: Record<string, unknown> | null
    error: string | null
    started_at: string | null
    finished_at: string | null
  }>

  // Signed URLs per analysis job artifact
  const jobsWithUrls: AnalysisJobPayload[] = await Promise.all(
    jobs.map(async (job) => {
      let signedUrl: string | null = null
      if (job.artifact_path) {
        const { data } = await supabase.storage
          .from('recordings')
          .createSignedUrl(job.artifact_path, 3600)
        signedUrl = data?.signedUrl ?? null
      }
      return { ...job, signedUrl }
    })
  )

  // Resolve the splat scene if its job succeeded — server-side fetch manifest,
  // then mint signed URLs for the sibling artifacts.
  let splatScene: SplatScenePayload | null = null
  const splatJob = jobsWithUrls.find((j) => j.kind === 'gaussian_splat' && j.status === 'succeeded')
  if (splatJob && splatJob.signedUrl && splatJob.artifact_path) {
    try {
      const manifestRes = await fetch(splatJob.signedUrl, { cache: 'no-store' })
      if (manifestRes.ok) {
        const manifest = (await manifestRes.json()) as {
          version?: number
          splat: { path: string; size_bytes: number; num_gaussians: number }
          camera_path: { path: string; frame_count: number; fps: number }
          seed_points?: { path: string }
        }
        const manifestDir = splatJob.artifact_path.replace(/\/[^/]+$/, '')
        const sign = async (rel: string) => {
          const { data } = await supabase.storage
            .from('recordings')
            .createSignedUrl(`${manifestDir}/${rel}`, 3600)
          return data?.signedUrl ?? null
        }
        const [splatUrl, cameraPathUrl, seedPointsUrl] = await Promise.all([
          sign(manifest.splat.path),
          sign(manifest.camera_path.path),
          manifest.seed_points ? sign(manifest.seed_points.path) : Promise.resolve(null),
        ])
        if (splatUrl && cameraPathUrl) {
          splatScene = {
            splatUrl,
            cameraPathUrl,
            seedPointsUrl,
            numGaussians: manifest.splat.num_gaussians,
            frameCount: manifest.camera_path.frame_count,
            fps: manifest.camera_path.fps,
          }
        }
      }
    } catch {
      // Best-effort — fall through with splatScene = null.
    }
  }

  const siblings = ((siblingsResult.data ?? []) as Array<{
    id: string
    score: number | null
    success: boolean | null
    status: string
    created_at: string
    duration_ms: number | null
  }>).map<SiblingRecording>((s, idx) => ({
    id: s.id,
    takeNumber: idx + 1,
    score: s.score,
    success: s.success,
    status: s.status as SiblingRecording['status'],
    createdAt: s.created_at,
    durationMs: s.duration_ms ?? null,
    isCurrent: s.id === id,
  }))

  const streamUrls = Object.fromEntries(streamUrlEntries) as Record<
    (typeof STREAM_NAMES)[number],
    string | null
  >

  const collectorName = (collectorResult.data as { display_name: string | null } | null)?.display_name ?? null

  const detectedObjects = Array.isArray(recording.detected_objects)
    ? (recording.detected_objects as Array<{
        class_name?: string
        count?: number
        max_confidence?: number
        representative_frame?: number
      }>).filter((o) => o && typeof o === 'object')
    : []

  const studioData: StudioData = {
    recording: {
      id: recording.id,
      status: recording.status,
      isScoring: recording.is_scoring,
      summary: recording.summary,
      success: recording.success,
      successReasoning: recording.success_reasoning,
      score: recording.score,
      scoreReasoning: recording.score_reasoning,
      detectedObjects,
      createdAt: recording.created_at,
      deviceModel: recording.device_model,
      durationMs: recording.duration_ms,
      sizeBytes: recording.size_bytes,
      gpsLat: recording.gps_lat,
      gpsLon: recording.gps_lon,
      gpsAccuracyM: recording.gps_accuracy_m,
      streams: recording.streams ?? [],
      collectorId: recording.collector_id,
      collectorName,
      depthWidth: recording.depth_width,
      depthHeight: recording.depth_height,
      depthFrameCount: recording.depth_frame_count,
    },
    task: task && {
      id: task.id,
      title: task.title,
      description: task.description,
      objects: task.objects ?? [],
      requiredCapabilities: task.required_capabilities ?? [],
      bountyAmount: task.bounty_amount,
      dataType: task.data_type,
    },
    siblings,
    jobs: jobsWithUrls,
    streamUrls,
    splatScene,
  }

  return <Studio data={studioData} />
}
