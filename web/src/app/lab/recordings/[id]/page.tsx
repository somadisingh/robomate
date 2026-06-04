import { createClient } from '@/lib/supabase/server'
import { notFound } from 'next/navigation'
import Link from 'next/link'
import ArtifactPreview from './artifact-preview'

type AnalysisJobKind =
  | 'gemini_eval'
  | 'mediapipe_hands'
  | 'yolo_objects'
  | 'sam_segments'
  | 'temporal_actions'

type AnalysisJobStatus = 'pending' | 'running' | 'succeeded' | 'failed'

type RecordingStatus = 'uploaded' | 'analyzing' | 'analyzed' | 'analysis_failed'

type AnalysisJobRow = {
  kind: AnalysisJobKind
  status: AnalysisJobStatus
  artifact_path: string | null
  summary: Record<string, unknown> | null
  error: string | null
  started_at: string | null
  finished_at: string | null
}

type DetectedObject = {
  class_name?: string
  count?: number
  max_confidence?: number
  representative_frame?: number | string
}

const JOB_KIND_LABEL: Record<AnalysisJobKind, string> = {
  gemini_eval: 'Gemini scoring',
  mediapipe_hands: 'MediaPipe hands',
  yolo_objects: 'YOLO objects',
  sam_segments: 'SAM segments',
  temporal_actions: 'Temporal actions',
}

const JOB_KIND_ORDER: AnalysisJobKind[] = [
  'gemini_eval',
  'mediapipe_hands',
  'yolo_objects',
  'sam_segments',
  'temporal_actions',
]

const JOB_STATUS_STYLES: Record<AnalysisJobStatus, string> = {
  pending: 'border border-[var(--border)] bg-[var(--surface-muted)] text-[var(--foreground-secondary)]',
  running: 'bg-[rgba(216,163,71,0.16)] text-[#f0cb7c]',
  succeeded: 'bg-[rgba(47,158,68,0.16)] text-[#99ddaa]',
  failed: 'bg-[rgba(210,100,100,0.16)] text-[#f3a8a8]',
}

const RECORDING_STATUS_STYLES: Record<RecordingStatus, string> = {
  uploaded: 'bg-[rgba(115,120,131,0.18)] text-[#d3d7de]',
  analyzing: 'bg-[rgba(216,163,71,0.16)] text-[#f0cb7c]',
  analyzed: 'bg-[rgba(47,158,68,0.16)] text-[#99ddaa]',
  analysis_failed: 'bg-[rgba(210,100,100,0.16)] text-[#f3a8a8]',
}

function formatTimestamp(value: string | null): string {
  if (!value) return '—'
  try {
    return new Date(value).toLocaleString()
  } catch {
    return value
  }
}

function detectedObjectsArray(value: unknown): DetectedObject[] {
  if (!Array.isArray(value)) return []
  return value.filter((item): item is DetectedObject =>
    typeof item === 'object' && item !== null
  )
}

export default async function RecordingPage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params
  const supabase = await createClient()

  const { data: recording, error: recordingError } = await supabase
    .from('recordings')
    .select(
      'id, bounty_id, storage_path, status, is_scoring, summary, success, success_reasoning, score, score_reasoning, detected_objects, analysis_artifacts, created_at'
    )
    .eq('id', id)
    .single()

  if (recordingError || !recording) notFound()

  const taskPromise = recording.bounty_id
    ? supabase
        .from('tasks')
        .select('id, title, description, objects')
        .eq('id', recording.bounty_id)
        .single()
    : Promise.resolve({ data: null, error: null })

  const jobsPromise = supabase
    .from('recording_analysis_jobs')
    .select('kind, status, artifact_path, summary, error, started_at, finished_at')
    .eq('recording_id', id)
    .order('kind', { ascending: true })

  const videoSignedUrlPromise = supabase.storage
    .from('recordings')
    .createSignedUrl(
      String(recording.storage_path).replace(/\/$/, '') + '/video.mp4',
      3600
    )

  const [taskResult, jobsResult, videoSignedUrlResult] = await Promise.all([
    taskPromise,
    jobsPromise,
    videoSignedUrlPromise,
  ])

  const task = taskResult.data as
    | { id: string; title: string | null; description: string | null; objects: string[] | null }
    | null

  const jobs = (jobsResult.data ?? []) as AnalysisJobRow[]

  const videoSignedUrl = videoSignedUrlResult.data?.signedUrl ?? null

  const artifactSignedUrls = await Promise.all(
    jobs.map(async (job) => {
      if (!job.artifact_path) return [job.kind, null] as const
      const { data } = await supabase.storage
        .from('recordings')
        .createSignedUrl(job.artifact_path, 3600)
      return [job.kind, data?.signedUrl ?? null] as const
    })
  )

  const artifactSignedUrlByKind = new Map<AnalysisJobKind, string | null>(artifactSignedUrls)

  const jobByKind = new Map<AnalysisJobKind, AnalysisJobRow>()
  for (const job of jobs) jobByKind.set(job.kind, job)

  const extraKinds = jobs
    .map((job) => job.kind)
    .filter((kind) => !JOB_KIND_ORDER.includes(kind))
  const orderedKinds: AnalysisJobKind[] = [...JOB_KIND_ORDER, ...extraKinds]

  const detectedObjects = detectedObjectsArray(recording.detected_objects)
  const recordingStatus = recording.status as RecordingStatus

  return (
    <div className="space-y-6">
      {task && (
        <Link
          href={`/lab/tasks/${recording.bounty_id}`}
          className="inline-block text-sm text-[var(--foreground-secondary)] transition-colors hover:text-white"
        >
          ← Back to task
        </Link>
      )}

      <div className="surface-panel p-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0 flex-1">
            {task?.title && (
              <h1 className="text-xl font-bold text-white">{task.title}</h1>
            )}
            <p className="mt-1 truncate font-mono text-xs text-[var(--foreground-secondary)]">
              {recording.id}
            </p>
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${RECORDING_STATUS_STYLES[recordingStatus] ?? RECORDING_STATUS_STYLES.uploaded}`}>
                {recordingStatus}
              </span>
              {recording.is_scoring ? (
                <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-[rgba(216,163,71,0.16)] text-[#f0cb7c]">
                  Scoring
                </span>
              ) : recording.score !== null ? (
                <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-[rgba(47,158,68,0.16)] text-[#99ddaa]">
                  Score ready
                </span>
              ) : (
                <span className="rounded-full border border-[var(--border)] px-2 py-0.5 text-xs text-[var(--foreground-secondary)]">
                  Not scored
                </span>
              )}
              <span className="text-xs text-[var(--foreground-secondary)]">
                {formatTimestamp(recording.created_at as string | null)}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="surface-panel p-6">
        {recording.score === null ? (
          <p className="text-sm text-[var(--foreground-secondary)]">
            Awaiting Gemini score
          </p>
        ) : (
          <div className="grid gap-5 sm:grid-cols-[auto_minmax(0,1fr)]">
            <div className="flex flex-col items-start rounded-lg border border-[var(--border)] bg-[var(--surface-muted)] px-5 py-4">
              <span className="text-xs uppercase tracking-[0.14em] text-[var(--foreground-secondary)]">Score</span>
              <span className="text-4xl font-black tracking-[-0.04em] text-white">
                {recording.score}/10
              </span>
              {recording.success !== null && (
                <span
                  className={`mt-3 text-xs px-2 py-0.5 rounded-full font-medium ${
                    recording.success
                      ? 'bg-[rgba(47,158,68,0.16)] text-[#99ddaa]'
                      : 'bg-[rgba(210,100,100,0.16)] text-[#f3a8a8]'
                  }`}
                >
                  {recording.success ? 'Success' : 'Not yet'}
                </span>
              )}
            </div>
            <div className="space-y-3 text-sm">
              {recording.summary && (
                <p className="text-white">{recording.summary}</p>
              )}
              {recording.success_reasoning && (
                <div>
                  <p className="text-xs uppercase tracking-[0.14em] text-[var(--foreground-secondary)]">
                    Success reasoning
                  </p>
                  <p className="mt-1 text-[var(--foreground-secondary)]">{recording.success_reasoning}</p>
                </div>
              )}
              {recording.score_reasoning && (
                <div>
                  <p className="text-xs uppercase tracking-[0.14em] text-[var(--foreground-secondary)]">
                    Score reasoning
                  </p>
                  <p className="mt-1 text-[var(--foreground-secondary)]">{recording.score_reasoning}</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {detectedObjects.length > 0 && (
        <div className="surface-panel p-6">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-[0.14em] text-white">
            Detected objects
          </h2>
          <div className="flex flex-wrap gap-2">
            {detectedObjects.map((obj, index) => {
              const className = obj.class_name ?? 'unknown'
              const count = typeof obj.count === 'number' ? obj.count : null
              const conf = typeof obj.max_confidence === 'number'
                ? `${(obj.max_confidence * 100).toFixed(0)}%`
                : null
              const frame = obj.representative_frame
              return (
                <span
                  key={`${className}-${index}`}
                  title={frame !== undefined ? `Representative frame: ${frame}` : undefined}
                  className="rounded-full border border-[var(--border)] bg-[var(--surface-muted)] px-3 py-1 text-xs text-white"
                >
                  {className}
                  {count !== null && (
                    <span className="ml-1 text-[var(--foreground-secondary)]">×{count}</span>
                  )}
                  {conf && (
                    <span className="ml-2 text-[#aebeff]">{conf}</span>
                  )}
                </span>
              )
            })}
          </div>
        </div>
      )}

      <div className="surface-panel overflow-hidden">
        {videoSignedUrl ? (
          <video
            src={videoSignedUrl}
            controls
            preload="metadata"
            className="w-full max-h-[60vh] bg-black"
          />
        ) : (
          <div className="flex h-48 w-full items-center justify-center bg-[var(--surface-muted)]">
            <span className="text-sm text-[var(--foreground-secondary)]">Video preview unavailable</span>
          </div>
        )}
      </div>

      <div className="surface-panel p-6">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-[0.14em] text-white">
          Analyzer jobs
        </h2>
        {orderedKinds.length === 0 ? (
          <p className="text-sm text-[var(--foreground-secondary)]">No analyzer jobs recorded yet.</p>
        ) : (
          <ul className="space-y-3">
            {orderedKinds.map((kind) => {
              const job = jobByKind.get(kind)
              const signedUrl = artifactSignedUrlByKind.get(kind) ?? null
              const label = JOB_KIND_LABEL[kind] ?? kind

              if (!job) {
                return (
                  <li
                    key={kind}
                    className="rounded-lg border border-[var(--border)] bg-[var(--surface-muted)] p-4"
                  >
                    <div className="flex flex-wrap items-center gap-3">
                      <div className="min-w-0 flex-1">
                        <p className="font-medium text-white">{label}</p>
                      </div>
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${JOB_STATUS_STYLES.pending}`}>
                        not started
                      </span>
                    </div>
                    <p className="mt-2 text-xs text-[var(--foreground-secondary)]">
                      Job row not created yet
                    </p>
                  </li>
                )
              }

              return (
                <li
                  key={kind}
                  className="rounded-lg border border-[var(--border)] bg-[var(--surface-muted)] p-4"
                >
                  <div className="flex flex-wrap items-center gap-3">
                    <div className="min-w-0 flex-1">
                      <p className="font-medium text-white">{label}</p>
                      <p className="mt-0.5 text-xs text-[var(--foreground-secondary)]">
                        Started {formatTimestamp(job.started_at)} · Finished {formatTimestamp(job.finished_at)}
                      </p>
                    </div>
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${JOB_STATUS_STYLES[job.status]}`}>
                      {job.status}
                    </span>
                    {signedUrl && (
                      <a
                        href={signedUrl}
                        target="_blank"
                        rel="noreferrer"
                        className="text-xs font-medium text-[#aebeff] transition-colors hover:text-white"
                      >
                        Download JSON ↗
                      </a>
                    )}
                  </div>
                  {job.status === 'failed' && job.error && (
                    <p className="mt-2 line-clamp-3 text-xs text-[#f3a8a8]">{job.error}</p>
                  )}
                  {signedUrl && <ArtifactPreview signedUrl={signedUrl} kind={kind} />}
                </li>
              )
            })}
          </ul>
        )}
      </div>

      {task?.objects && task.objects.length > 0 && (
        <div className="surface-panel p-6">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-[0.14em] text-white">
            Task object tags
          </h2>
          <div className="flex flex-wrap gap-2">
            {task.objects.map((obj) => (
              <span
                key={obj}
                className="rounded-full border border-[var(--border)] bg-[var(--surface-muted)] px-3 py-1 text-xs text-[var(--foreground-secondary)]"
              >
                {obj}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
