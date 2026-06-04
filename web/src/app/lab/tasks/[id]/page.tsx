import { createClient } from '@/lib/supabase/server'
import { notFound } from 'next/navigation'
import Link from 'next/link'
import SubmissionsLive, { type AnalysisJob, type RecordingAnalysis } from './submissions-live'
import EditableTaskHeader from './editable-task-header'

export default async function LabTaskPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  const supabase = await createClient()

  // Load task
  const { data: task, error } = await supabase
    .from('tasks')
    .select('*')
    .eq('id', id)
    .single()

  if (error || !task) notFound()

  // Load initial submissions
  const { data: rawSubmissions } = await supabase
    .from('submissions')
    .select('id, collector_id, storage_path, status, metadata, created_at')
    .eq('task_id', id)
    .order('created_at', { ascending: false })

  const submissionsList = rawSubmissions ?? []

  const recordingIds = submissionsList
    .map((s) => {
      const meta = s.metadata as Record<string, unknown> | null
      const recordingId = meta?.recording_id
      return typeof recordingId === 'string' ? recordingId : null
    })
    .filter((value): value is string => Boolean(value))

  const recordingById = new Map<string, RecordingAnalysis>()
  const jobsByRecordingId = new Map<string, AnalysisJob[]>()

  if (recordingIds.length > 0) {
    const [recordingsResult, jobsResult] = await Promise.all([
      supabase
        .from('recordings')
        .select(
          'id, status, is_scoring, summary, success, success_reasoning, score, score_reasoning, detected_objects, analysis_artifacts'
        )
        .in('id', recordingIds),
      supabase
        .from('recording_analysis_jobs')
        .select(
          'recording_id, kind, status, artifact_path, summary, error, started_at, finished_at'
        )
        .in('recording_id', recordingIds),
    ])

    for (const recording of recordingsResult.data ?? []) {
      recordingById.set(recording.id, recording as RecordingAnalysis)
    }
    for (const job of jobsResult.data ?? []) {
      const typed = job as AnalysisJob
      const list = jobsByRecordingId.get(typed.recording_id) ?? []
      list.push(typed)
      jobsByRecordingId.set(typed.recording_id, list)
    }
  }

  // Generate signed URLs server-side for initial load
  const submissions = await Promise.all(
    submissionsList.map(async (s) => {
      const { data } = await supabase.storage
        .from('recordings')
        .createSignedUrl(s.storage_path.replace(/\/$/, '') + '/video.mp4', 3600)
      const meta = s.metadata as Record<string, unknown> | null
      const recordingId = typeof meta?.recording_id === 'string' ? meta.recording_id : null
      return {
        ...s,
        signedUrl: data?.signedUrl ?? null,
        recording: recordingId ? recordingById.get(recordingId) ?? null : null,
        analysisJobs: recordingId ? jobsByRecordingId.get(recordingId) ?? [] : [],
      }
    })
  )

  const progress = Math.min(100, (task.quantity_filled / task.quantity_needed) * 100)
  const totalPaid = task.quantity_filled * task.bounty_amount

  return (
    <div>
      {/* Back */}
      <Link href="/lab/dashboard" className="mb-6 inline-block text-sm text-[var(--foreground-secondary)] transition-colors hover:text-white">
        ← Dashboard
      </Link>

      {/* Task header */}
      <div className="surface-panel mb-6 p-6">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <EditableTaskHeader
              taskId={id}
              initialTitle={task.title}
              initialDescription={task.description ?? null}
              status={task.status}
            />
            <div className="flex flex-wrap gap-2 mt-3">
              <span className="rounded-full border border-[var(--border)] bg-[var(--surface-muted)] px-2 py-0.5 text-xs text-[var(--foreground-secondary)]">
                {task.data_type}
              </span>
              {((task.required_capabilities as string[] | null) ?? []).map((cap: string) => (
                <span key={cap} className="rounded-full border border-[var(--border)] bg-[rgba(59,91,219,0.12)] px-2 py-0.5 text-xs text-[#aebeff]">
                  {cap}
                </span>
              ))}
            </div>
          </div>
          <div className="text-right shrink-0">
            <div className="text-2xl font-bold text-[#aebeff]">${task.bounty_amount}</div>
            <div className="text-xs text-[var(--foreground-secondary)]">per submission</div>
          </div>
        </div>

        {/* Progress */}
        <div className="mt-5">
          <div className="flex justify-between text-sm mb-1.5">
            <span className="text-[var(--foreground-secondary)]">
              {task.quantity_filled} / {task.quantity_needed} collected
            </span>
            <span className="text-[var(--foreground-secondary)]">${totalPaid.toFixed(2)} paid out</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-[rgba(255,255,255,0.08)]">
            <div
              className="h-full rounded-full bg-[#3b5bdb] transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {task.deadline && (
          <p className="mt-3 text-xs text-[var(--foreground-secondary)]">
            Deadline: {new Date(task.deadline).toLocaleDateString('en-US', {
              month: 'long', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit'
            })}
          </p>
        )}
      </div>

      {/* Live submissions — client component owns Realtime */}
      <div className="flex items-center gap-2 mb-4">
        <span className="h-2 w-2 rounded-full bg-[#2f9e44] animate-pulse" />
        <span className="text-xs text-[var(--foreground-secondary)]">Live — updates automatically when collectors upload</span>
      </div>

      <SubmissionsLive taskId={id} initialSubmissions={submissions} />
    </div>
  )
}
