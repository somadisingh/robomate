import { NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'

async function requireLabUser() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return { error: NextResponse.json({ error: 'Unauthorized' }, { status: 401 }) }

  const { data: profile } = await supabase
    .from('profiles')
    .select('role')
    .eq('id', user.id)
    .single()

  if (profile?.role !== 'lab') {
    return { error: NextResponse.json({ error: 'Forbidden' }, { status: 403 }) }
  }

  return { supabase }
}

export async function GET() {
  const auth = await requireLabUser()
  if ('error' in auth && auth.error) return auth.error
  const { supabase } = auth

  // Fetch all submissions that have been indexed (have twelvelabs_video_id in metadata)
  const { data: submissions, error } = await supabase
    .from('submissions')
    .select('id, task_id, storage_path, metadata, created_at')
    .not('metadata->>twelvelabs_video_id', 'is', null)
    .order('created_at', { ascending: false })

  if (error) {
    return NextResponse.json({ error: 'Failed to fetch videos' }, { status: 500 })
  }

  if (!submissions || submissions.length === 0) {
    return NextResponse.json({ videos: [] })
  }

  // Fetch task names for all unique task IDs
  const taskIds = [...new Set(submissions.map(s => s.task_id).filter(Boolean))]
  const { data: tasks } = await supabase
    .from('tasks')
    .select('id, title')
    .in('id', taskIds)

  const taskNameById = new Map((tasks ?? []).map(t => [t.id, t.title]))

  // Generate signed URLs for each submission
  const videos = await Promise.all(
    submissions.map(async (sub) => {
      const meta = sub.metadata as Record<string, unknown> | null
      const twelvelabsVideoId = meta?.twelvelabs_video_id as string | undefined

      const { data: urlData } = await supabase.storage
        .from('recordings')
        .createSignedUrl(sub.storage_path.replace(/\/$/, '') + '/video.mp4', 3600)

      return {
        submissionId: sub.id,
        taskId: sub.task_id,
        taskTitle: taskNameById.get(sub.task_id) ?? null,
        twelvelabsVideoId: twelvelabsVideoId ?? null,
        signedUrl: urlData?.signedUrl ?? null,
        createdAt: sub.created_at,
      }
    })
  )

  return NextResponse.json({ videos })
}
