import { createClient } from '@supabase/supabase-js'
import {
  embedDocument,
  pineconeUpsert,
  pineconeConfigured,
  NAMESPACE_COLLECTORS,
} from '@/lib/pinecone'

type HistoryEntry = {
  task_title: string
  task_description: string
  objects: string[]
  score: number
  passed: boolean
}

function buildCollectorDocument(history: HistoryEntry[]): string {
  if (history.length === 0) return 'Collector with no completed submissions.'

  const passing = history.filter((h) => h.passed)
  if (passing.length === 0) {
    return `Collector with ${history.length} submissions, none passing yet.`
  }

  const avgScore =
    passing.reduce((sum, h) => sum + (h.score || 0), 0) / passing.length
  const allObjects = [...new Set(passing.flatMap((h) => h.objects))]
  const allTasks = [...new Set(passing.map((h) => h.task_title).filter(Boolean))]
  const recentDescriptions = passing
    .slice(-5)
    .map((h) => (h.task_description || '').slice(0, 80))
    .join(' | ')

  return (
    `Experienced with tasks: ${allTasks.slice(0, 10).join(', ')}. ` +
    `Objects handled: ${allObjects.slice(0, 15).join(', ')}. ` +
    `Recent task descriptions: ${recentDescriptions}. ` +
    `Average quality score: ${avgScore.toFixed(1)}/10. ` +
    `Total approved submissions: ${passing.length}.`
  )
}

/**
 * Rebuilds a collector's Pinecone profile vector ("collectors" namespace) from
 * their full approved-submission history. Uses a service-role client so the
 * profile reflects work across all labs (RLS would otherwise scope a lab to its
 * own tasks). Fire-and-forget: never throws into the caller.
 */
export async function updateCollectorProfile(collectorId: string): Promise<void> {
  try {
    const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY
    const url = process.env.NEXT_PUBLIC_SUPABASE_URL
    if (!serviceKey || !url || !pineconeConfigured()) return

    const admin = createClient(url, serviceKey, {
      auth: { persistSession: false },
    })

    const { data: subs } = await admin
      .from('submissions')
      .select('id, metadata, tasks(title, description, objects)')
      .eq('collector_id', collectorId)
      .eq('status', 'approved')

    if (!subs || subs.length === 0) return

    // Map each approved submission to its recording's score/success.
    const recordingIds = subs
      .map((s) => {
        const meta = (s.metadata ?? {}) as Record<string, unknown>
        const rid = meta.recording_id
        return typeof rid === 'string' ? rid : null
      })
      .filter((v): v is string => Boolean(v))

    const recMap: Record<string, { score: number | null; success: boolean | null }> = {}
    if (recordingIds.length > 0) {
      const { data: recs } = await admin
        .from('recordings')
        .select('id, score, success')
        .in('id', recordingIds)
      for (const r of recs ?? []) {
        recMap[r.id] = { score: r.score, success: r.success }
      }
    }

    const history: HistoryEntry[] = subs.map((s) => {
      // tasks is an embedded resource (object, or array depending on the join).
      const task = (Array.isArray(s.tasks) ? s.tasks[0] : s.tasks) as
        | { title?: string; description?: string; objects?: string[] }
        | null
      const meta = (s.metadata ?? {}) as Record<string, unknown>
      const rid = typeof meta.recording_id === 'string' ? meta.recording_id : ''
      const rec = recMap[rid]
      return {
        task_title: task?.title ?? '',
        task_description: task?.description ?? '',
        objects: task?.objects ?? [],
        score: rec?.score ?? 0,
        passed: Boolean(rec?.success),
      }
    })

    const passing = history.filter((h) => h.passed)
    const document = buildCollectorDocument(history)
    const vector = await embedDocument(document)

    const objectSpecialties = [
      ...new Set(history.flatMap((h) => h.objects)),
    ].slice(0, 20)
    const avgScore = passing.length
      ? passing.reduce((sum, h) => sum + (h.score || 0), 0) / passing.length
      : 0

    await pineconeUpsert({
      vectors: [
        {
          id: collectorId,
          values: vector,
          metadata: {
            collector_id: collectorId,
            total_approved: passing.length,
            avg_score: avgScore,
            object_specialties: objectSpecialties,
            embedding_document: document.slice(0, 800),
          },
        },
      ],
      namespace: NAMESPACE_COLLECTORS,
    })
  } catch (e) {
    console.error('[collector-profile]', e)
  }
}
