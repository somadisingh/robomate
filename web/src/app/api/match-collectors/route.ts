import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'
import {
  embedQuery,
  pineconeQuery,
  pineconeConfigured,
  NAMESPACE_COLLECTORS,
} from '@/lib/pinecone'

async function requireLabUser() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return { error: NextResponse.json({ matches: [] }) }

  const { data: profile } = await supabase
    .from('profiles')
    .select('role')
    .eq('id', user.id)
    .single()

  if (profile?.role !== 'lab') {
    return { error: NextResponse.json({ matches: [] }) }
  }
  return { supabase }
}

export async function POST(req: NextRequest) {
  try {
    const auth = await requireLabUser()
    if ('error' in auth && auth.error) return auth.error
    const { supabase } = auth

    if (!pineconeConfigured()) return NextResponse.json({ matches: [] })

    const { taskTitle, taskDescription, objects } = await req.json()
    const queryText = `Task: ${taskTitle}. ${taskDescription || ''}. Objects: ${(objects || []).join(', ')}.`
    const queryVector = await embedQuery(queryText)

    const rawMatches = await pineconeQuery({
      vector: queryVector,
      topK: 10,
      namespace: NAMESPACE_COLLECTORS,
    })

    // Enrich with display names (collectors metadata only carries the id).
    const collectorIds = rawMatches.map((m) => m.id)
    const nameMap: Record<string, string> = {}
    if (collectorIds.length > 0) {
      const { data: profiles } = await supabase
        .from('profiles')
        .select('id, display_name')
        .in('id', collectorIds)
      for (const p of profiles ?? []) {
        if (p.display_name) nameMap[p.id] = p.display_name
      }
    }

    const matches = rawMatches.map((m) => ({
      collector_id: m.id,
      display_name: nameMap[m.id] ?? null,
      match_score: m.score,
      total_approved: (m.metadata.total_approved as number) ?? 0,
      avg_score: (m.metadata.avg_score as number) ?? 0,
      object_specialties: (m.metadata.object_specialties as string[]) ?? [],
    }))

    return NextResponse.json({ matches })
  } catch (err) {
    console.error('[match-collectors]', err)
    return NextResponse.json({ matches: [] })
  }
}
