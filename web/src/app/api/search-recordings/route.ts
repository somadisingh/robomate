import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'
import {
  embedQuery,
  pineconeQuery,
  pineconeConfigured,
  NAMESPACE_RECORDINGS,
} from '@/lib/pinecone'

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
  return { supabase, user }
}

export async function POST(req: NextRequest) {
  const auth = await requireLabUser()
  if ('error' in auth && auth.error) return auth.error
  const { supabase, user } = auth

  if (!pineconeConfigured()) {
    return NextResponse.json({ error: 'Search not configured' }, { status: 500 })
  }

  let body: { query?: string }
  try {
    body = await req.json()
  } catch {
    return NextResponse.json({ error: 'Invalid JSON' }, { status: 400 })
  }

  const query = body.query?.trim()
  if (!query) {
    return NextResponse.json({ error: 'query required' }, { status: 400 })
  }

  try {
    // 1. Embed the search query.
    const queryVector = await embedQuery(query)

    // 2. Query Pinecone — labs only see their own recordings.
    const matches = await pineconeQuery({
      vector: queryVector,
      topK: 10,
      namespace: NAMESPACE_RECORDINGS,
      filter: { lab_id: { $eq: user.id } },
    })

    if (matches.length === 0) {
      return NextResponse.json({ results: [] })
    }

    // 3. Enrich with live Supabase data (status, score, timestamps).
    const recordingIds = matches.map((m) => m.id)
    const { data: recordings } = await supabase
      .from('recordings')
      .select('id, created_at, bounty_id, status, success, score')
      .in('id', recordingIds)

    const recordingMap = Object.fromEntries(
      (recordings ?? []).map((r) => [r.id, r])
    )

    const results = matches.map((match) => {
      const meta = match.metadata
      return {
        recording_id: match.id,
        score: match.score,
        task_title: (meta.task_title as string) ?? '',
        objects: (meta.objects as string[]) ?? [],
        detected_objects: (meta.detected_objects as string[]) ?? [],
        gemini_score: (meta.gemini_score as number) ?? 0,
        passed: Boolean(meta.passed),
        embedding_snippet: (meta.embedding_document as string) ?? '',
        supabase_data: recordingMap[match.id] ?? null,
      }
    })

    return NextResponse.json({ results })
  } catch (err) {
    console.error('[search-recordings]', err)
    return NextResponse.json({ error: 'Search failed' }, { status: 500 })
  }
}
