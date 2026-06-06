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
  if (!user) return { error: NextResponse.json({ matches: [] }) }

  const { data: profile } = await supabase
    .from('profiles')
    .select('role')
    .eq('id', user.id)
    .single()

  if (profile?.role !== 'lab') {
    return { error: NextResponse.json({ matches: [] }) }
  }
  return { user }
}

export async function POST(req: NextRequest) {
  // Fail silently with empty matches everywhere — this is advisory and must
  // never block task creation.
  try {
    const auth = await requireLabUser()
    if ('error' in auth && auth.error) return auth.error
    const { user } = auth

    if (!pineconeConfigured()) return NextResponse.json({ matches: [] })

    const { title, description, objects } = await req.json()
    if (!title) return NextResponse.json({ matches: [] })

    const queryText = `Task: ${title}. ${description || ''}. Objects: ${(objects || []).join(', ')}.`
    const queryVector = await embedQuery(queryText)

    const rawMatches = await pineconeQuery({
      vector: queryVector,
      topK: 5,
      namespace: NAMESPACE_RECORDINGS,
      filter: {
        lab_id: { $eq: user.id },
        passed: { $eq: true },
      },
    })

    // Only surface genuinely relevant matches.
    const matches = rawMatches
      .filter((m) => m.score >= 0.7)
      .map((m) => ({
        recording_id: m.id,
        score: m.score,
        task_title: (m.metadata.task_title as string) ?? '',
        gemini_score: (m.metadata.gemini_score as number) ?? 0,
        objects: (m.metadata.objects as string[]) ?? [],
      }))

    return NextResponse.json({ matches })
  } catch (err) {
    console.error('[similar-recordings]', err)
    return NextResponse.json({ matches: [] })
  }
}
