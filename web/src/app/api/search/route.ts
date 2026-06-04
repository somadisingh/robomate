import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'

const TL_API = 'https://api.twelvelabs.io/v1.3'

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

interface TLClip {
  video_id: string
  start: number
  end: number
  rank: number
  score?: number
  confidence?: string
  thumbnail_url?: string
}

export async function GET(request: NextRequest) {
  const auth = await requireLabUser()
  if ('error' in auth && auth.error) return auth.error
  const { supabase } = auth

  const q = request.nextUrl.searchParams.get('q')?.trim()
  if (!q) return NextResponse.json({ error: 'Missing query' }, { status: 400 })

  const apiKey = process.env.TWELVELABS_API_KEY
  const indexId = process.env.TWELVELABS_INDEX_ID
  if (!apiKey || !indexId) {
    return NextResponse.json(
      { error: 'TwelveLabs not configured. Set TWELVELABS_API_KEY and TWELVELABS_INDEX_ID in .env.local.' },
      { status: 500 }
    )
  }

  try {
    const form = new FormData()
    form.append('index_id', indexId)
    form.append('query_text', q)
    form.append('search_options', 'visual')
    form.append('search_options', 'audio')
    form.append('page_limit', '12')

    const res = await fetch(`${TL_API}/search`, {
      method: 'POST',
      headers: { 'x-api-key': apiKey },
      body: form,
    })

    if (!res.ok) {
      const err = await res.text()
      console.error('TwelveLabs search error:', res.status, err)
      return NextResponse.json({ error: `Search API error (${res.status})` }, { status: 502 })
    }

    const body = await res.json()
    const clips: TLClip[] = body.data ?? []

    // Enrich with Supabase signed URLs by matching twelvelabs_video_id stored in metadata
    const enriched = await Promise.all(
      clips.map(async (clip) => {
        const { data: sub } = await supabase
          .from('submissions')
          .select('id, task_id, storage_path')
          .filter('metadata->>twelvelabs_video_id', 'eq', clip.video_id)
          .maybeSingle()

        if (!sub) return { ...clip, submissionId: null, taskId: null, signedUrl: null }

        const { data: urlData } = await supabase.storage
          .from('recordings')
          .createSignedUrl(sub.storage_path.replace(/\/$/, '') + '/video.mp4', 3600)

        return {
          ...clip,
          submissionId: sub.id,
          taskId: sub.task_id,
          signedUrl: urlData?.signedUrl ?? null,
        }
      })
    )

    return NextResponse.json({ clips: enriched })
  } catch (err) {
    console.error('Search error:', err)
    return NextResponse.json({ error: 'Search failed' }, { status: 500 })
  }
}
