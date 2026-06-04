import { createClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'

const CAPABILITY_QUESTIONS = [
  {
    question: 'What environments can you record in?',
    options: [
      { value: 'outdoor', label: 'Outdoors (streets, parks, nature)' },
      { value: 'indoor', label: 'Indoors (home, office, public spaces)' },
    ],
  },
  {
    question: 'What can you capture?',
    options: [
      { value: 'video', label: 'Video (I have a working camera)' },
      { value: 'audio', label: 'Audio (I have a good microphone)' },
      { value: 'motion', label: 'Motion / movement (I can walk, run, exercise on camera)' },
    ],
  },
]

async function submitQuestionnaire(formData: FormData) {
  'use server'
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) redirect('/login')

  const capabilities = formData.getAll('capabilities') as string[]
  const locationCity = formData.get('location_city') as string

  // Derive capabilities from answers — both arrays are capability values
  await supabase.from('collector_profiles').upsert({
    user_id: user.id,
    capabilities,
    location_city: locationCity,
    questionnaire_data: { capabilities, location_city: locationCity },
  })

  redirect('/collector/tasks')
}

export default async function OnboardPage() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()

  // Skip onboarding if already completed
  const { data: existing } = await supabase
    .from('collector_profiles')
    .select('user_id')
    .eq('user_id', user!.id)
    .single()

  if (existing) redirect('/collector/tasks')

  return (
    <div className="max-w-lg">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">Tell us about yourself</h1>
        <p className="mt-1 text-sm text-[var(--foreground-secondary)]">
          We use this to match you with the right data collection tasks.
        </p>
      </div>

      <form action={submitQuestionnaire} className="space-y-8">
        <div>
          <label className="mb-1 block text-sm font-medium text-white">Your city</label>
          <input
            name="location_city"
            type="text"
            placeholder="e.g. San Francisco"
            className="input-dark text-sm"
          />
        </div>

        {CAPABILITY_QUESTIONS.map((q, i) => (
          <div key={i}>
            <p className="mb-3 text-sm font-medium text-white">{q.question}</p>
            <div className="space-y-2">
              {q.options.map(opt => (
                <label key={opt.value} className="surface-muted flex cursor-pointer items-start gap-3 p-3">
                  <input
                    type="checkbox"
                    name="capabilities"
                    value={opt.value}
                    className="mt-0.5 rounded border-white/20 bg-transparent"
                  />
                  <span className="text-sm text-[var(--foreground-secondary)]">{opt.label}</span>
                </label>
              ))}
            </div>
          </div>
        ))}

        <button
          type="submit"
          className="btn-collector w-full rounded-lg py-2.5 text-sm font-medium transition-colors"
        >
          Start earning →
        </button>
      </form>
    </div>
  )
}
