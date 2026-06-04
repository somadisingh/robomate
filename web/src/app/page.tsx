'use client'

import Link from 'next/link'
import { useState, useEffect } from 'react'
import { createClient } from '@/lib/supabase/client'

function GabrielHornLogo() {
  return (
    <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" shapeRendering="geometricPrecision" className="w-8 h-8">
      <line x1="2" y1="24" x2="46" y2="24" stroke="currentColor" strokeWidth="0.35" strokeDasharray="1 1.6" opacity="0.38" />
      <ellipse cx="9" cy="24" rx="1.55" ry="10" stroke="currentColor" strokeWidth="0.75" opacity="0.55" />
      <ellipse cx="13" cy="24" rx="1.05" ry="6.2" stroke="currentColor" strokeWidth="0.65" opacity="0.42" />
      <ellipse cx="18" cy="24" rx="0.72" ry="4.1" stroke="currentColor" strokeWidth="0.6" opacity="0.32" />
      <ellipse cx="25" cy="24" rx="0.5" ry="2.8" stroke="currentColor" strokeWidth="0.55" opacity="0.24" />
      <ellipse cx="34" cy="24" rx="0.36" ry="2.05" stroke="currentColor" strokeWidth="0.5" opacity="0.18" />
      <path d="M 5.6 5.4 C 7.5 9, 11 22.2, 44.5 22.85" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M 5.6 42.6 C 7.5 39, 11 25.8, 44.5 25.15" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <ellipse cx="5.6" cy="24" rx="2.6" ry="18.6" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  )
}

const TICKER_EVENTS = [
  { task: 'Walking around · downtown', amt: '+$5.00', who: 'collector_8421', loc: 'SF, CA' },
  { task: 'Pick and place a cup', amt: '+$5.00', who: 'collector_2917', loc: 'Austin, TX' },
  { task: 'Outdoor traversal (60s)', amt: '+$12.50', who: 'collector_0413', loc: 'Berlin, DE' },
  { task: 'Kitchen counter, low light', amt: '+$8.00', who: 'collector_5552', loc: 'Tokyo, JP' },
  { task: 'Stairs descent, handheld', amt: '+$15.00', who: 'collector_1188', loc: 'Mexico City' },
  { task: 'Crowd density, sidewalk POV', amt: '+$6.25', who: 'collector_9077', loc: 'Lagos, NG' },
  { task: 'Door knob, varied lighting', amt: '+$4.96', who: 'collector_3304', loc: 'Seoul, KR' }
]

function Ticker() {
  const [index, setIndex] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => setIndex((p) => (p + 1) % TICKER_EVENTS.length), 2200)
    return () => clearInterval(interval)
  }, [])

  const event = TICKER_EVENTS[index]

  return (
    <div className="mt-12 flex items-center gap-6 flex-wrap">
      <span className="flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-[var(--foreground-tertiary)]">
        <span className="w-2 h-2 rounded-full bg-[#5fd78a] shadow-[0_0_0_0_rgba(95,215,138,0.55)] animate-pulse" />
        Live collections
      </span>
      <div className="flex-1 min-w-80 overflow-hidden relative h-5">
        <div key={index} className="flex items-center gap-7 font-mono text-sm text-[var(--foreground-secondary)] whitespace-nowrap animate-fadeIn">
          <span className="text-[#99ddaa] font-semibold">{event.amt}</span>
          <span className="text-[var(--foreground-tertiary)]">/</span>
          <span>{event.task}</span>
          <span className="text-[var(--foreground-tertiary)]">·</span>
          <span>{event.who}</span>
          <span className="text-[var(--foreground-tertiary)]">·</span>
          <span>{event.loc}</span>
        </div>
      </div>
    </div>
  )
}

function NavAuth() {
  const [user, setUser] = useState<{ email?: string; role?: string } | null | undefined>(undefined)

  useEffect(() => {
    const supabase = createClient()
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) { setUser(null); return }
      const role = session.user.user_metadata?.role as string | undefined
      setUser({ email: session.user.email, role })
    })
  }, [])

  // Still loading — render nothing to avoid flash
  if (user === undefined) return null

  if (!user) {
    return (
      <div className="ml-auto flex items-center gap-5 text-sm text-[var(--foreground-secondary)]">
        <Link href="/login" className="transition-colors hover:text-white">Sign in</Link>
        <Link href="/signup" className="btn-lab rounded-lg px-4 py-2 text-sm font-medium transition-colors hover:bg-[#4b6af0]">Get started</Link>
      </div>
    )
  }

  const dashHref = user.role === 'collector' ? '/collector/tasks' : '/lab/dashboard'

  async function handleSignOut() {
    const supabase = createClient()
    await supabase.auth.signOut()
    window.location.href = '/login'
  }

  return (
    <>
      <nav className="hidden md:flex items-center gap-7 text-sm text-[var(--foreground-secondary)]">
        <Link href={dashHref} className="transition-colors hover:text-white">Dashboard</Link>
        {user.role === 'lab' && (
          <Link href="/lab/tasks/new" className="transition-colors hover:text-white">New Task</Link>
        )}
        {user.role === 'lab' && (
          <Link href="/lab/search" className="transition-colors hover:text-white">Search</Link>
        )}
      </nav>
      <div className="ml-auto flex items-center gap-5 text-sm text-[var(--foreground-secondary)]">
        <span>{user.email}</span>
        <button onClick={handleSignOut} className="transition-colors hover:text-white">Sign out</button>
      </div>
    </>
  )
}

export default function Home() {
  return (
    <div className="min-h-screen bg-[var(--background)] text-white flex flex-col">
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeIn { animation: fadeIn 360ms cubic-bezier(0.2, 0.7, 0.2, 1); }
        @keyframes pulse {
          0% { box-shadow: 0 0 0 0 rgba(95,215,138,0.55); }
          70% { box-shadow: 0 0 0 8px rgba(95,215,138,0); }
          100% { box-shadow: 0 0 0 0 rgba(95,215,138,0); }
        }
      `}</style>

      <header className="border-b border-[rgba(255,255,255,0.08)] bg-[rgba(15,15,15,0.85)] backdrop-blur-sm sticky top-0 z-30">
        <div className="mx-auto max-w-5xl px-8 py-5 flex items-center justify-between gap-6">
          <Link href="/" className="flex items-center gap-3 transition-opacity hover:opacity-80">
            <GabrielHornLogo />
            <span className="text-base font-bold tracking-[0.01em] text-white">Robomate</span>
          </Link>

          <NavAuth />
        </div>
      </header>

      <main className="flex-1 mx-auto max-w-5xl px-8 py-16 w-full">
        <p className="text-xs font-semibold uppercase tracking-[0.22em] text-[var(--foreground-secondary)] mb-7">
          The physical data layer for AI
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1.45fr)_minmax(280px,0.55fr)] gap-20 items-start">
          <section>
            <h1 className="text-[clamp(56px,8.6vw,124px)] font-black leading-[0.92] tracking-[-0.045em] text-white">
              AI needs<br />
              <span className="italic font-normal text-[1.04em]" style={{ fontFamily: 'Georgia, serif', lineHeight: '0.95' }}>
                the real world.
              </span>
            </h1>

            <p className="mt-9 text-[19px] leading-[1.55] text-[var(--foreground-secondary)] max-w-lg font-medium">
              Models can&apos;t touch reality. You can. Capture the world, get paid.
            </p>

            <div className="mt-11 flex flex-col sm:flex-row gap-3 flex-wrap">
              <Link href="/signup" className="btn-lab rounded-[10px] px-5.5 py-3.5 text-[15px] font-semibold inline-flex items-center gap-2 transition-colors hover:bg-[#4b6af0]">
                Post a task <span>→</span>
              </Link>
              <Link href="/signup" className="btn-collector rounded-[10px] px-5.5 py-3.5 text-[15px] font-semibold inline-flex items-center gap-2 transition-colors hover:bg-[#38b754]">
                Start collecting <span>→</span>
              </Link>
              <a href="#stats" className="bg-transparent border border-[rgba(255,255,255,0.16)] rounded-[10px] px-5.5 py-3.5 text-[15px] font-medium inline-flex items-center gap-2 transition-colors hover:bg-[rgba(255,255,255,0.04)]">
                How it works
              </a>
            </div>
          </section>

          <aside className="border-l border-[var(--border)] pl-10 pt-2">
            <div className="text-[clamp(64px,7.4vw,110px)] font-black leading-[0.95] tracking-[-0.05em] text-[#aebeff]" style={{ fontVariantNumeric: 'tabular-nums' }}>
              2,400
            </div>
            <div className="mt-3.5 text-xs uppercase tracking-[0.16em] text-[var(--foreground-secondary)] max-w-[16ch] leading-[1.5]">
              Active collection tasks across 41 countries
            </div>

            <div className="mt-11 border-t border-[var(--border)] pt-6">
              <div className="text-[15px] font-semibold text-white">Are you a lab?</div>
              <Link href="/signup" className="mt-2 inline-flex items-center gap-1.5 text-[15px] font-semibold text-[#aebeff] transition-colors hover:text-white">
                Set up in 2 minutes <span>→</span>
              </Link>
            </div>
          </aside>
        </div>

        <section id="stats" className="mt-24 border-t border-[var(--border)] pt-14">
          <div className="flex items-baseline justify-between gap-6 mb-9">
            <h2 className="text-[22px] font-bold tracking-[-0.01em] text-white">
              The data bottleneck
              <span className="italic font-normal text-[var(--foreground-secondary)]" style={{ fontFamily: 'Georgia, serif' }}> in numbers.</span>
            </h2>
            <span className="text-xs uppercase tracking-[0.2em] text-[var(--foreground-tertiary)]">2025 — Industry Indicators</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-0 border-t border-b border-[var(--border)]">
            <div className="md:border-r border-[var(--border)] py-9 md:pr-8 md:py-9">
              <div className="text-[clamp(48px,5.6vw,76px)] font-black tracking-[-0.04em] text-white leading-[0.95]" style={{ fontVariantNumeric: 'tabular-nums' }}>
                73<span className="text-[0.42em] font-semibold text-[var(--foreground-secondary)] tracking-[0.04em]">%</span>
              </div>
              <div className="mt-3.5 text-[15px] leading-[1.5] text-[var(--foreground-secondary)] max-w-[32ch]">
                of robotics teams cite scarcity of real-world data as their top development blocker.
              </div>
              <div className="mt-auto pt-3 text-xs uppercase tracking-[0.14em] text-[var(--foreground-tertiary)]">
                SRC · Embodied AI Survey &apos;25
              </div>
            </div>

            <div className="md:border-r border-[var(--border)] py-9 md:px-8 md:py-9">
              <div className="text-[clamp(48px,5.6vw,76px)] font-black tracking-[-0.04em] leading-[0.95]" style={{ color: '#99ddaa', fontVariantNumeric: 'tabular-nums' }}>
                47<span className="text-[0.42em] font-semibold text-[var(--foreground-secondary)] tracking-[0.04em]">% YoY</span>
              </div>
              <div className="mt-3.5 text-[15px] leading-[1.5] text-[var(--foreground-secondary)] max-w-[32ch]">
                growth in demand for embodied training data — outpacing text and image markets combined.
              </div>
              <div className="mt-auto pt-3 text-xs uppercase tracking-[0.14em] text-[var(--foreground-tertiary)]">
                SRC · MarketScope Labs
              </div>
            </div>

            <div className="py-9 md:pl-8 md:py-9">
              <div className="text-[clamp(48px,5.6vw,76px)] font-black tracking-[-0.04em] leading-[0.95]" style={{ color: '#d8a347', fontVariantNumeric: 'tabular-nums' }}>
                $8.4<span className="text-[0.42em] font-semibold text-[var(--foreground-secondary)] tracking-[0.04em]">B by &apos;28</span>
              </div>
              <div className="mt-3.5 text-[15px] leading-[1.5] text-[var(--foreground-secondary)] max-w-[32ch]">
                projected spend on first-party, physical-world AI training data within three years.
              </div>
              <div className="mt-auto pt-3 text-xs uppercase tracking-[0.14em] text-[var(--foreground-tertiary)]">
                SRC · Pierce &amp; Co. Research
              </div>
            </div>
          </div>

          <Ticker />
        </section>
      </main>
    </div>
  )
}
