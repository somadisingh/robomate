alter table public.recordings
  add column if not exists summary text,
  add column if not exists detected_objects jsonb,
  add column if not exists analysis_artifacts jsonb;

alter table public.recordings
  alter column is_scoring set default true;

alter table public.tasks
  add column if not exists objects text[] not null default '{}';

create table if not exists public.recording_analysis_jobs (
  id uuid primary key default gen_random_uuid(),
  recording_id uuid not null references public.recordings(id) on delete cascade,
  kind text not null,
  status text not null default 'pending',
  artifact_path text,
  summary jsonb,
  error text,
  started_at timestamptz,
  finished_at timestamptz,
  created_at timestamptz not null default now(),
  unique (recording_id, kind),
  constraint recording_analysis_jobs_kind_check check (
    kind in (
      'gemini_eval',
      'mediapipe_hands',
      'yolo_objects',
      'sam_segments',
      'temporal_actions'
    )
  ),
  constraint recording_analysis_jobs_status_check check (
    status in ('pending', 'running', 'succeeded', 'failed')
  )
);

alter table public.recording_analysis_jobs enable row level security;

drop policy if exists "collectors read own recording analysis jobs"
  on public.recording_analysis_jobs;
create policy "collectors read own recording analysis jobs"
  on public.recording_analysis_jobs
  for select
  to authenticated
  using (
    exists (
      select 1
      from public.recordings r
      where r.id = recording_analysis_jobs.recording_id
        and r.collector_id = auth.uid()
    )
  );

drop policy if exists "labs read task recording analysis jobs"
  on public.recording_analysis_jobs;
create policy "labs read task recording analysis jobs"
  on public.recording_analysis_jobs
  for select
  to authenticated
  using (
    exists (
      select 1
      from public.recordings r
      join public.tasks t on t.id = r.bounty_id
      where r.id = recording_analysis_jobs.recording_id
        and t.lab_id = auth.uid()
    )
  );

create index if not exists recording_analysis_jobs_recording_id_idx
  on public.recording_analysis_jobs(recording_id);

create index if not exists recordings_bounty_id_idx
  on public.recordings(bounty_id);
