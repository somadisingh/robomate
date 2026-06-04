-- Base schema for the Copilot Hackathon data-collection marketplace.
-- Reconstructed from web/iOS/edge-function/backend usage because the original
-- base DDL was never committed (only analysis-pipeline deltas were).
-- This migration MUST run before 20260524020850_backend_analysis_pipeline.sql,
-- which ALTERs public.recordings / public.tasks and references them via FK.

-- ---------------------------------------------------------------------------
-- profiles (1:1 with auth.users, populated by handle_new_user trigger)
-- ---------------------------------------------------------------------------
create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  role text not null default 'collector' check (role in ('lab', 'collector')),
  display_name text,
  created_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- collector_profiles (onboarding questionnaire / capabilities)
-- ---------------------------------------------------------------------------
create table if not exists public.collector_profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  capabilities text[] not null default '{}',
  location_city text,
  questionnaire_data jsonb,
  created_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- tasks (a.k.a. bounties created by labs)
-- ---------------------------------------------------------------------------
create table if not exists public.tasks (
  id uuid primary key default gen_random_uuid(),
  lab_id uuid not null references public.profiles(id) on delete cascade,
  title text not null,
  description text,
  data_type text,
  required_capabilities text[] not null default '{}',
  bounty_amount numeric not null default 0,
  quantity_needed integer not null default 1,
  quantity_filled integer not null default 0,
  status text not null default 'open',
  deadline timestamptz,
  reference_path text,
  reference_brief text,
  metadata jsonb,
  reference_assets jsonb,
  created_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- recordings (one per captured bundle; bounty_id -> tasks.id)
-- is_scoring is required here because 20260524020850 only ALTERs its default.
-- ---------------------------------------------------------------------------
create table if not exists public.recordings (
  id uuid primary key default gen_random_uuid(),
  bounty_id uuid references public.tasks(id) on delete set null,
  collector_id uuid not null references public.profiles(id) on delete cascade,
  device_model text,
  duration_ms integer,
  size_bytes bigint,
  gps_lat double precision,
  gps_lon double precision,
  gps_accuracy_m double precision,
  storage_path text not null,
  streams jsonb,
  status text not null default 'uploaded',
  is_scoring boolean not null default true,
  success boolean,
  success_reasoning text,
  score numeric,
  score_reasoning text,
  created_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- submissions (marketplace ledger row; task_id is NOT NULL per edge function)
-- ---------------------------------------------------------------------------
create table if not exists public.submissions (
  id uuid primary key default gen_random_uuid(),
  task_id uuid not null references public.tasks(id) on delete cascade,
  collector_id uuid not null references public.profiles(id) on delete cascade,
  storage_path text not null,
  status text not null default 'pending' check (status in ('pending', 'approved', 'rejected')),
  metadata jsonb,
  created_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- earnings (created when a lab approves a submission)
-- ---------------------------------------------------------------------------
create table if not exists public.earnings (
  id uuid primary key default gen_random_uuid(),
  collector_id uuid not null references public.profiles(id) on delete cascade,
  submission_id uuid references public.submissions(id) on delete cascade,
  amount numeric not null default 0,
  status text not null default 'pending',
  created_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- handle_new_user: mirror auth.users -> public.profiles on signup
-- ---------------------------------------------------------------------------
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (id, role, display_name)
  values (
    new.id,
    coalesce(nullif(new.raw_user_meta_data ->> 'role', ''), 'collector'),
    coalesce(
      nullif(new.raw_user_meta_data ->> 'display_name', ''),
      nullif(new.raw_user_meta_data ->> 'full_name', ''),
      nullif(new.raw_user_meta_data ->> 'name', ''),
      split_part(coalesce(new.email, ''), '@', 1)
    )
  )
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- ---------------------------------------------------------------------------
-- increment_quantity_filled RPC (called by the approve server action)
-- ---------------------------------------------------------------------------
create or replace function public.increment_quantity_filled(task_id uuid)
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
  update public.tasks t
     set quantity_filled = t.quantity_filled + 1
   where t.id = increment_quantity_filled.task_id;
end;
$$;

-- ---------------------------------------------------------------------------
-- Row Level Security
-- ---------------------------------------------------------------------------
alter table public.profiles enable row level security;
drop policy if exists "profiles_select_all" on public.profiles;
create policy "profiles_select_all" on public.profiles
  for select to authenticated using (true);
drop policy if exists "profiles_update_own" on public.profiles;
create policy "profiles_update_own" on public.profiles
  for update to authenticated using (id = auth.uid()) with check (id = auth.uid());

alter table public.collector_profiles enable row level security;
drop policy if exists "collector_profiles_all_own" on public.collector_profiles;
create policy "collector_profiles_all_own" on public.collector_profiles
  for all to authenticated using (user_id = auth.uid()) with check (user_id = auth.uid());

alter table public.tasks enable row level security;
drop policy if exists "tasks_select_all" on public.tasks;
create policy "tasks_select_all" on public.tasks
  for select to authenticated using (true);
drop policy if exists "tasks_insert_lab" on public.tasks;
create policy "tasks_insert_lab" on public.tasks
  for insert to authenticated with check (lab_id = auth.uid());
drop policy if exists "tasks_update_lab" on public.tasks;
create policy "tasks_update_lab" on public.tasks
  for update to authenticated using (lab_id = auth.uid()) with check (lab_id = auth.uid());
drop policy if exists "tasks_delete_lab" on public.tasks;
create policy "tasks_delete_lab" on public.tasks
  for delete to authenticated using (lab_id = auth.uid());

alter table public.recordings enable row level security;
drop policy if exists "recordings_select_own_collector" on public.recordings;
create policy "recordings_select_own_collector" on public.recordings
  for select to authenticated using (collector_id = auth.uid());
drop policy if exists "recordings_select_lab" on public.recordings;
create policy "recordings_select_lab" on public.recordings
  for select to authenticated using (
    exists (
      select 1 from public.tasks t
      where t.id = recordings.bounty_id and t.lab_id = auth.uid()
    )
  );
drop policy if exists "recordings_update_own_collector" on public.recordings;
create policy "recordings_update_own_collector" on public.recordings
  for update to authenticated using (collector_id = auth.uid()) with check (collector_id = auth.uid());

alter table public.submissions enable row level security;
drop policy if exists "submissions_select_own_collector" on public.submissions;
create policy "submissions_select_own_collector" on public.submissions
  for select to authenticated using (collector_id = auth.uid());
drop policy if exists "submissions_select_lab" on public.submissions;
create policy "submissions_select_lab" on public.submissions
  for select to authenticated using (
    exists (
      select 1 from public.tasks t
      where t.id = submissions.task_id and t.lab_id = auth.uid()
    )
  );
drop policy if exists "submissions_update_lab" on public.submissions;
create policy "submissions_update_lab" on public.submissions
  for update to authenticated using (
    exists (
      select 1 from public.tasks t
      where t.id = submissions.task_id and t.lab_id = auth.uid()
    )
  ) with check (true);

alter table public.earnings enable row level security;
drop policy if exists "earnings_select_own" on public.earnings;
create policy "earnings_select_own" on public.earnings
  for select to authenticated using (collector_id = auth.uid());
drop policy if exists "earnings_insert_lab" on public.earnings;
create policy "earnings_insert_lab" on public.earnings
  for insert to authenticated with check (
    exists (
      select 1
      from public.submissions s
      join public.tasks t on t.id = s.task_id
      where s.id = earnings.submission_id and t.lab_id = auth.uid()
    )
  );

-- ---------------------------------------------------------------------------
-- Storage: private 'recordings' bucket + permissive authenticated access.
-- ---------------------------------------------------------------------------
insert into storage.buckets (id, name, public)
values ('recordings', 'recordings', false)
on conflict (id) do nothing;

drop policy if exists "recordings_bucket_authenticated_all" on storage.objects;
create policy "recordings_bucket_authenticated_all" on storage.objects
  for all to authenticated
  using (bucket_id = 'recordings')
  with check (bucket_id = 'recordings');

-- ---------------------------------------------------------------------------
-- Realtime: labs subscribe to new submissions per task.
-- ---------------------------------------------------------------------------
do $$
begin
  alter publication supabase_realtime add table public.submissions;
exception
  when duplicate_object then null;
  when undefined_object then null;
end
$$;
