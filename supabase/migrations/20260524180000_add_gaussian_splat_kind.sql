-- Add 'gaussian_splat' to the recording_analysis_jobs kind enum so the
-- splatfacto pipeline can register/track jobs alongside the other analyzers.

alter table public.recording_analysis_jobs
  drop constraint if exists recording_analysis_jobs_kind_check;

alter table public.recording_analysis_jobs
  add constraint recording_analysis_jobs_kind_check check (
    kind in (
      'gemini_eval',
      'mediapipe_hands',
      'yolo_objects',
      'sam_segments',
      'temporal_actions',
      'gaussian_splat'
    )
  );
