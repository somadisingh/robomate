-- Add 'gemini_temporal_annotations' to the recording_analysis_jobs kind enum so
-- the Gemini temporal annotator (full-video timestamped annotations + Pinecone
-- semantic-search embedding) can register/track a job alongside the other
-- analyzers. Mirrors 20260524180000_add_gaussian_splat_kind.sql.

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
      'gaussian_splat',
      'gemini_temporal_annotations'
    )
  );
