-- Add LiDAR depth grid dimensions to recordings so depth.bin can be decoded
-- without an out-of-band metadata file. Record format is:
--   per-frame: 8 bytes (double timestamp) + depth_width * depth_height * 4 bytes (float32 metres)

alter table public.recordings
  add column if not exists depth_width integer,
  add column if not exists depth_height integer,
  add column if not exists depth_frame_count integer;

-- Sanity check: if any of width/height/frame_count is set, all three must be set together
alter table public.recordings
  drop constraint if exists recordings_depth_dims_complete;
alter table public.recordings
  add constraint recordings_depth_dims_complete
  check (
    (depth_width is null and depth_height is null and depth_frame_count is null)
    or (depth_width is not null and depth_height is not null and depth_frame_count is not null)
  );
