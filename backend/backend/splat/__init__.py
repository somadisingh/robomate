"""Gaussian splat training pipeline pieces shared between Modal and the
playground research scripts.

The package is intentionally split so each module has one job and can be
exercised in isolation:

- ``dataset``: build a nerfstudio dataset directory from a Supabase recording
  (video.mp4, depth.bin, poses.jsonl, imu.jsonl, intrinsics.json).
- ``convert``: convert a ``splat.ply`` exported by nerfstudio to the compact
  ``.spz`` format consumed by sparkjs.
- ``camera_path``: build a per-frame camera-path JSON aligned to the source
  video, in the same coordinate frame as the exported ``.spz``.
- ``manifest``: typed builder for the ``manifest.json`` artifact that the web
  studio consumes.
"""
