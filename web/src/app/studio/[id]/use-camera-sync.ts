'use client'

import { useEffect } from 'react'

type CameraPathFrame = {
  videoFrameIndex: number | null
  timestamp: number
  position: [number, number, number]
  transformMatrix: number[][] // 4x4 row-major
}

export type CameraPath = {
  count: number
  durationSeconds: number
  fps: number
  frames: CameraPathFrame[]
}

export type SceneViewHandle = {
  /** Apply a camera-path frame to the three.js camera (model-space pose). */
  setCameraPathFrame: (frame: CameraPathFrame) => void
}

/**
 * Drives the SceneView's camera from a `<video>` element's currentTime.
 *
 * When `enabled` is true, snaps the camera to the nearest camera_path frame on
 * every video frame (uses requestVideoFrameCallback when available, falling
 * back to timeupdate). Toggling enabled false → true re-applies the current
 * frame on the next render so the camera "snaps back" after manual control.
 */
export function useCameraSync({
  videoRef,
  cameraPath,
  sceneHandle,
  enabled,
}: {
  videoRef: React.RefObject<HTMLVideoElement | null>
  cameraPath: CameraPath | null
  sceneHandle: SceneViewHandle | null
  enabled: boolean
}) {
  useEffect(() => {
    if (!enabled) return
    const video = videoRef.current
    if (!video || !cameraPath || cameraPath.frames.length === 0 || !sceneHandle) return

    const frames = cameraPath.frames
    const startTimestamp = frames[0].timestamp
    const elapsedTimestamps = frames.map((f) => f.timestamp - startTimestamp)

    // Lower-bound binary search on elapsed time (seconds).
    const lookup = (t: number): CameraPathFrame => {
      if (t <= elapsedTimestamps[0]) return frames[0]
      const last = elapsedTimestamps.length - 1
      if (t >= elapsedTimestamps[last]) return frames[last]
      let lo = 0
      let hi = last
      while (lo < hi - 1) {
        const mid = (lo + hi) >> 1
        if (elapsedTimestamps[mid] <= t) lo = mid
        else hi = mid
      }
      // Snap to whichever neighbour is closer (no interp in v1).
      const a = elapsedTimestamps[lo]
      const b = elapsedTimestamps[lo + 1]
      return t - a <= b - t ? frames[lo] : frames[lo + 1]
    }

    const apply = (currentTime: number) => {
      sceneHandle.setCameraPathFrame(lookup(currentTime))
    }

    type RVFCMeta = { mediaTime: number }
    type VideoEl = HTMLVideoElement & {
      requestVideoFrameCallback?: (cb: (now: number, meta: RVFCMeta) => void) => number
      cancelVideoFrameCallback?: (h: number) => void
    }
    const ve = video as VideoEl
    let handle = 0
    let cancelled = false
    const tick = (_now: number, meta: RVFCMeta) => {
      if (cancelled) return
      apply(meta.mediaTime)
      if (ve.requestVideoFrameCallback) handle = ve.requestVideoFrameCallback(tick)
    }
    if (typeof ve.requestVideoFrameCallback === 'function') {
      handle = ve.requestVideoFrameCallback(tick)
    }
    const onTime = () => apply(video.currentTime)
    const onSeeked = () => apply(video.currentTime)
    video.addEventListener('timeupdate', onTime)
    video.addEventListener('seeked', onSeeked)
    apply(video.currentTime)

    return () => {
      cancelled = true
      if (handle && ve.cancelVideoFrameCallback) ve.cancelVideoFrameCallback(handle)
      video.removeEventListener('timeupdate', onTime)
      video.removeEventListener('seeked', onSeeked)
    }
  }, [enabled, videoRef, cameraPath, sceneHandle])
}
