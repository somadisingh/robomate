'use client'

import type { SceneLayers } from './scene-view'

type Props = {
  layers: SceneLayers
  onLayersChange: (next: SceneLayers) => void
  cameraMode: 'synced' | 'free'
  onCameraModeChange: (next: 'synced' | 'free') => void
  summary: {
    numGaussians: number
    frameCount: number
    fps: number
  } | null
}

const ROW_STYLE: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  padding: '6px 10px',
  fontSize: 12,
  color: '#cdd0d6',
  cursor: 'pointer',
  userSelect: 'none',
}

export default function SceneControls({
  layers,
  onLayersChange,
  cameraMode,
  onCameraModeChange,
  summary,
}: Props) {
  const toggle = (key: keyof SceneLayers) => () =>
    onLayersChange({ ...layers, [key]: !layers[key] })

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
        padding: 8,
        background: '#15181d',
        border: '1px solid #1f2329',
        borderRadius: 6,
        fontFamily: 'ui-sans-serif,system-ui,sans-serif',
      }}
    >
      <div style={{ fontSize: 11, color: '#8a8d94', padding: '4px 10px 2px', textTransform: 'uppercase', letterSpacing: 0.6 }}>
        3D layers
      </div>
      <label style={ROW_STYLE}>
        <input type="checkbox" checked={layers.splats} onChange={toggle('splats')} />
        <span>Splats</span>
        {summary && (
          <span style={{ marginLeft: 'auto', color: '#8a8d94', fontSize: 10 }}>
            {summary.numGaussians.toLocaleString()} gaussians
          </span>
        )}
      </label>
      <label style={ROW_STYLE}>
        <input type="checkbox" checked={layers.trajectory} onChange={toggle('trajectory')} />
        <span>Camera trajectory</span>
        {summary && (
          <span style={{ marginLeft: 'auto', color: '#8a8d94', fontSize: 10 }}>
            {summary.frameCount} frames
          </span>
        )}
      </label>
      <label style={ROW_STYLE}>
        <input type="checkbox" checked={layers.seedPoints} onChange={toggle('seedPoints')} />
        <span>Seed point cloud</span>
      </label>

      <div style={{ height: 1, background: '#1f2329', margin: '6px 0' }} />
      <div style={{ fontSize: 11, color: '#8a8d94', padding: '4px 10px 2px', textTransform: 'uppercase', letterSpacing: 0.6 }}>
        Camera
      </div>
      <div style={{ display: 'flex', gap: 4, padding: '0 10px 6px' }}>
        <button
          onClick={() => onCameraModeChange('synced')}
          style={{
            flex: 1,
            padding: '6px 10px',
            background: cameraMode === 'synced' ? '#2563eb' : '#1f2329',
            color: cameraMode === 'synced' ? 'white' : '#cdd0d6',
            border: '1px solid #2a2f37',
            borderRadius: 4,
            cursor: 'pointer',
            fontSize: 12,
          }}
        >
          Synced
        </button>
        <button
          onClick={() => onCameraModeChange('free')}
          style={{
            flex: 1,
            padding: '6px 10px',
            background: cameraMode === 'free' ? '#2563eb' : '#1f2329',
            color: cameraMode === 'free' ? 'white' : '#cdd0d6',
            border: '1px solid #2a2f37',
            borderRadius: 4,
            cursor: 'pointer',
            fontSize: 12,
          }}
        >
          Free
        </button>
      </div>
    </div>
  )
}
