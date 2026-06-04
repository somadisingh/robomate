'use client'

import { useEffect, useImperativeHandle, useRef, useState, forwardRef } from 'react'

import type { CameraPath, SceneViewHandle } from './use-camera-sync'

export type SceneViewRef = SceneViewHandle

export type SceneLayers = {
  splats: boolean
  trajectory: boolean
  seedPoints: boolean
}

export type SplatScene = {
  splatUrl: string
  cameraPathUrl: string
  seedPointsUrl: string | null
  numGaussians: number
  frameCount: number
  fps: number
}

type Props = {
  scene: SplatScene
  layers: SceneLayers
  cameraPath: CameraPath | null
  /** When true, WASD/QE move the camera and click-drag rotates it.
      When false, camera is driven externally (via setCameraPathFrame). */
  manualControl: boolean
}

type ThreeNS = typeof import('three')

type SceneRefs = {
  scene: import('three').Scene
  camera: import('three').PerspectiveCamera
  renderer: import('three').WebGLRenderer
  sceneRoot: import('three').Group
  splatMesh: import('@sparkjsdev/spark').SplatMesh | null
  trajectory: import('three').Line | null
  seedPoints: import('three').Points | null
  THREE: ThreeNS
  destroyed: boolean
  manualControl: boolean
  /** Reusable scratch buffers. */
  tmpQuat: import('three').Quaternion
  tmpMatrix: import('three').Matrix4
  tmpForward: import('three').Vector3
  tmpRight: import('three').Vector3
  tmpUp: import('three').Vector3
  tmpMove: import('three').Vector3
  tmpYawQ: import('three').Quaternion
  tmpPitchQ: import('three').Quaternion
}

const WORLD_UP = { x: 0, y: 1, z: 0 } as const
const MOVE_SPEED = 0.75
const FAST_MULT = 4.0
const MOUSE_SENS = 0.0022
const CAMERA_KEYS = new Set([
  'KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyE', 'KeyQ',
])

type LoadPhase = 'fetching' | 'decoding' | 'ready' | 'error'

type LoadState = {
  phase: LoadPhase
  /** 0..1 fraction during 'fetching' (null when content-length is unknown). */
  progress: number | null
  bytesReceived: number
  bytesTotal: number
  error: string | null
}

const SceneView = forwardRef<SceneViewHandle, Props>(function SceneView(
  { scene, layers, cameraPath, manualControl },
  ref,
) {
  const containerRef = useRef<HTMLDivElement>(null)
  const refsRef = useRef<SceneRefs | null>(null)
  const [load, setLoad] = useState<LoadState>({
    phase: 'fetching',
    progress: null,
    bytesReceived: 0,
    bytesTotal: 0,
    error: null,
  })

  // Refs-of-props. The mount IIFE is async; by the time it finishes wiring up
  // `refsRef.current`, the manualControl-sync useEffect has already run once
  // (against a null refs) and won't fire again until the prop *changes*.
  // We read these refs inside the IIFE to initialize `refs.manualControl`
  // and the splat's initial visibility correctly.
  const manualControlRef = useRef(manualControl)
  const layersRef = useRef(layers)
  useEffect(() => { manualControlRef.current = manualControl }, [manualControl])
  useEffect(() => { layersRef.current = layers }, [layers])

  // Mount three.js + sparkjs once.
  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    let cancelled = false
    let cleanupListeners: (() => void) | null = null

    ;(async () => {
      try {
        const THREE: ThreeNS = await import('three')
        const { SparkRenderer, SplatMesh } = await import('@sparkjsdev/spark')

        if (cancelled || !container) return

        const width = container.clientWidth || 800
        const height = container.clientHeight || 600

        const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' })
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
        renderer.setSize(width, height)
        container.appendChild(renderer.domElement)
        renderer.domElement.style.outline = 'none'
        renderer.domElement.style.touchAction = 'none'
        renderer.domElement.tabIndex = 0  // allow focus for keyboard

        const sceneObj = new THREE.Scene()
        sceneObj.background = new THREE.Color(0x0d1014)
        const camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 1000)
        camera.position.set(0, 1.3, 8)

        const spark = new SparkRenderer({ renderer })
        sceneObj.add(spark)

        const sceneRoot = new THREE.Group()
        sceneObj.add(sceneRoot)

        // Set up refs + listeners + render loop BEFORE the network fetch so the
        // user can pan/drag and WASD around the (empty) scene the moment they
        // enter 3D mode. The splat is added to `refs.splatMesh` later.
        const refs: SceneRefs = {
          scene: sceneObj,
          camera,
          renderer,
          sceneRoot,
          splatMesh: null,
          trajectory: null,
          seedPoints: null,
          THREE,
          destroyed: false,
          // IMPORTANT: read the latest prop via ref. The manualControl-sync
          // useEffect already fired before this point (against a null refs),
          // so without this we'd be stuck at `false` until the prop changes.
          manualControl: manualControlRef.current,
          tmpQuat: new THREE.Quaternion(),
          tmpMatrix: new THREE.Matrix4(),
          tmpForward: new THREE.Vector3(),
          tmpRight: new THREE.Vector3(),
          tmpUp: new THREE.Vector3(),
          tmpMove: new THREE.Vector3(),
          tmpYawQ: new THREE.Quaternion(),
          tmpPitchQ: new THREE.Quaternion(),
        }
        refsRef.current = refs
        renderer.domElement.style.cursor = refs.manualControl ? 'grab' : 'default'

        const worldUp = new THREE.Vector3(WORLD_UP.x, WORLD_UP.y, WORLD_UP.z)
        const keyState = new Set<string>()
        let pointerDown = false
        const clock = new THREE.Clock()

        const rotateCamera = (deltaYaw: number, deltaPitch: number) => {
          // POV-relative rotation: yaw around the camera's LOCAL up and pitch
          // around its LOCAL right. Without this, a tilted/rolled starting
          // pose (e.g. iPhone held in landscape — common for captured paths)
          // makes horizontal drags feel vertical and vice-versa.
          if (deltaYaw !== 0) {
            refs.tmpUp.set(0, 1, 0).applyQuaternion(refs.camera.quaternion).normalize()
            refs.tmpYawQ.setFromAxisAngle(refs.tmpUp, deltaYaw)
            refs.camera.quaternion.premultiply(refs.tmpYawQ)
          }
          if (deltaPitch !== 0) {
            refs.tmpRight.set(1, 0, 0).applyQuaternion(refs.camera.quaternion).normalize()
            refs.tmpPitchQ.setFromAxisAngle(refs.tmpRight, deltaPitch)
            refs.camera.quaternion.premultiply(refs.tmpPitchQ)
          }
          refs.camera.quaternion.normalize()
          refs.camera.updateMatrixWorld()
        }

        const isFormField = (el: EventTarget | null): boolean => {
          if (!(el instanceof HTMLElement)) return false
          const tag = el.tagName
          return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable
        }

        const onKeyDown = (e: KeyboardEvent) => {
          if (!refs.manualControl) return
          if (isFormField(e.target)) return
          if (CAMERA_KEYS.has(e.code) || e.code === 'ShiftLeft' || e.code === 'ShiftRight') {
            keyState.add(e.code)
            if (CAMERA_KEYS.has(e.code)) e.preventDefault()
          }
        }
        const onKeyUp = (e: KeyboardEvent) => {
          keyState.delete(e.code)
        }
        const onBlur = () => { keyState.clear(); pointerDown = false }
        window.addEventListener('keydown', onKeyDown)
        window.addEventListener('keyup', onKeyUp)
        window.addEventListener('blur', onBlur)

        const onPointerDown = (e: PointerEvent) => {
          if (!refs.manualControl || e.button !== 0) return
          pointerDown = true
          renderer.domElement.setPointerCapture(e.pointerId)
          renderer.domElement.style.cursor = 'grabbing'
        }
        const onPointerMove = (e: PointerEvent) => {
          if (!refs.manualControl || !pointerDown) return
          rotateCamera(-e.movementX * MOUSE_SENS, -e.movementY * MOUSE_SENS)
        }
        const releasePointer = (e: PointerEvent) => {
          pointerDown = false
          if (renderer.domElement.hasPointerCapture(e.pointerId)) {
            renderer.domElement.releasePointerCapture(e.pointerId)
          }
          renderer.domElement.style.cursor = refs.manualControl ? 'grab' : 'default'
        }
        renderer.domElement.addEventListener('pointerdown', onPointerDown)
        renderer.domElement.addEventListener('pointermove', onPointerMove)
        renderer.domElement.addEventListener('pointerup', releasePointer)
        renderer.domElement.addEventListener('pointercancel', releasePointer)

        cleanupListeners = () => {
          window.removeEventListener('keydown', onKeyDown)
          window.removeEventListener('keyup', onKeyUp)
          window.removeEventListener('blur', onBlur)
          renderer.domElement.removeEventListener('pointerdown', onPointerDown)
          renderer.domElement.removeEventListener('pointermove', onPointerMove)
          renderer.domElement.removeEventListener('pointerup', releasePointer)
          renderer.domElement.removeEventListener('pointercancel', releasePointer)
        }

        // Render loop: process WASD/QE movement when manual control is on,
        // then render the scene. The camera-sync hook (parent) handles snapping
        // to path frames on its own when sync is engaged.
        const loop = () => {
          if (refs.destroyed) return
          const dt = Math.min(clock.getDelta(), 0.05)

          if (refs.manualControl) {
            refs.tmpMove.set(0, 0, 0)
            refs.camera.getWorldDirection(refs.tmpForward).normalize()
            refs.tmpRight.set(1, 0, 0).applyQuaternion(refs.camera.quaternion).normalize()
            if (keyState.has('KeyW')) refs.tmpMove.add(refs.tmpForward)
            if (keyState.has('KeyS')) refs.tmpMove.sub(refs.tmpForward)
            if (keyState.has('KeyD')) refs.tmpMove.add(refs.tmpRight)
            if (keyState.has('KeyA')) refs.tmpMove.sub(refs.tmpRight)
            if (keyState.has('KeyE')) refs.tmpMove.add(worldUp)
            if (keyState.has('KeyQ')) refs.tmpMove.sub(worldUp)
            if (refs.tmpMove.lengthSq() > 0) {
              const fast = keyState.has('ShiftLeft') || keyState.has('ShiftRight')
              const speed = MOVE_SPEED * (fast ? FAST_MULT : 1)
              refs.camera.position.addScaledVector(refs.tmpMove.normalize(), speed * dt)
            }
          }
          refs.renderer.render(refs.scene, refs.camera)
        }
        renderer.setAnimationLoop(loop)

        // Resize handling
        const ro = new ResizeObserver(() => {
          if (refs.destroyed) return
          const w = container.clientWidth || 800
          const h = container.clientHeight || 600
          refs.camera.aspect = w / h
          refs.camera.updateProjectionMatrix()
          refs.renderer.setSize(w, h)
        })
        ro.observe(container)
        const prevCleanup = cleanupListeners
        cleanupListeners = () => {
          prevCleanup?.()
          ro.disconnect()
        }

        // === Now stream the .spz with progress tracking, then hand it to
        // SplatMesh via a blob URL. Listeners/loop are already live so the
        // user can interact while this is downloading.
        let blobUrl: string | null = null
        try {
          const res = await fetch(scene.splatUrl)
          if (!res.ok) throw new Error(`HTTP ${res.status} fetching splat.spz`)
          const total = Number(res.headers.get('content-length') || 0)
          const reader = res.body?.getReader()
          if (!reader) throw new Error('Streaming fetch unavailable')
          setLoad((p) => ({ ...p, phase: 'fetching', bytesTotal: total, bytesReceived: 0, progress: total ? 0 : null }))
          const chunks: Uint8Array[] = []
          let received = 0
          while (true) {
            const { done, value } = await reader.read()
            if (done) break
            if (cancelled) { reader.cancel(); return }
            chunks.push(value)
            received += value.byteLength
            setLoad((p) => ({
              ...p,
              bytesReceived: received,
              bytesTotal: total || received,
              progress: total > 0 ? received / total : null,
            }))
          }
          const blob = new Blob(chunks as BlobPart[], { type: 'application/octet-stream' })
          blobUrl = URL.createObjectURL(blob)
        } catch (err) {
          if (!cancelled) setLoad((p) => ({ ...p, phase: 'error', error: String(err) }))
          return
        }
        if (cancelled) {
          if (blobUrl) URL.revokeObjectURL(blobUrl)
          return
        }

        setLoad((p) => ({ ...p, phase: 'decoding' }))
        const splatMesh = new SplatMesh({ url: blobUrl, lod: true })
        // Apply current layer visibility before adding to scene.
        splatMesh.visible = layersRef.current.splats
        sceneRoot.add(splatMesh)
        refs.splatMesh = splatMesh

        // Report ready when the splat has loaded.
        splatMesh.initialized
          .then(() => {
            if (cancelled) return
            setLoad((p) => ({ ...p, phase: 'ready' }))
            // We can release the blob URL once the mesh has fully parsed.
            if (blobUrl) URL.revokeObjectURL(blobUrl)
          })
          .catch((err: unknown) => {
            if (cancelled) return
            setLoad((p) => ({ ...p, phase: 'error', error: String(err) }))
            if (blobUrl) URL.revokeObjectURL(blobUrl)
          })
      } catch (err) {
        if (!cancelled) setLoad((p) => ({ ...p, phase: 'error', error: String(err) }))
      }
    })()

    return () => {
      cancelled = true
      cleanupListeners?.()
      const refs = refsRef.current
      if (refs) {
        refs.destroyed = true
        refs.renderer.setAnimationLoop(null)
        refs.renderer.dispose()
        if (refs.renderer.domElement.parentElement === container) {
          container.removeChild(refs.renderer.domElement)
        }
        refsRef.current = null
      }
    }
    // We only want to mount once; subsequent layer/url changes are handled by other effects.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Track manualControl on the refs so the render-loop closure sees latest.
  useEffect(() => {
    const refs = refsRef.current
    if (!refs) return
    refs.manualControl = manualControl
    refs.renderer.domElement.style.cursor = manualControl ? 'grab' : 'default'
  }, [manualControl])

  // Apply layer visibility.
  useEffect(() => {
    const refs = refsRef.current
    if (!refs) return
    if (refs.splatMesh) refs.splatMesh.visible = layers.splats
    if (refs.trajectory) refs.trajectory.visible = layers.trajectory
    if (refs.seedPoints) refs.seedPoints.visible = layers.seedPoints
  }, [layers])

  // Imperative handle: parent feeds camera-path frames.
  useImperativeHandle(
    ref,
    () => ({
      setCameraPathFrame(frame) {
        const refs = refsRef.current
        if (!refs) return
        const { camera, tmpMatrix, tmpQuat } = refs
        const m = frame.transformMatrix
        tmpMatrix.set(
          m[0][0], m[0][1], m[0][2], m[0][3],
          m[1][0], m[1][1], m[1][2], m[1][3],
          m[2][0], m[2][1], m[2][2], m[2][3],
          m[3][0], m[3][1], m[3][2], m[3][3],
        )
        camera.position.setFromMatrixPosition(tmpMatrix)
        tmpQuat.setFromRotationMatrix(tmpMatrix)
        camera.quaternion.copy(tmpQuat)
        camera.updateMatrixWorld()
      },
    }),
    []
  )

  // Render the camera trajectory line whenever the camera path changes.
  useEffect(() => {
    const refs = refsRef.current
    if (!refs || !cameraPath) return
    const { THREE, sceneRoot } = refs
    if (refs.trajectory) {
      sceneRoot.remove(refs.trajectory)
      refs.trajectory.geometry.dispose()
      ;(refs.trajectory.material as import('three').Material).dispose()
      refs.trajectory = null
    }
    if (cameraPath.frames.length < 2) return
    const positions: import('three').Vector3[] = cameraPath.frames.map(
      (f) => new THREE.Vector3(f.position[0], f.position[1], f.position[2])
    )
    const geometry = new THREE.BufferGeometry().setFromPoints(positions)
    const material = new THREE.LineBasicMaterial({
      color: 0x4ade80,
      transparent: true,
      opacity: 0.9,
      depthTest: false,
    })
    const trajectory = new THREE.Line(geometry, material)
    trajectory.renderOrder = 10
    trajectory.visible = layers.trajectory
    sceneRoot.add(trajectory)
    refs.trajectory = trajectory
  }, [cameraPath, layers.trajectory])

  const showLoadOverlay = load.phase !== 'ready'
  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', background: '#0d1014' }}>
      <div
        ref={containerRef}
        style={{ position: 'absolute', inset: 0 }}
      />

      {showLoadOverlay && <SceneLoadingOverlay load={load} scene={scene} />}

      {manualControl && load.phase === 'ready' && (
        <div
          style={{
            position: 'absolute',
            bottom: 12,
            left: 12,
            padding: '6px 10px',
            background: 'rgba(13,16,20,0.7)',
            color: '#cdd0d6',
            fontSize: 11,
            fontFamily: 'ui-monospace,SFMono-Regular,Menlo,monospace',
            borderRadius: 4,
            lineHeight: 1.5,
            pointerEvents: 'none',
          }}
        >
          <div><b>WASD</b> move · <b>Q/E</b> down/up · <b>shift</b> fast</div>
          <div><b>drag</b> to look</div>
        </div>
      )}
    </div>
  )
})

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`
  return `${(n / 1024 / 1024).toFixed(2)} MB`
}

function SceneLoadingOverlay({ load, scene }: { load: LoadState; scene: SplatScene }) {
  const pct = load.progress != null ? Math.round(load.progress * 100) : null
  const isError = load.phase === 'error'
  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        zIndex: 20,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'column',
        background: 'radial-gradient(circle at center, rgba(13,16,20,0.85) 0%, rgba(7,8,10,0.96) 70%)',
        color: '#e8e8ea',
        fontFamily: 'var(--font-geist-sans), system-ui, sans-serif',
        pointerEvents: 'none',
      }}
    >
      <style>{`
        @keyframes splat-spin { to { transform: rotate(360deg); } }
        @keyframes splat-pulse {
          0%, 100% { opacity: 0.55; }
          50% { opacity: 1; }
        }
      `}</style>

      {isError ? (
        <div
          style={{
            width: 72, height: 72, borderRadius: '50%',
            border: '2px solid #f87171',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 36, fontWeight: 600, color: '#f87171',
          }}
        >!</div>
      ) : (
        <div
          style={{
            position: 'relative',
            width: 84,
            height: 84,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {/* Outer rotating ring */}
          <div
            style={{
              position: 'absolute',
              inset: 0,
              border: '3px solid rgba(255,255,255,0.08)',
              borderTopColor: 'var(--studio-accent, #c2e02b)',
              borderRadius: '50%',
              animation: 'splat-spin 1.1s linear infinite',
            }}
          />
          {/* Center percentage when known */}
          {pct != null ? (
            <span style={{ fontSize: 18, fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>
              {pct}%
            </span>
          ) : (
            <span
              style={{
                fontSize: 12,
                color: '#cdd0d6',
                animation: 'splat-pulse 1.4s ease-in-out infinite',
              }}
            >
              {load.phase === 'decoding' ? 'decode' : '…'}
            </span>
          )}
        </div>
      )}

      <div style={{ marginTop: 24, fontSize: 16, letterSpacing: 0.3 }}>
        {isError
          ? 'Failed to load 3D scene'
          : load.phase === 'decoding'
            ? 'Decoding splats'
            : 'Loading 3D scene'}
      </div>

      <div
        style={{
          marginTop: 6,
          fontSize: 12,
          color: '#8a8a93',
          fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
          textAlign: 'center',
        }}
      >
        {isError ? (
          <span style={{ color: '#f87171' }}>{load.error ?? 'unknown error'}</span>
        ) : (
          <>
            {scene.numGaussians.toLocaleString()} gaussians
            {load.phase === 'fetching' && load.bytesTotal > 0 && (
              <>
                {' · '}
                {formatBytes(load.bytesReceived)} / {formatBytes(load.bytesTotal)}
              </>
            )}
          </>
        )}
      </div>

      {/* Progress bar when we know the total */}
      {!isError && load.bytesTotal > 0 && load.phase === 'fetching' && (
        <div
          style={{
            marginTop: 18,
            width: 280,
            height: 4,
            background: 'rgba(255,255,255,0.08)',
            borderRadius: 2,
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              width: `${pct ?? 0}%`,
              height: '100%',
              background: 'var(--studio-accent, #c2e02b)',
              transition: 'width 100ms linear',
            }}
          />
        </div>
      )}
    </div>
  )
}

export default SceneView
