import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import "./styles.css";

const viewport = document.querySelector("#viewport");
const datasetLabel = document.querySelector("#dataset-name");
const status = document.querySelector("#status");
const trackCameraButton = document.querySelector("#track-camera");
const startViewButton = document.querySelector("#start-view");
const overviewButton = document.querySelector("#overview");
const playbackTime = document.querySelector("#playback-time");
const sceneOffset = new THREE.Vector3(0.25, -0.7, 0.35);
const sceneScale = 0.55;
const defaultDataset = "iphone-data-3";
const requestedDataset = new URLSearchParams(window.location.search).get("dataset");
const datasetName = /^[A-Za-z0-9_.-]+$/.test(requestedDataset ?? "")
  ? requestedDataset
  : defaultDataset;

datasetLabel.textContent = `${datasetName}.spz`;
document.title = `${datasetName} Spark Viewer`;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111418);

const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.01,
  1000,
);
camera.position.set(0, 1.3, 8);

const renderer = new THREE.WebGLRenderer({
  antialias: false,
  powerPreference: "high-performance",
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
viewport.appendChild(renderer.domElement);
const clock = new THREE.Clock();

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0.45, 0);
controls.update();

const spark = new SparkRenderer({ renderer });
scene.add(spark);

const sceneRoot = new THREE.Group();
sceneRoot.position.copy(sceneOffset);
sceneRoot.scale.setScalar(sceneScale);
scene.add(sceneRoot);

const splats = new SplatMesh({
  url: `/splats/${datasetName}.spz`,
  lod: true,
});
sceneRoot.add(splats);

let cameraPath = null;
let viewFrames = [];
let playbackActive = false;
let playbackStartedAt = 0;
let playbackPausedAt = 0;
let playbackDuration = 0;
let freeCameraActive = false;
let pointerDragActive = false;

const playbackPosition = new THREE.Vector3();
const playbackQuaternion = new THREE.Quaternion();
const playbackForward = new THREE.Vector3();
const keyState = new Set();
const moveVector = new THREE.Vector3();
const cameraForward = new THREE.Vector3();
const cameraRight = new THREE.Vector3();
const cameraUp = new THREE.Vector3();
const worldUp = new THREE.Vector3(0, 1, 0);
const yawQuaternion = new THREE.Quaternion();
const pitchQuaternion = new THREE.Quaternion();
const freeCameraMoveSpeed = 0.75;
const freeCameraFastMultiplier = 4.0;
const freeCameraMouseSensitivity = 0.0022;
const freeCameraKeyLookSpeed = 1.8;

function matrixFromRows(rows) {
  const matrix = new THREE.Matrix4();
  matrix.set(
    rows[0][0],
    rows[0][1],
    rows[0][2],
    rows[0][3],
    rows[1][0],
    rows[1][1],
    rows[1][2],
    rows[1][3],
    rows[2][0],
    rows[2][1],
    rows[2][2],
    rows[2][3],
    rows[3][0],
    rows[3][1],
    rows[3][2],
    rows[3][3],
  );
  return matrix;
}

function scenePositionFromModel(position) {
  return new THREE.Vector3(...position).multiplyScalar(sceneScale).add(sceneOffset);
}

function formatSeconds(seconds) {
  return `${seconds.toFixed(1)}s`;
}

function updatePlaybackLabel(elapsedSeconds = playbackPausedAt) {
  playbackTime.textContent = `${formatSeconds(elapsedSeconds)} / ${formatSeconds(playbackDuration)}`;
}

function syncControlsTargetToCamera() {
  playbackForward.set(0, 0, -1).applyQuaternion(camera.quaternion);
  controls.target.copy(camera.position).add(playbackForward.multiplyScalar(1.5));
}

function prepareViewFrames(pathData) {
  const startTimestamp = pathData.frames[0]?.timestamp ?? 0;

  return pathData.frames.map((frame) => {
    const matrix = matrixFromRows(frame.transformMatrix);
    const modelPosition = new THREE.Vector3();
    const rotation = new THREE.Matrix4();
    modelPosition.setFromMatrixPosition(matrix);
    rotation.extractRotation(matrix);

    return {
      timestamp: frame.timestamp,
      elapsed: frame.timestamp - startTimestamp,
      position: scenePositionFromModel(modelPosition.toArray()),
      quaternion: new THREE.Quaternion().setFromRotationMatrix(rotation),
    };
  });
}

function setOverviewView() {
  stopPlayback({ keepCurrentView: true });
  disableFreeCamera();
  controls.enabled = true;
  camera.position.set(0, 1.3, 8);
  controls.target.set(0, 0.45, 0);
  controls.update();
}

function applyCaptureView(position, quaternion) {
  controls.enabled = false;
  camera.position.copy(position);
  camera.quaternion.copy(quaternion);
  camera.updateMatrixWorld();

  syncControlsTargetToCamera();
}

function setCameraFromPathFrame(frame) {
  const prepared = prepareViewFrames({ frames: [frame] })[0];
  applyCaptureView(prepared.position, prepared.quaternion);
}

function addTrajectory(pathData) {
  const points = pathData.frames.map((frame) => new THREE.Vector3(...frame.position));
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color: 0x4ade80,
    transparent: true,
    opacity: 0.95,
    depthTest: false,
  });
  const trajectory = new THREE.Line(geometry, material);
  trajectory.renderOrder = 10;
  sceneRoot.add(trajectory);

  const start = makePoseMarker(pathData.frames[0], 0x22c55e);
  const end = makePoseMarker(pathData.frames.at(-1), 0xef4444);
  sceneRoot.add(start, end);
}

function makePoseMarker(frame, color) {
  const matrix = matrixFromRows(frame.transformMatrix);
  const position = new THREE.Vector3();
  const rotation = new THREE.Matrix4();
  position.setFromMatrixPosition(matrix);
  rotation.extractRotation(matrix);
  const quaternion = new THREE.Quaternion().setFromRotationMatrix(rotation);
  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(quaternion);

  const group = new THREE.Group();
  group.position.copy(position);
  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(0.025, 16, 12),
    new THREE.MeshBasicMaterial({ color, depthTest: false }),
  );
  const arrow = new THREE.ArrowHelper(forward, new THREE.Vector3(0, 0, 0), 0.22, color);
  sphere.renderOrder = 11;
  arrow.renderOrder = 11;
  group.add(sphere, arrow);
  return group;
}

function sampleCapturePose(elapsedSeconds) {
  if (viewFrames.length === 0) {
    return null;
  }

  if (elapsedSeconds <= 0) {
    return viewFrames[0];
  }

  if (elapsedSeconds >= playbackDuration) {
    return viewFrames.at(-1);
  }

  let lo = 0;
  let hi = viewFrames.length - 1;
  while (lo < hi - 1) {
    const mid = Math.floor((lo + hi) / 2);
    if (viewFrames[mid].elapsed <= elapsedSeconds) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  const a = viewFrames[lo];
  const b = viewFrames[lo + 1];
  const span = b.elapsed - a.elapsed;
  const t = span > 0 ? (elapsedSeconds - a.elapsed) / span : 0;

  playbackPosition.copy(a.position).lerp(b.position, t);
  playbackQuaternion.copy(a.quaternion).slerp(b.quaternion, t);

  return {
    position: playbackPosition,
    quaternion: playbackQuaternion,
  };
}

function setPlaybackElapsed(elapsedSeconds) {
  const pose = sampleCapturePose(elapsedSeconds);
  if (!pose) {
    return;
  }
  applyCaptureView(pose.position, pose.quaternion);
  updatePlaybackLabel(elapsedSeconds);
}

function startPlayback() {
  if (viewFrames.length === 0) {
    return;
  }
  disableFreeCamera();
  const startAt = playbackPausedAt >= playbackDuration ? 0 : playbackPausedAt;
  playbackPausedAt = startAt;
  playbackStartedAt = performance.now() / 1000 - startAt;
  playbackActive = true;
  trackCameraButton.textContent = "Pause Capture";
  setPlaybackElapsed(startAt);
}

function stopPlayback({ keepCurrentView = false } = {}) {
  if (playbackActive) {
    playbackPausedAt = Math.min(performance.now() / 1000 - playbackStartedAt, playbackDuration);
  }
  playbackActive = false;
  trackCameraButton.textContent = playbackPausedAt >= playbackDuration ? "Replay Capture" : "Play Capture";
  if (!keepCurrentView && viewFrames.length > 0) {
    setPlaybackElapsed(playbackPausedAt);
  }
}

function updatePlayback() {
  if (!playbackActive) {
    return;
  }

  const elapsedSeconds = performance.now() / 1000 - playbackStartedAt;
  if (elapsedSeconds >= playbackDuration) {
    playbackPausedAt = playbackDuration;
    setPlaybackElapsed(playbackDuration);
    stopPlayback({ keepCurrentView: true });
    return;
  }

  playbackPausedAt = elapsedSeconds;
  setPlaybackElapsed(elapsedSeconds);
}

function enableFreeCamera() {
  freeCameraActive = true;
  controls.enabled = false;
  syncControlsTargetToCamera();
}

function disableFreeCamera() {
  freeCameraActive = false;
  pointerDragActive = false;
  keyState.clear();
}

function rotateFreeCamera(deltaYaw, deltaPitch) {
  if (deltaYaw !== 0) {
    yawQuaternion.setFromAxisAngle(worldUp, deltaYaw);
    camera.quaternion.premultiply(yawQuaternion);
  }
  if (deltaPitch !== 0) {
    cameraRight.set(1, 0, 0).applyQuaternion(camera.quaternion).normalize();
    pitchQuaternion.setFromAxisAngle(cameraRight, deltaPitch);
    camera.quaternion.premultiply(pitchQuaternion);
  }
  camera.quaternion.normalize();
  camera.updateMatrixWorld();
  syncControlsTargetToCamera();
}

function updateFreeCamera(deltaSeconds) {
  if (!freeCameraActive) {
    return;
  }

  const lookDelta = freeCameraKeyLookSpeed * deltaSeconds;
  if (keyState.has("ArrowLeft")) {
    rotateFreeCamera(lookDelta, 0);
  }
  if (keyState.has("ArrowRight")) {
    rotateFreeCamera(-lookDelta, 0);
  }
  if (keyState.has("ArrowUp")) {
    rotateFreeCamera(0, lookDelta);
  }
  if (keyState.has("ArrowDown")) {
    rotateFreeCamera(0, -lookDelta);
  }

  moveVector.set(0, 0, 0);
  camera.getWorldDirection(cameraForward).normalize();
  cameraRight.set(1, 0, 0).applyQuaternion(camera.quaternion).normalize();
  cameraUp.set(0, 1, 0).applyQuaternion(camera.quaternion).normalize();

  if (keyState.has("KeyW")) {
    moveVector.add(cameraForward);
  }
  if (keyState.has("KeyS")) {
    moveVector.sub(cameraForward);
  }
  if (keyState.has("KeyD")) {
    moveVector.add(cameraRight);
  }
  if (keyState.has("KeyA")) {
    moveVector.sub(cameraRight);
  }
  if (keyState.has("Space") || keyState.has("KeyE")) {
    moveVector.add(cameraUp);
  }
  if (keyState.has("ControlLeft") || keyState.has("ControlRight") || keyState.has("KeyQ")) {
    moveVector.sub(cameraUp);
  }

  if (moveVector.lengthSq() > 0) {
    const speed = freeCameraMoveSpeed * (keyState.has("ShiftLeft") || keyState.has("ShiftRight")
      ? freeCameraFastMultiplier
      : 1);
    camera.position.addScaledVector(moveVector.normalize(), speed * deltaSeconds);
    camera.updateMatrixWorld();
    syncControlsTargetToCamera();
  }
}

async function loadCameraPath() {
  const response = await fetch(`/camera-paths/${datasetName}.json`);
  if (!response.ok) {
    throw new Error(`camera path request failed: ${response.status}`);
  }
  cameraPath = await response.json();
  viewFrames = prepareViewFrames(cameraPath);
  playbackDuration = viewFrames.at(-1).elapsed;
  addTrajectory(cameraPath);
  playbackPausedAt = 0;
  updatePlaybackLabel(0);
  setPlaybackElapsed(0);
}

async function updateLoadStatus() {
  try {
    await splats.initialized;
    const count = splats.packedSplats?.numSplats;
    status.textContent = count ? `${count.toLocaleString()} splats` : "ready";
  } catch (error) {
    console.error(error);
    status.textContent = "failed";
  }
}

updateLoadStatus();
loadCameraPath().catch((error) => {
  console.error(error);
});

trackCameraButton.addEventListener("click", () => {
  if (playbackActive) {
    stopPlayback({ keepCurrentView: true });
  } else {
    startPlayback();
  }
});

startViewButton.addEventListener("click", () => {
  stopPlayback({ keepCurrentView: true });
  if (viewFrames.length > 0) {
    playbackPausedAt = 0;
    setPlaybackElapsed(0);
    enableFreeCamera();
    trackCameraButton.textContent = "Play Capture";
  }
});

overviewButton.addEventListener("click", setOverviewView);

function resize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

window.addEventListener("resize", resize);
window.addEventListener("keydown", (event) => {
  if (!freeCameraActive) {
    return;
  }
  keyState.add(event.code);
  if (
    event.code.startsWith("Arrow") ||
    ["KeyW", "KeyA", "KeyS", "KeyD", "KeyE", "KeyQ", "Space"].includes(event.code)
  ) {
    event.preventDefault();
  }
});
window.addEventListener("keyup", (event) => {
  keyState.delete(event.code);
});
window.addEventListener("blur", () => {
  keyState.clear();
  pointerDragActive = false;
});
renderer.domElement.addEventListener("pointerdown", (event) => {
  if (!freeCameraActive || event.button !== 0) {
    return;
  }
  pointerDragActive = true;
  renderer.domElement.setPointerCapture(event.pointerId);
});
renderer.domElement.addEventListener("pointermove", (event) => {
  if (!freeCameraActive || !pointerDragActive) {
    return;
  }
  rotateFreeCamera(
    -event.movementX * freeCameraMouseSensitivity,
    -event.movementY * freeCameraMouseSensitivity,
  );
});
renderer.domElement.addEventListener("pointerup", (event) => {
  pointerDragActive = false;
  if (renderer.domElement.hasPointerCapture(event.pointerId)) {
    renderer.domElement.releasePointerCapture(event.pointerId);
  }
});
renderer.domElement.addEventListener("pointercancel", () => {
  pointerDragActive = false;
});

renderer.setAnimationLoop(() => {
  const deltaSeconds = Math.min(clock.getDelta(), 0.05);
  updatePlayback();
  updateFreeCamera(deltaSeconds);
  if (controls.enabled) {
    controls.update();
  }
  renderer.render(scene, camera);
});
