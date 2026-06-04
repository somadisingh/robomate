import fs from "node:fs/promises";
import path from "node:path";

const [transformsPath, dataparserPath, outputPath] = process.argv.slice(2);

if (!transformsPath || !dataparserPath || !outputPath) {
  console.error(
    "usage: node scripts/export-camera-path.js transforms.json dataparser_transforms.json output.json",
  );
  process.exit(2);
}

function identity4() {
  return [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
}

function multiply4(a, b) {
  const out = identity4().map((row) => row.map(() => 0));
  for (let r = 0; r < 4; r += 1) {
    for (let c = 0; c < 4; c += 1) {
      for (let k = 0; k < 4; k += 1) {
        out[r][c] += a[r][k] * b[k][c];
      }
    }
  }
  return out;
}

const transforms = JSON.parse(await fs.readFile(transformsPath, "utf8"));
const dataparser = JSON.parse(await fs.readFile(dataparserPath, "utf8"));

const nsTransform = identity4();
for (let r = 0; r < 3; r += 1) {
  for (let c = 0; c < 4; c += 1) {
    nsTransform[r][c] = dataparser.transform[r][c];
  }
}

const scale = dataparser.scale;
const frames = transforms.frames.map((frame) => {
  const matrix = multiply4(nsTransform, frame.transform_matrix);
  matrix[0][3] *= scale;
  matrix[1][3] *= scale;
  matrix[2][3] *= scale;
  return {
    videoFrameIndex: frame.video_frame_index,
    timestamp: frame.timestamp,
    position: [matrix[0][3], matrix[1][3], matrix[2][3]],
    transformMatrix: matrix,
  };
});

const positions = frames.map((frame) => frame.position);
const bounds = {
  min: [0, 1, 2].map((axis) => Math.min(...positions.map((p) => p[axis]))),
  max: [0, 1, 2].map((axis) => Math.max(...positions.map((p) => p[axis]))),
};

const payload = {
  source: {
    transforms: path.basename(transformsPath),
    dataparserTransforms: path.basename(dataparserPath),
  },
  coordinateSpace: "nerfstudio_export",
  count: frames.length,
  bounds,
  frames,
};

const outputFile = path.resolve(outputPath);
await fs.mkdir(path.dirname(outputFile), { recursive: true });
await fs.writeFile(outputFile, `${JSON.stringify(payload, null, 2)}\n`);
console.log(`wrote ${outputFile} (${frames.length} camera poses)`);
