"""PLY → SPZ conversion via the sparkjs Node CLI baked into the Modal image."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


SPARK_CONVERTER_ROOT = Path("/opt/spark-converter")
SPARK_NODE_BINARY = SPARK_CONVERTER_ROOT / "node_modules" / "node" / "bin" / "node"
SPARK_CONVERTER_SCRIPT = SPARK_CONVERTER_ROOT / "convert-to-spz.mjs"

SPARK_CONVERTER_JS = r"""
import fs from "node:fs/promises";
import path from "node:path";
import { transcodeSpz } from "@sparkjsdev/spark";

const [inputPath, outputPath] = process.argv.slice(2);
if (!inputPath || !outputPath) {
  console.error("usage: node convert-to-spz.mjs input.ply output.spz");
  process.exit(2);
}

const input = path.resolve(inputPath);
const output = path.resolve(outputPath);
const fileBytes = new Uint8Array(await fs.readFile(input));
const result = await transcodeSpz({
  inputs: [
    {
      fileBytes,
      pathOrUrl: input,
      transform: {
        translate: [0, 0, 0],
        quaternion: [0, 0, 0, 1],
        scale: 1,
      },
    },
  ],
});

await fs.mkdir(path.dirname(output), { recursive: true });
await fs.writeFile(output, result.fileBytes);
console.log(`wrote ${output} (${result.fileBytes.length} bytes)`);
if (result.clippedCount) {
  console.log(`clipped ${result.clippedCount} splats`);
}
"""


def ply_to_spz(ply_path: Path, spz_path: Path) -> None:
    """Run the sparkjs transcoder. Raises ``CalledProcessError`` on failure."""
    cmd = [
        str(SPARK_NODE_BINARY),
        str(SPARK_CONVERTER_SCRIPT),
        str(ply_path),
        str(spz_path),
    ]
    print("$", shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True)
