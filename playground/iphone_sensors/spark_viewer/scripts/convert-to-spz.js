import fs from "node:fs/promises";
import path from "node:path";
import { transcodeSpz } from "@sparkjsdev/spark";

const [inputPath, outputPath] = process.argv.slice(2);

if (!inputPath || !outputPath) {
  console.error("usage: node scripts/convert-to-spz.js input.ply output.spz");
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
