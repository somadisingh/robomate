# iPhone Sensors Playground

This folder processes the iPhone capture in `../data/iphone-data-1`.

Generated outputs go into `outputs/`:

- `depth_overlay.mp4`: LiDAR depth projected onto the video frame and rendered in display orientation.
- `fusion_panel.mp4`: video plus synchronized pose, depth, acceleration, and gyro traces.
- `depth_projection_samples.png`: representative RGB/depth/overlay samples.
- `sensor_fusion_dashboard.png`: static plots for trajectory, IMU, depth, and orientation.
- `world_point_cloud_map.png`: coarse world-frame LiDAR point map from depth plus camera poses.
- `world_point_cloud_sample.ply`: sampled fused point cloud for external 3D viewers.
- `fused_timeline.csv`: per-video-frame timestamp, depth, pose, and nearest IMU values.
- `index.html`: local report linking the generated videos and images.

Run:

```bash
uv run python process_capture.py
```

The script uses the encoded `1920x1440` camera orientation for projection because the depth map and intrinsics share that 4:3 camera frame. Rendered videos are rotated into the iPhone display orientation afterwards.

## Nerfstudio 3D Gaussian Splatting on Modal

Export the capture into Nerfstudio's native dataset format:

```bash
uv run python export_nerfstudio.py --overwrite
```

This writes `outputs/nerfstudio/iphone-data-1/` with:

- non-autorotated RGB frames in `images/`, matching `intrinsics.json`
- `transforms.json` with ARKit camera-to-world poses
- 16-bit millimetre LiDAR depth PNGs in `depths/`
- `sparse_pc.ply`, an RGB-coloured LiDAR point cloud for splat initialization

Upload the dataset to Modal and train/export a Nerfstudio `splatfacto` model:

```bash
uv run modal run modal_nerfstudio.py \
  --dataset-dir outputs/nerfstudio/iphone-data-1 \
  --max-num-iterations 7000
```

The Modal job uses a persistent volume named `copilot-hackathon-iphone-nerfstudio`. It runs `ns-train splatfacto` remotely on an L4/A10 fallback GPU, runs `ns-export gaussian-splat`, converts the exported PLY to Spark-compatible SPZ on Modal, writes `camera_path.json` in the exported Nerfstudio coordinate frame, and downloads the browser-ready assets into `spark_viewer/public/`.

## Spark SPZ Viewer

Convert the downloaded Nerfstudio Gaussian PLY for `iphone-data-3` to SPZ:

```bash
cd spark_viewer
npm install
npm run convert:iphone-data-3
npm run camera-path:iphone-data-3
```

Run the local Spark viewer:

```bash
npm run dev -- --port 5177
```

Open `http://127.0.0.1:5177/?dataset=iphone-data-3`. The viewer loads `public/splats/<dataset>.spz` with Spark's `SplatMesh`, starts at the first captured camera pose, overlays the captured camera trajectory from `public/camera-paths/<dataset>.json`, and can replay the original camera motion using the capture timestamps.

Click `Start View` to jump to the first recorded camera pose and enter free-camera mode. In free-camera mode, use `W/A/S/D` to move, drag the mouse or use arrow keys to rotate, `Space`/`E` and `Ctrl`/`Q` to move vertically, and `Shift` to move faster.
