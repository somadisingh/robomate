import Foundation
import AVFoundation
import CoreGraphics
import simd

/// Per-task target ranges for the live coaching HUD, derived from a lab's
/// reference recording. Uploaded next to the reference as `coaching.json` and
/// fetched by collectors so the HUD checks against *that task's* example
/// instead of generic defaults. All fields optional — a missing field means
/// "no task-specific target; use the generic default".
struct CoachingProfile: Codable, Equatable {
    var luminanceMin: Double? = nil   // 0...1 average brightness band
    var luminanceMax: Double? = nil
    var motionMax: Double? = nil      // deg/sec camera-rotation ceiling
    var distanceMin: Double? = nil    // metres to subject (LiDAR)
    var distanceMax: Double? = nil
    var source: String? = nil         // "reference"
}

/// Computes a `CoachingProfile` from a local reference bundle folder. Runs the
/// heavy file parsing off the main actor.
enum CoachingProfileBuilder {
    static func build(folderURL: URL) async -> CoachingProfile {
        var profile = CoachingProfile(source: "reference")

        // Pose + depth parsing are CPU/IO-bound — keep them off the main actor.
        let parsed = await Task.detached(priority: .utility) { () -> (Double?, (Double, Double)?) in
            let motion = motionCeiling(posesURL: folderURL.appendingPathComponent("poses.jsonl"))
            let distance = distanceBand(folderURL: folderURL)
            return (motion, distance)
        }.value
        profile.motionMax = parsed.0
        if let (lo, hi) = parsed.1 { profile.distanceMin = lo; profile.distanceMax = hi }

        if let (lo, hi) = await luminanceBand(videoURL: folderURL.appendingPathComponent("video.mp4")) {
            profile.luminanceMin = lo
            profile.luminanceMax = hi
        }
        return profile
    }

    // MARK: - Motion (poses.jsonl)

    /// Mirrors the live EMA in `Recorder.updateLiveMetrics`: smooth the per-frame
    /// rotational speed, then take the 90th percentile with headroom so a steady
    /// collector clears the bar comfortably.
    private static func motionCeiling(posesURL: URL) -> Double? {
        guard let content = try? String(contentsOf: posesURL, encoding: .utf8) else { return nil }
        var prevVec: simd_float4?
        var prevT: Double?
        var ema = 0.0
        var series: [Double] = []

        for line in content.split(separator: "\n") {
            guard let data = line.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { continue }
            func num(_ key: String) -> Double? { (obj[key] as? NSNumber)?.doubleValue }
            guard let t = num("t"),
                  let qw = num("qw"), let qx = num("qx"), let qy = num("qy"), let qz = num("qz")
            else { continue }

            let vec = simd_float4(Float(qx), Float(qy), Float(qz), Float(qw))
            if let pv = prevVec, let pt = prevT, t > pt {
                let dot = Double(min(1, abs(simd_dot(pv, vec))))
                let deg = 2.0 * acos(dot) * 180.0 / Double.pi
                let dt = t - pt
                ema = ema * 0.8 + (deg / dt) * 0.2
                series.append(ema)
            }
            prevVec = vec
            prevT = t
        }
        guard !series.isEmpty else { return nil }
        return max(percentile(series, 0.9) * 1.4, 12)   // headroom + sane floor
    }

    // MARK: - Distance (depth.bin center pixel)

    /// Reads the center depth of each LiDAR frame (seeking, so we never load the
    /// whole multi-MB file) and returns a lenient [p10, p90] band in metres.
    private static func distanceBand(folderURL: URL) -> (Double, Double)? {
        let metaURL = folderURL.appendingPathComponent("metadata.json")
        guard let metaData = try? Data(contentsOf: metaURL),
              let meta = try? JSONSerialization.jsonObject(with: metaData) as? [String: Any],
              let depth = meta["depth"] as? [String: Any],
              let w = (depth["width"] as? NSNumber)?.intValue,
              let h = (depth["height"] as? NSNumber)?.intValue,
              w > 0, h > 0
        else { return nil }

        let depthURL = folderURL.appendingPathComponent("depth.bin")
        guard let handle = try? FileHandle(forReadingFrom: depthURL) else { return nil }
        defer { try? handle.close() }

        let frameBytes = 8 + w * h * 4                       // Float64 ts + W*H Float32
        let centerOffset = 8 + ((h / 2) * w + w / 2) * 4
        var values: [Double] = []
        var frame = 0
        while frame < 100_000 {                              // safety bound
            let offset = UInt64(frame * frameBytes + centerOffset)
            try? handle.seek(toOffset: offset)
            guard let chunk = try? handle.read(upToCount: 4), chunk.count == 4 else { break }
            let v = chunk.withUnsafeBytes { $0.load(as: Float32.self) }
            if v.isFinite && v > 0 { values.append(Double(v)) }
            frame += 1
        }
        guard values.count >= 3 else { return nil }
        let lo = percentile(values, 0.1) * 0.7
        let hi = percentile(values, 0.9) * 1.3
        return (max(0.1, lo), hi)
    }

    // MARK: - Lighting (sampled video frames)

    private static func luminanceBand(videoURL: URL) async -> (Double, Double)? {
        let asset = AVURLAsset(url: videoURL)
        guard let duration = try? await asset.load(.duration) else { return nil }
        let durSec = CMTimeGetSeconds(duration)
        guard durSec.isFinite, durSec > 0 else { return nil }

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 64, height: 64)

        var lums: [Double] = []
        for i in 0..<8 {
            let frac = (Double(i) + 0.5) / 8.0
            let time = CMTime(seconds: durSec * frac, preferredTimescale: 600)
            if let result = try? await generator.image(at: time),
               let l = averageLuminance(result.image) {
                lums.append(l)
            }
        }
        guard !lums.isEmpty else { return nil }
        return (max(0, (lums.min() ?? 0) - 0.12), min(1, (lums.max() ?? 1) + 0.12))
    }

    /// Whole-image average luma via a 1×1 downscale (Rec. 601 weights), 0...1.
    private static func averageLuminance(_ cg: CGImage) -> Double? {
        var pixel = [UInt8](repeating: 0, count: 4)
        let space = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &pixel, width: 1, height: 1, bitsPerComponent: 8,
                                  bytesPerRow: 4, space: space,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: 1, height: 1))
        let r = Double(pixel[0]), g = Double(pixel[1]), b = Double(pixel[2])
        return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    }

    // MARK: - Helpers

    private static func percentile(_ values: [Double], _ p: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let idx = Int((Double(sorted.count - 1) * p).rounded())
        return sorted[min(max(idx, 0), sorted.count - 1)]
    }
}
