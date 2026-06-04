import Foundation

/// The self-describing manifest written as `metadata.json` into every bundle.
/// Cloud agents read this to understand a bundle without decoding the video.
struct RecordingMetadata: Codable {
    let recordingId: String
    let bountyId: String?
    /// Wall-clock time the recording started (ISO 8601).
    let startedAt: String
    /// Monotonic clock (seconds) at start — lets the cloud align sensor streams to the video.
    let startMonotonic: Double
    let durationMs: Int
    let device: Device
    let video: Video
    let gps: GPS?
    let depth: Depth?
    let streams: [String]
    let schemaVersion: Int

    struct Device: Codable {
        let model: String
        let systemName: String
        let systemVersion: String
    }
    struct Video: Codable {
        let file: String
        let container: String
        let codec: String
    }
    struct GPS: Codable {
        let lat: Double
        let lon: Double
        let accuracyM: Double
    }
    struct Depth: Codable {
        let file: String
        let width: Int
        let height: Int
        let dtype: String
        /// How to parse depth.bin (it has no header).
        let layout: String
    }
}
