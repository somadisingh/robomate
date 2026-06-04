import Foundation

/// File URLs that make up one recording bundle on disk.
struct RecordingBundle {
    let id: UUID
    let folderURL: URL
    let videoURL: URL
    let imuURL: URL
    let posesURL: URL
    let intrinsicsURL: URL
    let depthURL: URL
    let transcriptURL: URL
    let metadataURL: URL

    var streamFilenames: [String] {
        [videoURL, imuURL, posesURL, intrinsicsURL, depthURL, transcriptURL, metadataURL]
            .map(\.lastPathComponent)
    }
}

/// Owns the on-disk layout: Documents/recordings/<id>/{video.mov, imu.jsonl, metadata.json}.
enum RecordingStore {
    static var recordingsDir: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("recordings", isDirectory: true)
    }

    static func folderURL(for folderName: String) -> URL {
        recordingsDir.appendingPathComponent(folderName, isDirectory: true)
    }

    /// Creates an empty bundle folder and returns the file URLs to write into.
    static func makeBundle(id: UUID = UUID()) throws -> RecordingBundle {
        let folder = recordingsDir.appendingPathComponent(id.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return RecordingBundle(
            id: id,
            folderURL: folder,
            videoURL: folder.appendingPathComponent("video.mp4"),
            imuURL: folder.appendingPathComponent("imu.jsonl"),
            posesURL: folder.appendingPathComponent("poses.jsonl"),
            intrinsicsURL: folder.appendingPathComponent("intrinsics.json"),
            depthURL: folder.appendingPathComponent("depth.bin"),
            transcriptURL: folder.appendingPathComponent("transcript.json"),
            metadataURL: folder.appendingPathComponent("metadata.json")
        )
    }

    static func writeMetadata(_ metadata: RecordingMetadata, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(metadata).write(to: url, options: .atomic)
    }

    /// Write the speech transcript: full text + timestamped segments. `status` is a
    /// temporary debug field (so an empty transcript explains itself).
    static func writeTranscript(_ segments: [WhisperTranscriber.Segment],
                                status: String, to url: URL) throws {
        struct Transcript: Codable {
            let text: String
            let segments: [WhisperTranscriber.Segment]
            let status: String
        }
        let text = segments.map(\.text).joined(separator: " ")
        let transcript = Transcript(text: text, segments: segments, status: status)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(transcript).write(to: url, options: .atomic)
    }

    /// Total bytes used by a bundle folder.
    static func directorySize(_ url: URL) -> Int {
        guard let enumerator = FileManager.default.enumerator(
            at: url, includingPropertiesForKeys: [.fileSizeKey]
        ) else { return 0 }
        var total = 0
        for case let fileURL as URL in enumerator {
            total += (try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        }
        return total
    }

    /// Keep big media out of the user's iCloud backup.
    static func excludeFromBackup(_ url: URL) {
        var url = url
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? url.setResourceValues(values)
    }

    static func deleteBundle(folderName: String) {
        try? FileManager.default.removeItem(at: folderURL(for: folderName))
    }
}
