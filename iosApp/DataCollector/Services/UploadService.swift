import Foundation
import Supabase

/// Plain-value snapshot of a recording, built on the main thread and handed to the
/// uploader so we never touch a SwiftData model off the main thread.
struct UploadPayload {
    let id: UUID
    let folderName: String
    let streams: [String]
    let bountyId: String?
    let durationMs: Int
    let sizeBytes: Int
    let gpsLat: Double?
    let gpsLon: Double?
    let gpsAccuracyM: Double?
}

enum UploadError: LocalizedError {
    case notSignedIn
    case server(step: String, status: Int, body: String)

    var errorDescription: String? {
        switch self {
        case .notSignedIn:
            return "You're not signed in."
        case let .server(step, status, body):
            return "\(step) failed (HTTP \(status)). \(body)"
        }
    }
}

/// Uploads a recording bundle in two memory-safe steps, reporting overall
/// progress (0...1) via `onProgress` (always called on the main thread):
///  1. Stream each file straight to Supabase Storage from disk (constant memory).
///  2. POST a small JSON to the `submit-recording` function to write the
///     `recordings` + `submissions` rows (collector_id derived from the JWT).
enum UploadService {
    static func upload(
        _ payload: UploadPayload,
        onProgress: @escaping (Double) -> Void = { _ in }
    ) async throws {
        guard let token = Backend.supabase.auth.currentSession?.accessToken else {
            throw UploadError.notSignedIn
        }
        let bucket = SupabaseConfig.recordingsBucket
        let folder = RecordingStore.folderURL(for: payload.folderName)

        func report(_ value: Double) {
            DispatchQueue.main.async { onProgress(min(max(value, 0), 1)) }
        }

        // Files that exist, with their sizes (for an overall byte-based progress).
        let files: [(url: URL, name: String, size: Int64)] = payload.streams.compactMap { name in
            let url = folder.appendingPathComponent(name)
            guard let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize else { return nil }
            return (url, name, Int64(size))
        }
        let total = max(1, files.reduce(0) { $0 + $1.size })

        // 1) Stream each file to Storage.
        var completed: Int64 = 0
        var uploaded: [String] = []
        for file in files {
            try await streamToStorage(
                fileURL: file.url,
                bucket: bucket,
                // Lowercase so the storage path matches the DB id (Postgres lowercases UUIDs).
                objectPath: "\(payload.id.uuidString.lowercased())/\(file.name)",
                mimeType: contentType(for: file.name),
                token: token,
                onBytesSent: { sent in
                    report(Double(completed + sent) / Double(total) * 0.98) // leave headroom for finalize
                }
            )
            completed += file.size
            uploaded.append(file.name)
            report(Double(completed) / Double(total) * 0.98)
        }

        // 2) Finalize (tiny payload).
        try await finalize(payload: payload, streams: uploaded, token: token)
        report(1.0)
    }

    /// Streams one file to Storage. `URLSession.upload(fromFile:)` reads from disk,
    /// so even a 100MB+ depth file never sits fully in memory.
    private static func streamToStorage(
        fileURL: URL, bucket: String, objectPath: String, mimeType: String, token: String,
        onBytesSent: @escaping (Int64) -> Void
    ) async throws {
        let url = SupabaseConfig.url
            .appendingPathComponent("storage/v1/object/\(bucket)/\(objectPath)")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.setValue(SupabaseConfig.anonKey, forHTTPHeaderField: "apikey")
        request.setValue(mimeType, forHTTPHeaderField: "Content-Type")
        request.setValue("true", forHTTPHeaderField: "x-upsert")   // overwrite on retry

        let delegate = UploadProgressDelegate { sent, _ in onBytesSent(sent) }
        let (data, response) = try await URLSession.shared.upload(
            for: request, fromFile: fileURL, delegate: delegate
        )
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(status) else {
            throw UploadError.server(step: "Storage upload", status: status,
                                     body: String(data: data, encoding: .utf8) ?? "")
        }
    }

    private static func finalize(payload: UploadPayload, streams: [String], token: String) async throws {
        let endpoint = SupabaseConfig.url.appendingPathComponent("functions/v1/submit-recording")
        let key = payload.id.uuidString.lowercased() // match the storage path + DB id
        var body: [String: Any] = [
            "recording_id": key,
            "device_model": deviceIdentifier(),
            "duration_ms": payload.durationMs,
            "size_bytes": payload.sizeBytes,
            "storage_path": "\(key)/",
            "streams": streams,
        ]
        if let taskId = payload.bountyId { body["task_id"] = taskId }
        if let lat = payload.gpsLat { body["gps_lat"] = lat }
        if let lon = payload.gpsLon { body["gps_lon"] = lon }
        if let acc = payload.gpsAccuracyM { body["gps_accuracy_m"] = acc }

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.setValue(SupabaseConfig.anonKey, forHTTPHeaderField: "apikey")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(status) else {
            throw UploadError.server(step: "Finalize", status: status,
                                     body: String(data: data, encoding: .utf8) ?? "")
        }
    }

    /// Uploads a bundle as a task's reference example to `references/<taskId>/` in
    /// the recordings bucket, and returns that storage path (for tasks.reference_path).
    static func uploadReference(
        folderName: String,
        streams: [String],
        taskId: String,
        onProgress: @escaping (Double) -> Void = { _ in }
    ) async throws -> String {
        guard let token = Backend.supabase.auth.currentSession?.accessToken else {
            throw UploadError.notSignedIn
        }
        let bucket = SupabaseConfig.recordingsBucket
        let folder = RecordingStore.folderURL(for: folderName)
        let prefix = "references/\(taskId.lowercased())"

        func report(_ v: Double) { DispatchQueue.main.async { onProgress(min(max(v, 0), 1)) } }

        let files: [(url: URL, name: String, size: Int64)] = streams.compactMap { name in
            let url = folder.appendingPathComponent(name)
            guard let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize else { return nil }
            return (url, name, Int64(size))
        }
        let total = max(1, files.reduce(0) { $0 + $1.size })
        var completed: Int64 = 0
        for file in files {
            try await streamToStorage(
                fileURL: file.url, bucket: bucket,
                objectPath: "\(prefix)/\(file.name)",
                mimeType: contentType(for: file.name), token: token,
                onBytesSent: { sent in report(Double(completed + sent) / Double(total)) }
            )
            completed += file.size
            report(Double(completed) / Double(total))
        }
        return "\(prefix)/"
    }

    private static func contentType(for filename: String) -> String {
        if filename.hasSuffix(".mp4")   { return "video/mp4" }
        if filename.hasSuffix(".mov")   { return "video/quicktime" }
        if filename.hasSuffix(".json")  { return "application/json" }
        if filename.hasSuffix(".jsonl") { return "application/x-ndjson" }
        if filename.hasSuffix(".bin")   { return "application/octet-stream" }
        return "application/octet-stream"
    }

    /// Real hardware identifier, e.g. "iPhone15,2" (UIDevice.model is only "iPhone").
    private static func deviceIdentifier() -> String {
        var info = utsname()
        uname(&info)
        return Mirror(reflecting: info.machine).children.reduce(into: "") { result, element in
            if let code = element.value as? Int8, code != 0 {
                result.append(Character(UnicodeScalar(UInt8(code))))
            }
        }
    }
}

/// Reports per-task upload progress (bytes sent / expected).
private final class UploadProgressDelegate: NSObject, URLSessionTaskDelegate {
    let handler: (Int64, Int64) -> Void
    init(_ handler: @escaping (Int64, Int64) -> Void) { self.handler = handler }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    didSendBodyData bytesSent: Int64,
                    totalBytesSent: Int64,
                    totalBytesExpectedToSend: Int64) {
        handler(totalBytesSent, totalBytesExpectedToSend)
    }
}
