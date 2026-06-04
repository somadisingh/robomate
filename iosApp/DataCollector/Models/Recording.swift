import Foundation
import SwiftData

/// One captured recording bundle, tracked locally so the app knows what's on the
/// device and whether it has been uploaded yet. The actual media/sensor files live
/// on disk under Documents/recordings/<folderName>/.
@Model
final class Recording {
    @Attribute(.unique) var id: UUID
    var bountyId: String?
    var createdAt: Date
    var durationMs: Int
    var sizeBytes: Int
    var gpsLat: Double?
    var gpsLon: Double?
    var gpsAccuracyM: Double?
    /// Name of the bundle folder under Documents/recordings/.
    var folderName: String
    /// Filenames in the bundle, e.g. ["video.mov", "imu.jsonl", "metadata.json"].
    var streams: [String]
    private var statusRaw: String

    var status: RecordingStatus {
        get { RecordingStatus(rawValue: statusRaw) ?? .local }
        set { statusRaw = newValue.rawValue }
    }

    init(
        id: UUID = UUID(),
        bountyId: String? = nil,
        createdAt: Date = .now,
        durationMs: Int = 0,
        sizeBytes: Int = 0,
        gpsLat: Double? = nil,
        gpsLon: Double? = nil,
        gpsAccuracyM: Double? = nil,
        folderName: String,
        streams: [String] = [],
        status: RecordingStatus = .local
    ) {
        self.id = id
        self.bountyId = bountyId
        self.createdAt = createdAt
        self.durationMs = durationMs
        self.sizeBytes = sizeBytes
        self.gpsLat = gpsLat
        self.gpsLon = gpsLon
        self.gpsAccuracyM = gpsAccuracyM
        self.folderName = folderName
        self.streams = streams
        self.statusRaw = status.rawValue
    }
}
