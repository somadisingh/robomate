import Foundation

/// Lifecycle of a recording bundle: captured locally → uploading → uploaded.
enum RecordingStatus: String, Codable, CaseIterable {
    case local
    case uploading
    case uploaded
    case failed

    var label: String {
        switch self {
        case .local:     return "On device"
        case .uploading: return "Uploading…"
        case .uploaded:  return "Uploaded"
        case .failed:    return "Failed"
        }
    }
}
