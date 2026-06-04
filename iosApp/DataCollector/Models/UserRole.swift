import Foundation

/// The two account types, mirroring the web (`profiles.role`).
enum UserRole: String, CaseIterable, Identifiable {
    case collector
    case lab

    var id: String { rawValue }
    var label: String { self == .lab ? "Lab" : "Collector" }
}
