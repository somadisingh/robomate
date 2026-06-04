import Foundation
import Supabase

/// A task owned by the current lab (the `tasks` table).
struct LabTask: Identifiable, Decodable, Hashable {
    let id: UUID
    let title: String
    let description: String?
    let bountyAmount: Double?
    let quantityNeeded: Int?
    let quantityFilled: Int?
    let status: String?

    enum CodingKeys: String, CodingKey {
        case id, title, description, status
        case bountyAmount = "bounty_amount"
        case quantityNeeded = "quantity_needed"
        case quantityFilled = "quantity_filled"
    }
}

enum LabTasksService {
    static func myTasks() async throws -> [LabTask] {
        guard let uid = Backend.supabase.auth.currentUser?.id else { return [] }
        return try await Backend.supabase
            .from("tasks")
            .select("id,title,description,bounty_amount,quantity_needed,quantity_filled,status")
            .eq("lab_id", value: uid.uuidString)
            .order("created_at", ascending: false)
            .execute()
            .value
    }

    static func createTask(
        title: String,
        description: String,
        capabilities: [String],
        dataType: String,
        bountyAmount: Double,
        quantityNeeded: Int
    ) async throws {
        guard let uid = Backend.supabase.auth.currentUser?.id else { throw UploadError.notSignedIn }
        struct NewTask: Encodable {
            let lab_id: String
            let title: String
            let description: String
            let data_type: String
            let required_capabilities: [String]
            let bounty_amount: Double
            let quantity_needed: Int
        }
        try await Backend.supabase
            .from("tasks")
            .insert(NewTask(
                lab_id: uid.uuidString,
                title: title,
                description: description,
                data_type: dataType,
                required_capabilities: capabilities,
                bounty_amount: bountyAmount,
                quantity_needed: quantityNeeded
            ))
            .execute()
    }

    /// Set the task's reference storage path (after recording the example in-app).
    static func setReferencePath(taskId: String, path: String) async throws {
        struct Patch: Encodable { let reference_path: String }
        try await Backend.supabase
            .from("tasks")
            .update(Patch(reference_path: path))
            .eq("id", value: taskId)
            .execute()
    }

    /// Current reference path for a task, if one has been recorded.
    static func referencePath(taskId: String) async throws -> String? {
        struct Row: Decodable { let reference_path: String? }
        let rows: [Row] = try await Backend.supabase
            .from("tasks")
            .select("reference_path")
            .eq("id", value: taskId)
            .execute()
            .value
        return rows.first?.reference_path ?? nil
    }

    /// Reference path + AI-generated brief for a task, in one query.
    struct ReferenceInfo: Decodable { let reference_path: String?; let reference_brief: String? }
    static func referenceInfo(taskId: String) async throws -> ReferenceInfo? {
        let rows: [ReferenceInfo] = try await Backend.supabase
            .from("tasks")
            .select("reference_path,reference_brief")
            .eq("id", value: taskId)
            .execute()
            .value
        return rows.first
    }

    /// Store the AI-generated text brief describing the reference.
    static func setReferenceBrief(taskId: String, brief: String) async throws {
        struct Patch: Encodable { let reference_brief: String }
        try await Backend.supabase
            .from("tasks")
            .update(Patch(reference_brief: brief))
            .eq("id", value: taskId)
            .execute()
    }

    /// Signed URL to a task's reference video (private bucket), if a reference exists.
    static func signedReferenceURL(taskId: String) async throws -> URL? {
        guard let path = try await referencePath(taskId: taskId), !path.isEmpty else { return nil }
        let videoPath = (path.hasSuffix("/") ? path : path + "/") + "video.mp4"
        return try await Backend.supabase.storage
            .from(SupabaseConfig.recordingsBucket)
            .createSignedURL(path: videoPath, expiresIn: 3600)
    }
}
