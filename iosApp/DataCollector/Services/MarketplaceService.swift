import Foundation
import Supabase

/// Read-side of the marketplace: review status of submissions + earnings.

// MARK: - Earnings

struct Earning: Decodable {
    let amount: Double
    let status: String   // "pending" | "approved"
}

enum EarningsService {
    static func fetch() async throws -> [Earning] {
        guard let uid = Backend.supabase.auth.currentUser?.id else { return [] }
        return try await Backend.supabase
            .from("earnings")
            .select("amount,status")
            .eq("collector_id", value: uid.uuidString)
            .execute()
            .value
    }
}

// MARK: - AI quality score (read from the recordings table)

/// A recording's quality result, written by the scoring microservice.
struct RecordingScore {
    let status: String            // recordings.status (e.g. "uploaded")
    let isScoring: Bool           // true while the model hasn't returned yet
    let success: Bool?
    let successReasoning: String?
    let score: Double?            // 0...10
    let scoreReasoning: String?
}

private struct RecordingRow: Decodable {
    let id: String
    let status: String
    let is_scoring: Bool?
    let success: Bool?
    let success_reasoning: String?
    let score: Double?
    let score_reasoning: String?
}

enum ScoringService {
    /// Map of recordingId (lowercased) → status + score, for one task's recordings.
    static func scores(taskId: String) async throws -> [String: RecordingScore] {
        let rows: [RecordingRow] = try await Backend.supabase
            .from("recordings")
            .select("id,status,is_scoring,success,success_reasoning,score,score_reasoning")
            .eq("bounty_id", value: taskId)
            .execute()
            .value
        var map: [String: RecordingScore] = [:]
        for row in rows {
            // recordings.id == the recording UUID (stored lowercased).
            map[row.id.lowercased()] = RecordingScore(
                status: row.status,
                isScoring: row.is_scoring ?? false,
                success: row.success,
                successReasoning: row.success_reasoning,
                score: row.score,
                scoreReasoning: row.score_reasoning
            )
        }
        return map
    }
}
