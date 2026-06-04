import Foundation
import Supabase

/// Fetches jobs (the `tasks` table) for contributors to browse.
enum JobsService {
    static func fetchOpenJobs() async throws -> [Job] {
        let jobs: [Job] = try await Backend.supabase
            .from("tasks")
            .select()
            .eq("status", value: "open")
            .order("created_at", ascending: false)
            .execute()
            .value
        return jobs
    }
}
