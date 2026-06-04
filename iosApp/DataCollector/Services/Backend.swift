import Foundation
import Supabase

/// Single shared Supabase client for the whole app.
/// Use `Backend.supabase` to reach Storage, the database, etc.
enum Backend {
    static let supabase = SupabaseClient(
        supabaseURL: SupabaseConfig.url,
        supabaseKey: SupabaseConfig.anonKey
    )
}
