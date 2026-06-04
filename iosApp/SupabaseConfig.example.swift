// TEMPLATE — not compiled (lives outside the DataCollector/ source folder).
//
// On a fresh clone, copy this file to:
//     DataCollector/Config/SupabaseConfig.swift
// and fill in your project's values. The real file is git-ignored so keys
// never get committed.

import Foundation

enum SupabaseConfig {
    static let url = URL(string: "https://uiminnwdvqjkrrjoyylx.supabase.co/")!
    static let anonKey = "YOUR_SUPABASE_ANON_KEY"
    static let publishableKey = "YOUR_SUPABASE_PUBLISHABLE_KEY"
    static let recordingsBucket = "recordings"

    // Gemini Live (real-time coaching). Leave "" to disable. ⚠️ raw key in-app is
    // demo-only — use ephemeral tokens from a backend for production.
    static let geminiAPIKey = "YOUR_GEMINI_API_KEY"
    static let geminiLiveModel = "models/gemini-3.1-flash-live-preview"
    static let geminiVisionModel = "models/gemini-3.5-flash"
}
