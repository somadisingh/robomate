import SwiftUI
import SwiftData

/// Root view. Shows sign-in until authenticated, then routes by role:
/// Lab → tasks dashboard; Collector → jobs.
struct ContentView: View {
    @StateObject private var auth = AuthManager()

    var body: some View {
        Group {
            if auth.isAuthenticated {
                if auth.role == .lab {
                    LabHomeView()
                } else {
                    collectorTabs
                }
            } else {
                AuthView()
            }
        }
        .environmentObject(auth)
        .preferredColorScheme(.dark)
    }

    private var collectorTabs: some View {
        TabView {
            JobsView()
                .tabItem { Label("Jobs", systemImage: "list.bullet.rectangle") }
            ProfileView()
                .tabItem { Label("Profile", systemImage: "person.crop.circle") }
        }
        .tint(.appAccent)
    }
}

#Preview {
    ContentView()
        .modelContainer(for: Recording.self, inMemory: true)
}
