import SwiftUI
import SwiftData

/// App entry point. Sets up the SwiftData store that tracks recordings.
@main
struct DataCollectorApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .modelContainer(for: Recording.self)
    }
}
