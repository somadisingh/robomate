import SwiftUI
import AVKit

/// Plays a task's reference example video. Fetches a signed URL (private bucket),
/// usable by both labs (review) and collectors (see what to record).
///
/// The player is created ONCE and held in state (not inline), and paused when the
/// view goes off-screen — otherwise it kept decoding in the background and fought
/// the camera/GPU when navigating to a camera screen (record / AI coach).
struct ReferenceVideoView: View {
    let taskId: String
    @State private var player: AVPlayer?
    @State private var loading = true

    var body: some View {
        Group {
            if loading {
                HStack(spacing: 8) {
                    ProgressView()
                    Text("Loading reference…").font(.caption).foregroundStyle(.secondary)
                }
            } else if let player {
                VideoPlayer(player: player)
                    .frame(height: 200)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            } else {
                Text("No reference video yet.").font(.caption).foregroundStyle(.secondary)
            }
        }
        .task { await load() }
        .onDisappear { player?.pause() }   // stop decoding when navigating away
    }

    private func load() async {
        guard player == nil else { return }
        let signed = try? await LabTasksService.signedReferenceURL(taskId: taskId)
        await MainActor.run {
            if let signed { player = AVPlayer(url: signed) }
            loading = false
        }
    }
}
