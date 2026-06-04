import SwiftUI
import AVFoundation
import AVKit
import UIKit

/// Generates a still-frame thumbnail from a local video file.
enum ThumbnailGenerator {
    static func thumbnail(for url: URL) async -> UIImage? {
        let asset = AVURLAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 240, height: 240)
        let time = CMTime(seconds: 0.1, preferredTimescale: 600)
        guard let cgImage = try? await generator.image(at: time).image else { return nil }
        return UIImage(cgImage: cgImage)
    }
}

/// A square video thumbnail with a play overlay. Loads asynchronously.
struct VideoThumbnail: View {
    let url: URL
    @State private var image: UIImage?

    var body: some View {
        ZStack {
            if let image {
                Image(uiImage: image).resizable().scaledToFill()
            } else {
                Rectangle().fill(Color.gray.opacity(0.15))
                Image(systemName: "video").foregroundStyle(.secondary)
            }
            Image(systemName: "play.circle.fill")
                .font(.title3).foregroundStyle(.white).shadow(radius: 2)
                .opacity(image == nil ? 0 : 0.9)
        }
        .frame(width: 56, height: 56)
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .task(id: url) { image = await ThumbnailGenerator.thumbnail(for: url) }
    }
}

/// Wrapper so a video URL can drive `.sheet(item:)`.
struct PlayableVideo: Identifiable {
    let id = UUID()
    let url: URL
}

/// Full-screen player for a recorded clip.
struct VideoPlayerSheet: View {
    let url: URL
    var body: some View {
        VideoPlayer(player: AVPlayer(url: url)).ignoresSafeArea()
    }
}

/// A no-chrome, auto-looping video view (raw `AVPlayerLayer`). Has no transport
/// controls, so overlaid buttons (e.g. Save/Discard) receive taps cleanly — and
/// it avoids AVKit's Live Text analysis that spams `VKCImageAnalyzer` errors.
struct LoopingPlayerView: UIViewRepresentable {
    let url: URL

    func makeUIView(context: Context) -> LoopingPlayerUIView { LoopingPlayerUIView(url: url) }
    func updateUIView(_ uiView: LoopingPlayerUIView, context: Context) {}
    static func dismantleUIView(_ uiView: LoopingPlayerUIView, coordinator: ()) { uiView.cleanup() }
}

final class LoopingPlayerUIView: UIView {
    private let player = AVQueuePlayer()
    private var looper: AVPlayerLooper?

    override class var layerClass: AnyClass { AVPlayerLayer.self }
    private var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }

    init(url: URL) {
        super.init(frame: .zero)
        looper = AVPlayerLooper(player: player, templateItem: AVPlayerItem(url: url))
        playerLayer.player = player
        playerLayer.videoGravity = .resizeAspect
        player.play()
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    func cleanup() { player.pause(); looper = nil; playerLayer.player = nil }
}

extension Recording {
    /// On-disk URL of the recorded video file, if present.
    var videoURL: URL? {
        guard let name = streams.first(where: { $0.hasSuffix(".mov") || $0.hasSuffix(".mp4") }) else {
            return nil
        }
        return RecordingStore.folderURL(for: folderName).appendingPathComponent(name)
    }
}

extension RecordingStatus {
    var color: Color {
        switch self {
        case .local:     return .blue
        case .uploading: return .orange
        case .uploaded:  return .green
        case .failed:    return .red
        }
    }
}

/// Small display formatters shared by recording lists.
enum RecordingFormat {
    static func duration(_ ms: Int) -> String {
        let s = ms / 1000
        return String(format: "%d:%02d", s / 60, s % 60)
    }
    static func size(_ bytes: Int) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }
}
