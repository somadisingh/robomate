import SwiftUI
import AVKit

/// Tap a recording to see it play back plus its AI quality score + reasoning.
/// Polls while the score is still being computed by the microservice.
struct RecordingDetailView: View {
    let recording: Recording
    let taskId: String

    @Environment(\.dismiss) private var dismiss
    @State private var score: RecordingScore?
    @State private var loading = true
    @State private var transcript: TranscriptFile?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    if let url = recording.videoURL {
                        VideoPlayer(player: AVPlayer(url: url))
                            .frame(height: 220)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                    scoreCard
                    transcriptCard
                    metaCard
                }
                .padding()
            }
            .background(Color.appBackground.ignoresSafeArea())
            .navigationTitle("Recording")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) { Button("Done") { dismiss() } }
            }
            .task { await load() }
            .task { await pollTranscript() }
        }
    }

    // MARK: - Score

    @ViewBuilder
    private var scoreCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Quality score").font(.headline)

            if recording.status != .uploaded {
                hint("Upload this recording to get it scored.")
            } else if let s = score, !s.isScoring, let value = s.score {
                let passed = s.success ?? (value >= 5)
                HStack(spacing: 12) {
                    Text("\(scoreText(value))/10")
                        .font(.system(size: 40, weight: .bold))
                        .foregroundStyle(passed ? Color.appCollector : Color.appDanger)
                    Label(passed ? "Passed" : "Failed",
                          systemImage: passed ? "checkmark.seal.fill" : "xmark.seal.fill")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(passed ? Color.appCollector : Color.appDanger)
                }
                if let r = s.scoreReasoning, !r.isEmpty { reason("Score notes", r) }
            } else {
                HStack(spacing: 8) {
                    ProgressView()
                    Text("Scoring in progress…").foregroundStyle(.secondary)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color.appSurface, in: RoundedRectangle(cornerRadius: 12))
    }

    private func reason(_ title: String, _ text: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            Text(text).font(.subheadline)
        }
    }

    private func hint(_ text: String) -> some View {
        Text(text).font(.subheadline).foregroundStyle(.secondary)
    }

    private func scoreText(_ value: Double) -> String {
        value == value.rounded() ? String(Int(value)) : String(format: "%.1f", value)
    }

    // MARK: - Transcript

    private var transcriptCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Transcript").font(.headline)
                Spacer()
                Button { loadTranscript() } label: { Image(systemName: "arrow.clockwise") }
            }
            if let t = transcript {
                if t.segments.isEmpty {
                    hint(t.text.isEmpty ? "No speech transcribed." : t.text)
                } else {
                    ForEach(Array(t.segments.enumerated()), id: \.offset) { _, seg in
                        HStack(alignment: .top, spacing: 8) {
                            Text(String(format: "%.2f–%.2f", seg.startTime, seg.endTime))
                                .font(.caption.monospaced()).foregroundStyle(.secondary)
                                .frame(width: 96, alignment: .leading)
                            Text(seg.text).font(.subheadline)
                        }
                    }
                }
                if let s = t.status, !s.isEmpty {
                    Text(s).font(.caption2).foregroundStyle(.secondary)
                }
            } else {
                HStack(spacing: 8) {
                    ProgressView()
                    Text("Transcribing on-device… (tap ↻ to refresh)").foregroundStyle(.secondary)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color.appSurface, in: RoundedRectangle(cornerRadius: 12))
    }

    private func loadTranscript() {
        let url = RecordingStore.folderURL(for: recording.folderName)
            .appendingPathComponent("transcript.json")
        guard let data = try? Data(contentsOf: url),
              let t = try? JSONDecoder().decode(TranscriptFile.self, from: data) else { return }
        transcript = t
    }

    /// transcript.json is written asynchronously after recording — poll for it.
    private func pollTranscript() async {
        for _ in 0..<25 {
            loadTranscript()
            if transcript != nil { return }
            try? await Task.sleep(for: .seconds(3))
        }
    }

    // MARK: - Meta

    private var metaCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Details").font(.headline)
            LabeledContent("Duration", value: RecordingFormat.duration(recording.durationMs))
            LabeledContent("Size", value: RecordingFormat.size(recording.sizeBytes))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color.appSurface, in: RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Load + poll

    private func load() async {
        guard recording.status == .uploaded else { loading = false; return }
        await refresh()
        loading = false
        // Keep polling while the model hasn't returned a score yet.
        while !Task.isCancelled, (score?.isScoring ?? true), score?.score == nil {
            try? await Task.sleep(for: .seconds(4))
            await refresh()
        }
    }

    private func refresh() async {
        let map = (try? await ScoringService.scores(taskId: taskId)) ?? [:]
        let mine = map[recording.id.uuidString.lowercased()]
        await MainActor.run { score = mine }
    }
}

/// Decodes the on-disk transcript.json (text + timestamped segments + debug status).
struct TranscriptFile: Decodable {
    let text: String
    let status: String?
    let segments: [Seg]

    struct Seg: Decodable {
        let startTime: Double
        let endTime: Double
        let text: String
        enum CodingKeys: String, CodingKey {
            case startTime = "start_time"
            case endTime = "end_time"
            case text
        }
    }
}
