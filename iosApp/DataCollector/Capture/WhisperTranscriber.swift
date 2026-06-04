import Foundation
import AVFoundation
import WhisperKit

/// On-device speech-to-text with WhisperKit (CoreML). Runs AFTER recording on the
/// clip's audio, so the camera is already off — no camera/GPU conflict. Whisper's
/// own segments carry real start/end times. First run downloads the model (~150MB).
enum WhisperTranscriber {
    /// One phrase segment, timed from the start of the audio.
    struct Segment: Codable {
        let startTime: Double
        let endTime: Double
        let text: String
        enum CodingKeys: String, CodingKey {
            case startTime = "start_time"
            case endTime = "end_time"
            case text
        }
    }

    /// English base model — small (~150MB) and fast; cached after first use.
    private static let model = "base.en"

    /// Transcribe a recorded video's audio, with a hard timeout so it never hangs
    /// silently (always returns a status). Returns (segments, status).
    static func transcribe(videoURL: URL) async -> ([Segment], String) {
        await withTaskGroup(of: ([Segment], String).self) { group in
            group.addTask { await transcribeInner(videoURL: videoURL) }
            group.addTask {
                try? await Task.sleep(nanoseconds: 90 * 1_000_000_000)
                return ([], "whisper timed out (90s) — model download/cache stall?")
            }
            let result = await group.next() ?? ([], "whisper: no result")
            group.cancelAll()
            return result
        }
    }

    private static func transcribeInner(videoURL: URL) async -> ([Segment], String) {
        guard let audioURL = await extractAudio(from: videoURL) else {
            return ([], "no audio track to transcribe")
        }
        defer { try? FileManager.default.removeItem(at: audioURL) }
        do {
            let whisper = try await WhisperKit(WhisperKitConfig(model: model))
            // Word timestamps for real timing; disable Whisper's drop-filters so quiet
            // or near-duplicate phrases aren't discarded.
            let options = DecodingOptions(
                wordTimestamps: true,
                compressionRatioThreshold: nil,   // don't drop "repetitive"-looking text
                logProbThreshold: nil,            // don't drop low-confidence text
                noSpeechThreshold: nil            // don't drop quiet text as silence
            )
            let results = try await whisper.transcribe(audioPath: audioURL.path, decodeOptions: options)

            // Prefer word-level timings → group into sentence/phrase segments.
            let words = results.flatMap { $0.segments }.flatMap { $0.words ?? [] }
                .map { (start: Double($0.start), end: Double($0.end), text: clean($0.word)) }
                // Drop Whisper non-speech markers like [BLANK_AUDIO], [Music] (bracketed).
                .filter { !$0.text.isEmpty && !$0.text.contains("[") && !$0.text.contains("]") }
            if !words.isEmpty {
                return (groupIntoPhrases(words), "whisper \(model)")
            }

            // Fallback: segment-level with special tokens stripped.
            let segs = results.flatMap { $0.segments }.compactMap { seg -> Segment? in
                let text = clean(seg.text)
                guard !text.isEmpty else { return nil }
                return Segment(startTime: round2(Double(seg.start)),
                               endTime: round2(Double(seg.end)), text: text)
            }
            return (segs, segs.isEmpty ? "whisper: no speech" : "whisper \(model) (no word times)")
        } catch {
            return ([], "whisper error: \(error.localizedDescription)")
        }
    }

    /// Group words into phrases: break on sentence-ending punctuation or a >1s pause.
    private static func groupIntoPhrases(_ words: [(start: Double, end: Double, text: String)]) -> [Segment] {
        var phrases: [Segment] = []
        var buffer: [(start: Double, end: Double, text: String)] = []
        func flush() {
            guard let first = buffer.first, let last = buffer.last else { return }
            phrases.append(Segment(startTime: round2(first.start), endTime: round2(last.end),
                                   text: buffer.map(\.text).joined(separator: " ")))
            buffer.removeAll()
        }
        for (i, w) in words.enumerated() {
            buffer.append(w)
            let endsSentence = w.text.last.map { ".!?".contains($0) } ?? false
            let gapToNext = i + 1 < words.count ? words[i + 1].start - w.end : Double.infinity
            if endsSentence || gapToNext > 1.0 { flush() }
        }
        flush()
        return phrases
    }

    /// Strip Whisper special/timestamp tokens like <|startoftranscript|>, <|6.12|>,
    /// and bracketed non-speech markers like [BLANK_AUDIO], [Music].
    private static func clean(_ s: String) -> String {
        s.replacingOccurrences(of: "<\\|[^|]*\\|>", with: "", options: .regularExpression)
            .replacingOccurrences(of: "\\[[^\\]]*\\]", with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Extract the audio track to a temp .m4a (WhisperKit reads audio files, not .mp4).
    private static func extractAudio(from videoURL: URL) async -> URL? {
        let asset = AVURLAsset(url: videoURL)
        let tracks = (try? await asset.loadTracks(withMediaType: .audio)) ?? []
        guard !tracks.isEmpty else { return nil }
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".m4a")
        guard let export = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetAppleM4A) else {
            return nil
        }
        export.outputURL = out
        export.outputFileType = .m4a
        await withCheckedContinuation { cont in
            export.exportAsynchronously { cont.resume() }
        }
        return export.status == .completed ? out : nil
    }

    private static func round2(_ x: Double) -> Double { (x * 100).rounded() / 100 }
}
