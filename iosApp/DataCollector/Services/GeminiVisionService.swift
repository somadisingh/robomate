import Foundation
import AVFoundation
import UIKit

/// One-shot vision calls to Gemini (non-streaming `generateContent`). Used to turn
/// a lab's reference video into a detailed, collector-facing text brief.
enum GeminiVisionService {
    /// Sample frames from a reference video and ask Gemini for a task brief.
    static func describeReference(videoURL: URL,
                                  taskTitle: String?,
                                  taskDescription: String?) async -> String? {
        let frames = await sampleJPEGs(from: videoURL, count: 6)
        guard !frames.isEmpty else { return nil }
        let prompt = """
        These frames are sampled in order from a reference video a research lab recorded to show \
        data collectors exactly what to capture for a robotics dataset.
        \(taskTitle.map { "Task title: \($0)." } ?? "")
        \(taskDescription.map { "Lab notes: \($0)." } ?? "")
        Write a concise brief (3–5 sentences) a collector can follow to replicate this: what is \
        happening, the key objects/scene, the camera framing, angle and distance, and the motion to \
        perform. Be concrete and imperative. No preamble — just the brief.
        """
        return await generate(prompt: prompt, frames: frames)
    }

    /// Low-level `generateContent` call: a text prompt + inline JPEG frames → text.
    static func generate(prompt: String, frames: [Data]) async -> String? {
        let key = SupabaseConfig.geminiAPIKey
        guard !key.isEmpty else { return nil }
        let model = SupabaseConfig.geminiVisionModel
        guard let url = URL(string:
            "https://generativelanguage.googleapis.com/v1beta/\(model):generateContent?key=\(key)")
        else { return nil }

        var parts: [[String: Any]] = [["text": prompt]]
        for jpeg in frames {
            parts.append(["inlineData": ["mimeType": "image/jpeg", "data": jpeg.base64EncodedString()]])
        }
        let body: [String: Any] = ["contents": [["parts": parts]]]

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        request.timeoutInterval = 30

        guard let (data, response) = try? await URLSession.shared.data(for: request),
              let http = response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }

        guard let candidates = obj["candidates"] as? [[String: Any]],
              let firstCandidate = candidates.first,
              let content = firstCandidate["content"] as? [String: Any],
              let outParts = content["parts"] as? [[String: Any]] else {
            return nil
        } 

        let text = outParts.compactMap { $0["text"] as? String }.joined()
            .trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
        return text.isEmpty ? nil : text
    }

    /// Evenly-spaced JPEG frames from a local video file.
    static func sampleJPEGs(from videoURL: URL, count: Int) async -> [Data] {
        let asset = AVURLAsset(url: videoURL)
        guard let duration = try? await asset.load(.duration) else { return [] }
        let durSec = CMTimeGetSeconds(duration)
        guard durSec.isFinite, durSec > 0 else { return [] }
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 640, height: 640)
        var out: [Data] = []
        for i in 0..<count {
            let frac = (Double(i) + 0.5) / Double(count)
            let time = CMTime(seconds: durSec * frac, preferredTimescale: 600)
            if let result = try? await generator.image(at: time),
               let jpeg = UIImage(cgImage: result.image).jpegData(compressionQuality: 0.6) {
                out.append(jpeg)
            }
        }
        return out
    }
}

/// Builds the system instruction for the live coach — shared by the record screen
/// and the test screen so behaviour stays consistent.
enum CoachPrompt {
    static func build(taskTitle: String?, taskDescription: String?, brief: String?) -> String {
        var p = "You are a real-time camera coach for someone about to record a short first-person "
            + "video for a robotics training dataset. Help them frame and perform THIS specific task."
        if let t = taskTitle, !t.isEmpty { p += "\n\nTask: \"\(t)\"." }
        if let d = taskDescription, !d.isEmpty { p += "\nWhat to capture: \(d)" }
        if let brief, !brief.isEmpty {
            p += "\n\nReference — what a GOOD capture of this task looks like:\n\(brief)"
        }
        p += "\n\nLook at the live camera. When asked, reply with ONE very short imperative tip "
            + "(max 8 words) about the single most important thing to fix RIGHT NOW for THIS task — "
            + "e.g. whether the correct object/action is in frame, framing, distance, or lighting. "
            + "Refer to the task's objects by name when relevant. Only mention camera shake if the view "
            + "is SEVERELY shaky. If the shot looks right for the task, reply exactly \"Looks good\". "
            + "Never explain or add pleasantries."
        return p
    }
}
