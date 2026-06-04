import SwiftUI

/// Standalone test harness for the Gemini Live coaching agent: opens the camera,
/// streams ~1 JPEG/sec to Gemini over WebSocket, and shows the model's short
/// text tips plus a raw connection log. Not wired into the recording path yet —
/// this exists to prove the API end-to-end before integrating.
struct LiveCoachView: View {
    @StateObject private var recorder = Recorder()
    @StateObject private var gemini = GeminiLiveClient()
    @State private var showLog = false

    /// Optional task context so the coach knows what "good" looks like.
    var taskTitle: String? = nil
    var taskDescription: String? = nil
    var brief: String? = nil          // lab's AI-generated reference brief

    // Poke for a tip frequently; the client's idle-guard prevents overlapping
    // requests, so the real cadence is "as soon as the last tip finished".
    private let askTimer = Timer.publish(every: 2, on: .main, in: .common).autoconnect()

    var body: some View {
        ZStack(alignment: .bottom) {
            switch recorder.phase {
            case .denied:
                Color.black.ignoresSafeArea(); deniedView
            case .idle:
                Color.black.ignoresSafeArea(); ProgressView().tint(.white)
            default:
                CameraPreviewView(recorder: recorder).ignoresSafeArea()
                overlay
            }
        }
        .navigationTitle("Live Coach (beta)")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button(showLog ? "Hide log" : "Log") { showLog.toggle() }
            }
        }
        .onAppear {
            AppOrientation.lock(.landscapeRight)   // coaching is for landscape recording
            recorder.onCoachingFrame = { data in
                Task { @MainActor in gemini.sendFrame(data) }
            }
            recorder.configureIfNeeded()
            gemini.connect(systemInstruction:
                CoachPrompt.build(taskTitle: taskTitle, taskDescription: taskDescription, brief: brief))
        }
        .onDisappear {
            recorder.onCoachingFrame = nil
            recorder.pause()
            gemini.disconnect()
            AppOrientation.lock(.portrait)
        }
        .onReceive(askTimer) { _ in
            if gemini.isReady { gemini.requestTip("Give ONE short coaching tip now.") }
        }
    }

    // MARK: - Overlay

    private var overlay: some View {
        VStack {
            HStack {
                statusPill
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.top, 8)

            if let taskTitle, !taskTitle.isEmpty {
                HStack {
                    Text("\(taskTitle) · \(brief != nil ? "reference brief ✓" : "no brief")")
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.8))
                        .padding(.horizontal, 8).padding(.vertical, 3)
                        .background(.black.opacity(0.5), in: Capsule())
                    Spacer()
                }
                .padding(.horizontal, 12)
            }

            if showLog { logPanel }

            Spacer()

            if !gemini.latestTip.isEmpty {
                tipBanner(gemini.latestTip)
                    .padding(.horizontal, 16)
                    .padding(.bottom, 28)
            }
        }
    }

    private var statusPill: some View {
        let (text, color): (String, Color) = {
            switch gemini.status {
            case .idle:        return ("Idle", .gray)
            case .connecting:  return ("Connecting…", .appAmber)
            case .ready:       return ("Live", .appCollector)
            case .closed:      return ("Closed", .gray)
            case .error(let m): return ("Error: \(m)", .appDanger)
            }
        }()
        return HStack(spacing: 6) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(text).font(.caption.weight(.semibold)).foregroundStyle(.white)
                .lineLimit(2)
        }
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(.black.opacity(0.55), in: Capsule())
    }

    private func tipBanner(_ tip: String) -> some View {
        let good = tip.lowercased().contains("looks good")
        return HStack(spacing: 10) {
            Image(systemName: good ? "checkmark.circle.fill" : "lightbulb.fill")
            Text(tip).font(.headline)
            Spacer()
        }
        .foregroundStyle(.white)
        .padding(14)
        .background((good ? Color.appCollector : Color.appAccent).opacity(0.85),
                    in: RoundedRectangle(cornerRadius: 14))
        .shadow(radius: 8)
    }

    private var logPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 2) {
                ForEach(Array(gemini.log.enumerated()), id: \.offset) { _, line in
                    Text(line).font(.caption2.monospaced()).foregroundStyle(.white.opacity(0.85))
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .padding(8)
        }
        .frame(height: 160)
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 10))
        .padding(.horizontal, 12)
    }

    private var deniedView: some View {
        VStack(spacing: 12) {
            Image(systemName: "video.slash.fill").font(.largeTitle)
            Text("Camera access needed").font(.headline)
        }
        .foregroundStyle(.white)
    }
}
