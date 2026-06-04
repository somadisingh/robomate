import SwiftUI
import SwiftData
import UIKit
import AVKit

/// Full-screen LANDSCAPE camera with a record button. Records for a given `job`
/// and inserts a `Recording` into SwiftData when finished. The screen is locked to
/// landscape and the record button is disabled until the phone is held horizontally.
struct RecordView: View {
    @Environment(\.modelContext) private var context
    @Environment(\.dismiss) private var dismiss
    @StateObject private var recorder = Recorder()
    var job: Job? = nil
    var referenceTaskId: String? = nil   // when set, this records a task's reference example
    var referenceTaskTitle: String? = nil        // for generating the reference brief
    var referenceTaskDescription: String? = nil
    var profile: CoachingProfile? = nil  // task-specific coaching targets (from the lab's reference)
    var profileDebug: String? = nil      // diagnostic of how the profile loaded

    @State private var isDeviceLandscape = UIDevice.current.orientation.isLandscape
    @State private var review: ReviewClip?
    @State private var refUploading = false
    @State private var refError: String?
    @State private var showDebug = true   // on-screen coaching debug readout

    var body: some View {
        Group {
            switch recorder.phase {
            case .denied:
                ZStack { Color.black.ignoresSafeArea(); deniedView }
            case .idle:
                ZStack { Color.black.ignoresSafeArea(); ProgressView().tint(.white) }
            default:
                ZStack(alignment: .bottom) {
                    CameraPreviewView(recorder: recorder)

                    recordButton
                        .padding(.bottom, 20)
                        .zIndex(1)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color.black)
                .overlay(alignment: .top) {
                    if recorder.phase == .recording {
                        timerBadge
                    } else if let job {
                        jobPrompt(job)
                    } else if referenceTaskId != nil {
                        referencePrompt
                    }
                }
                // Nudge the user to hold the phone horizontally.
                .overlay {
                    if !isDeviceLandscape && recorder.phase != .recording {
                        rotateHint
                    }
                }
                // Live, on-device coaching chips.
                .overlay(alignment: .topLeading) {
                    if recorder.live.hasData {
                        coachingHUD.padding(12)
                    }
                }
                // Debug readout: which ranges are active (reference vs defaults).
                .overlay(alignment: .topTrailing) { debugOverlay }
            }
        }
        .navigationTitle(referenceTaskId != nil ? "Reference" : (job?.title ?? "Record"))
        .navigationBarTitleDisplayMode(.inline)
        .toolbar(.hidden, for: .tabBar)
        .onAppear {
            AppOrientation.lock(.landscapeRight)               // require landscape here
            UIDevice.current.beginGeneratingDeviceOrientationNotifications()
            isDeviceLandscape = UIDevice.current.orientation.isLandscape
            recorder.bountyId = job?.id.uuidString
            recorder.onFinished = { recording in
                if referenceTaskId == nil {   // collector clips persist locally; references upload on Save
                    context.insert(recording)
                    try? context.save()
                }
                review = ReviewClip(recording: recording)
            }
            recorder.configureIfNeeded()
        }
        .onDisappear {
            recorder.pause()
            UIDevice.current.endGeneratingDeviceOrientationNotifications()
            AppOrientation.lock(.portrait)                     // restore portrait for the rest of the app
        }
        .onReceive(NotificationCenter.default.publisher(for: UIDevice.orientationDidChangeNotification)) { _ in
            let orientation = UIDevice.current.orientation
            if orientation.isValidInterfaceOrientation {
                isDeviceLandscape = orientation.isLandscape
            }
        }
        .fullScreenCover(item: $review) { clip in
            reviewView(clip.recording)
        }
    }

    /// Shown right after a recording (WhatsApp-style): play it back, then Save or Delete.
    private func reviewView(_ rec: Recording) -> some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if let url = rec.videoURL {
                LoopingPlayerView(url: url)
                    .ignoresSafeArea()
            }
            VStack {
                Spacer()
                if let refError {
                    Text(refError).font(.footnote).foregroundStyle(.red)
                        .multilineTextAlignment(.center)
                        .padding(10).background(.black.opacity(0.6), in: Capsule())
                        .padding(.bottom, 8)
                }
                HStack(spacing: 56) {
                    // Discard — red circle with an ✗; stays on the camera.
                    VStack(spacing: 6) {
                        Button(role: .destructive) {
                            RecordingStore.deleteBundle(folderName: rec.folderName)
                            context.delete(rec)
                            try? context.save()
                            review = nil
                        } label: {
                            Image(systemName: "xmark")
                                .font(.system(size: 24, weight: .bold))
                                .foregroundStyle(.white)
                                .frame(width: 64, height: 64)
                                .background(.red.opacity(0.9), in: Circle())
                                .overlay(Circle().stroke(.white.opacity(0.25), lineWidth: 1))
                                .shadow(radius: 6)
                        }
                        .buttonStyle(.plain)
                        Text("Discard").font(.caption).foregroundStyle(.white.opacity(0.85))
                    }

                    // Save — green circle with a ✓ (shows progress while a reference uploads).
                    VStack(spacing: 6) {
                        Button {
                            if let taskId = referenceTaskId {
                                saveReference(rec, taskId: taskId)
                            } else {
                                review = nil
                                dismiss()
                            }
                        } label: {
                            ZStack {
                                Circle()
                                    .fill(Color.green)
                                    .frame(width: 76, height: 76)
                                    .overlay(Circle().stroke(.white.opacity(0.3), lineWidth: 1))
                                    .shadow(color: .green.opacity(0.6), radius: 10)
                                if refUploading {
                                    ProgressView().tint(.white)
                                } else {
                                    Image(systemName: "checkmark")
                                        .font(.system(size: 30, weight: .bold))
                                        .foregroundStyle(.white)
                                }
                            }
                        }
                        .buttonStyle(.plain)
                        .disabled(refUploading)
                        Text(refUploading ? "Saving…" : "Save")
                            .font(.caption.weight(.semibold)).foregroundStyle(.white)
                    }
                }
                .padding(.bottom, 28)
            }
        }
    }

    private var referencePrompt: some View {
        Text("Record a reference example for this task")
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 14).padding(.vertical, 8)
            .background(.black.opacity(0.45), in: Capsule())
            .padding(.top, 8)
    }

    /// Reference mode Save: derive coaching ranges from this clip, upload the
    /// bundle (incl. coaching.json) to references/<taskId>/, and set tasks.reference_path.
    private func saveReference(_ rec: Recording, taskId: String) {
        refUploading = true
        refError = nil
        let folderName = rec.folderName
        var streams = rec.streams
        Task {
            do {
                // Compute per-task coaching targets and bundle them as coaching.json
                // so collectors get task-specific live guidance.
                let folder = RecordingStore.folderURL(for: folderName)
                let coachingProfile = await CoachingProfileBuilder.build(folderURL: folder)
                if let data = try? JSONEncoder().encode(coachingProfile) {
                    let coachingURL = folder.appendingPathComponent("coaching.json")
                    try? data.write(to: coachingURL)
                    if FileManager.default.fileExists(atPath: coachingURL.path),
                       !streams.contains("coaching.json") {
                        streams.append("coaching.json")
                    }
                }
                let path = try await UploadService.uploadReference(
                    folderName: folderName, streams: streams, taskId: taskId
                )
                try await LabTasksService.setReferencePath(taskId: taskId, path: path)

                // Turn the reference video into a text brief for collectors (best-effort;
                // do it before deleting the local bundle).
                if let brief = await GeminiVisionService.describeReference(
                    videoURL: folder.appendingPathComponent("video.mp4"),
                    taskTitle: referenceTaskTitle, taskDescription: referenceTaskDescription) {
                    try? await LabTasksService.setReferenceBrief(taskId: taskId, brief: brief)
                }

                await MainActor.run {
                    RecordingStore.deleteBundle(folderName: folderName)   // uploaded; no local copy needed
                    refUploading = false
                    review = nil
                    dismiss()
                }
            } catch {
                await MainActor.run {
                    refError = error.localizedDescription
                    refUploading = false
                }
            }
        }
    }

    // MARK: - Live coaching HUD

    /// Live chips, checked against the task's reference ranges when available,
    /// otherwise generic defaults.
    private var coachingChips: [CoachingChip] {
        let m = recorder.live
        var chips: [CoachingChip] = []

        let lumMin = profile?.luminanceMin ?? 0.22
        let lumMax = profile?.luminanceMax ?? 0.85
        if m.luminance < lumMin { chips.append(CoachingChip(label: "Too dark", ok: false)) }
        else if m.luminance > lumMax { chips.append(CoachingChip(label: "Too bright", ok: false)) }
        else { chips.append(CoachingChip(label: "Lighting", ok: true)) }

        let motionMax = profile?.motionMax ?? 45
        let steady = m.motionDegPerSec <= motionMax
        chips.append(CoachingChip(label: steady ? "Steady" : "Hold steadier", ok: steady))

        if let d = m.distanceM {
            let dMin = profile?.distanceMin ?? 0.3
            let dMax = profile?.distanceMax ?? 3.0
            if d < dMin { chips.append(CoachingChip(label: "Too close", ok: false)) }
            else if d > dMax { chips.append(CoachingChip(label: "Too far", ok: false)) }
            else { chips.append(CoachingChip(label: "Distance", ok: true)) }
        }
        return chips
    }

    private var coachingHUD: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(coachingChips) { chip in
                HStack(spacing: 6) {
                    Image(systemName: chip.ok ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                    Text(chip.label)
                }
                .font(.caption.weight(.semibold))
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(.black.opacity(0.5), in: Capsule())
                .foregroundStyle(chip.ok ? Color.green : Color.appAmber)
            }
        }
    }

    // MARK: - Debug readout (active ranges: reference vs defaults)

    private var debugOverlay: some View {
        VStack(alignment: .trailing, spacing: 6) {
            Button { showDebug.toggle() } label: {
                Image(systemName: "ladybug.fill")
                    .font(.footnote)
                    .padding(7)
                    .background(.black.opacity(0.55), in: Circle())
                    .foregroundStyle(showDebug ? .cyan : .white)
            }
            if showDebug { debugPanel }
        }
        .padding(12)
    }

    private var debugPanel: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("COACH · \(profile == nil ? "defaults (no profile)" : "reference")")
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundStyle(profile == nil ? .orange : .cyan)
            ForEach(debugRows) { row in
                HStack(spacing: 6) {
                    Text(row.name).frame(width: 42, alignment: .leading)
                    Text(row.live).frame(width: 56, alignment: .leading)
                    Text(row.range).frame(width: 84, alignment: .leading)
                    Text(row.source).foregroundStyle(.white.opacity(0.55))
                    Image(systemName: row.ok ? "checkmark" : "xmark")
                        .foregroundStyle(row.ok ? .green : .orange)
                }
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.white)
            }
            if let profileDebug {
                Text(profileDebug)
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.yellow.opacity(0.85))
                    .frame(maxWidth: 220, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(8)
        .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 8))
    }

    /// One line per metric: live value, active range, source (ref/def), pass/fail.
    private var debugRows: [CoachDebugRow] {
        let m = recorder.live
        var rows: [CoachDebugRow] = []

        let lumMin = profile?.luminanceMin ?? 0.22
        let lumMax = profile?.luminanceMax ?? 0.85
        rows.append(CoachDebugRow(
            name: "Light",
            live: String(format: "%.2f", m.luminance),
            range: String(format: "%.2f-%.2f", lumMin, lumMax),
            source: profile?.luminanceMin != nil ? "ref" : "def",
            ok: m.luminance >= lumMin && m.luminance <= lumMax))

        let motionMax = profile?.motionMax ?? 45
        rows.append(CoachDebugRow(
            name: "Steady",
            live: String(format: "%.0f/s", m.motionDegPerSec),
            range: String(format: "<=%.0f", motionMax),
            source: profile?.motionMax != nil ? "ref" : "def",
            ok: m.motionDegPerSec <= motionMax))

        if let d = m.distanceM {
            let dMin = profile?.distanceMin ?? 0.3
            let dMax = profile?.distanceMax ?? 3.0
            rows.append(CoachDebugRow(
                name: "Dist",
                live: String(format: "%.2fm", d),
                range: String(format: "%.2f-%.2f", dMin, dMax),
                source: profile?.distanceMin != nil ? "ref" : "def",
                ok: d >= dMin && d <= dMax))
        } else {
            rows.append(CoachDebugRow(name: "Dist", live: "n/a", range: "no LiDAR", source: "-", ok: true))
        }
        return rows
    }

    private var rotateHint: some View {
        VStack(spacing: 12) {
            Image(systemName: "rotate.right.fill")
                .font(.system(size: 44))
            Text("Hold your phone sideways to record")
                .font(.headline)
        }
        .foregroundStyle(.white)
        .padding(28)
        .background(.black.opacity(0.7), in: RoundedRectangle(cornerRadius: 18))
    }

    private func jobPrompt(_ job: Job) -> some View {
        VStack(spacing: 2) {
            Text("Record: \(job.title)").font(.subheadline.weight(.semibold))
            if let desc = job.description, !desc.isEmpty {
                Text(desc).font(.caption).foregroundStyle(.white.opacity(0.85)).lineLimit(2)
            }
        }
        .foregroundStyle(.white)
        .multilineTextAlignment(.center)
        .padding(.horizontal, 14).padding(.vertical, 8)
        .background(.black.opacity(0.45), in: Capsule())
        .padding(.top, 8)
    }

    private var timerBadge: some View {
        Text(timeString(recorder.elapsed))
            .font(.system(.title3, design: .monospaced).weight(.semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 14).padding(.vertical, 6)
            .background(.red, in: Capsule())
            .padding(.top, 8)
    }

    private var recordButton: some View {
        Button(action: recorder.toggle) {
            ZStack {
                Circle().stroke(.white, lineWidth: 5).frame(width: 80, height: 80)
                RoundedRectangle(cornerRadius: recorder.phase == .recording ? 7 : 34)
                    .fill(.red)
                    .frame(width: recorder.phase == .recording ? 36 : 68,
                           height: recorder.phase == .recording ? 36 : 68)
                    .animation(.easeInOut(duration: 0.2), value: recorder.phase)
            }
            .opacity(canRecord || recorder.phase == .recording ? 1 : 0.4)
        }
        // Allow stopping anytime; only allow starting when ready AND held landscape.
        .disabled(!(recorder.phase == .recording || canRecord))
    }

    private var canRecord: Bool {
        recorder.phase == .ready && isDeviceLandscape
    }

    private var deniedView: some View {
        VStack(spacing: 12) {
            Image(systemName: "video.slash.fill").font(.largeTitle).foregroundStyle(.white)
            Text("Camera access needed").font(.headline).foregroundStyle(.white)
            Text("Enable Camera, Location and Motion access in Settings to record.")
                .font(.subheadline).foregroundStyle(.white.opacity(0.7))
                .multilineTextAlignment(.center)
            Button("Open Settings") {
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }

    private func timeString(_ t: TimeInterval) -> String {
        String(format: "%02d:%02d", Int(t) / 60, Int(t) % 60)
    }
}

/// Wrapper so a just-recorded clip can drive `.fullScreenCover(item:)`.
private struct ReviewClip: Identifiable {
    let id = UUID()
    let recording: Recording
}

/// One live coaching chip (e.g. "Lighting ✓").
private struct CoachingChip: Identifiable {
    let id = UUID()
    let label: String
    let ok: Bool
}

/// One row in the on-screen coaching debug readout.
private struct CoachDebugRow: Identifiable {
    let id = UUID()
    let name: String     // metric
    let live: String     // current value
    let range: String    // active acceptable range
    let source: String   // "ref" | "def" | "-"
    let ok: Bool
}
