import SwiftUI
import SwiftData

/// A job's details + the recordings captured for it (filtered by bountyId),
/// with a Record button pinned at the bottom.
struct JobDetailView: View {
    let job: Job

    @Environment(\.modelContext) private var context
    @Query private var recordings: [Recording]
    @State private var progress: [UUID: Double] = [:]
    @State private var scores: [String: RecordingScore] = [:]   // recordingId -> status + score
    @State private var errorMessage: String?
    @State private var pendingDelete: Recording?
    @State private var playing: PlayableVideo?
    @State private var detail: DetailItem?
    @State private var pollTrigger = 0
    @State private var coachingProfile: CoachingProfile?   // task-specific live targets
    @State private var coachingDebug: String?              // why it loaded / didn't
    @State private var coachingBrief: String?              // lab's AI brief for the live coach

    @State private var showCoach = false   // AI coach presented as a full-screen modal

    init(job: Job) {
        self.job = job
        let jobId: String? = job.id.uuidString
        _recordings = Query(
            filter: #Predicate<Recording> { $0.bountyId == jobId },
            sort: [SortDescriptor(\Recording.createdAt, order: .reverse)]
        )
    }

    var body: some View {
        List {
            Section { jobInfo }
            Section("Reference example") {
                ReferenceVideoView(taskId: job.id.uuidString)
                if coachingBrief != nil {
                    Label("AI brief ready for coaching", systemImage: "checkmark.seal.fill")
                        .font(.caption).foregroundStyle(Color.appCollector)
                } else {
                    Label("No AI brief yet — lab should re-record the reference",
                          systemImage: "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            Section("Your recordings (\(recordings.count))") {
                if recordings.isEmpty {
                    Text("No recordings yet. Tap Record below to capture one.")
                        .font(.subheadline).foregroundStyle(.secondary)
                } else {
                    ForEach(recordings) { rec in row(rec) }
                }
            }
        }
        .navigationTitle(job.title)
        .navigationBarTitleDisplayMode(.inline)
        .scrollContentBackground(.hidden)
        .background(Color.appBackground.ignoresSafeArea())
        .task(id: pollTrigger) { await pollScores() }
        .task {
            let loaded = await CoachingService.load(taskId: job.id.uuidString)
            coachingProfile = loaded.profile
            coachingDebug = loaded.debug
            coachingBrief = loaded.brief
        }
        .refreshable { await loadScores() }
        .safeAreaInset(edge: .bottom) { recordBar }
        .sheet(item: $playing) { VideoPlayerSheet(url: $0.url) }
        .sheet(item: $detail) { item in
            RecordingDetailView(recording: item.recording, taskId: job.id.uuidString)
        }
        // Present the AI coach as a full-screen modal (not a nav push) — a clean,
        // top-level context like the beta, instead of pushed two levels deep.
        .fullScreenCover(isPresented: $showCoach) {
            NavigationStack {
                LiveCoachView(taskTitle: job.title, taskDescription: job.description,
                              brief: coachingBrief)
                    .toolbar {
                        ToolbarItem(placement: .topBarLeading) {
                            Button("Done") { showCoach = false }
                        }
                    }
            }
        }
        .alert("Upload failed", isPresented: .constant(errorMessage != nil)) {
            Button("OK") { errorMessage = nil }
        } message: {
            Text(errorMessage ?? "")
        }
        .confirmationDialog("Delete this recording?",
                            isPresented: .constant(pendingDelete != nil),
                            titleVisibility: .visible) {
            Button("Delete", role: .destructive) {
                if let rec = pendingDelete { performDelete(rec) }
                pendingDelete = nil
            }
            Button("Cancel", role: .cancel) { pendingDelete = nil }
        }
    }

    // MARK: - Job info

    private var jobInfo: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let amount = job.bountyAmount {
                Text("$\(amount, specifier: "%.0f")")
                    .font(.title3.bold()).foregroundStyle(.green)
            }
            if let desc = job.description, !desc.isEmpty {
                Text(desc).font(.subheadline)
            }
            HStack(spacing: 6) {
                if let dataType = job.dataType { tag(dataType) }
                ForEach(job.requiredCapabilities ?? [], id: \.self) { tag($0) }
            }
            if let needed = job.quantityNeeded {
                Text("\(job.quantityFilled ?? 0)/\(needed) collected")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }

    private func tag(_ text: String) -> some View {
        Text(text)
            .font(.caption2.weight(.medium))
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(Color.gray.opacity(0.15), in: Capsule())
    }

    // MARK: - Recording row

    private func row(_ rec: Recording) -> some View {
        HStack(spacing: 12) {
            if let url = rec.videoURL {
                VideoThumbnail(url: url)
                    .onTapGesture { playing = PlayableVideo(url: url) }
            }
            Button {
                detail = DetailItem(recording: rec)
            } label: {
                VStack(alignment: .leading, spacing: 4) {
                    Text(rec.createdAt.formatted(date: .abbreviated, time: .shortened))
                        .font(.subheadline.weight(.medium))
                    Text("\(RecordingFormat.duration(rec.durationMs)) · \(RecordingFormat.size(rec.sizeBytes))")
                        .font(.caption).foregroundStyle(.secondary)
                    statusBadge(rec)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            Spacer()
            VStack(spacing: 10) {
                if let p = progress[rec.id] {
                    VStack(spacing: 4) {
                        ProgressView(value: p)
                            .progressViewStyle(.linear)
                            .frame(width: 90)
                        Text("\(Int(p * 100))%")
                            .font(.caption2).monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                } else if rec.status == .local || rec.status == .failed {
                    Button("Upload") { upload(rec) }
                        .buttonStyle(.borderedProminent).controlSize(.small)
                }
                Button(role: .destructive) { pendingDelete = rec } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.borderless).tint(.red)
            }
        }
        .padding(.vertical, 2)
    }

    // MARK: - Record button

    private var recordBar: some View {
        VStack(spacing: 10) {
            // Real-time AI coaching happens here (before recording) — it can't run
            // during capture (camera/GPU is owned by the recorder).
            if !SupabaseConfig.geminiAPIKey.isEmpty {
                Button { showCoach = true } label: {
                    Label("Practice with AI coach", systemImage: "sparkles")
                        .font(.subheadline.weight(.semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 11)
                        .background(Color.appAccent.opacity(0.18), in: RoundedRectangle(cornerRadius: 12))
                        .foregroundStyle(Color.appAccent)
                }
            }
            NavigationLink {
                RecordView(job: job, profile: coachingProfile, profileDebug: coachingDebug)
            } label: {
                Label("Record", systemImage: "record.circle")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(.red, in: RoundedRectangle(cornerRadius: 14))
                    .foregroundStyle(.white)
            }
        }
        .padding(.horizontal).padding(.vertical, 10)
        .background(.bar)
    }

    // MARK: - Actions

    private func upload(_ rec: Recording) {
        let payload = UploadPayload(
            id: rec.id,
            folderName: rec.folderName,
            streams: rec.streams,
            bountyId: rec.bountyId,
            durationMs: rec.durationMs,
            sizeBytes: rec.sizeBytes,
            gpsLat: rec.gpsLat,
            gpsLon: rec.gpsLon,
            gpsAccuracyM: rec.gpsAccuracyM
        )
        let id = rec.id
        progress[id] = 0
        rec.status = .uploading
        try? context.save()

        Task {
            do {
                try await UploadService.upload(payload) { p in
                    progress[id] = p
                }
                await MainActor.run { rec.status = .uploaded; try? context.save() }
            } catch {
                await MainActor.run {
                    rec.status = .failed
                    errorMessage = error.localizedDescription
                    try? context.save()
                }
            }
            await MainActor.run { progress[id] = nil; pollTrigger += 1 }   // start polling for the score
        }
    }

    private func performDelete(_ rec: Recording) {
        RecordingStore.deleteBundle(folderName: rec.folderName)
        context.delete(rec)
        try? context.save()
    }

    // MARK: - Score badge

    @ViewBuilder
    private func statusBadge(_ rec: Recording) -> some View {
        let s = displayStatus(rec)
        HStack(spacing: 4) {
            if s.spinner { ProgressView().controlSize(.mini) }
            Text(s.text)
        }
        .font(.caption2.weight(.semibold))
        .padding(.horizontal, 8).padding(.vertical, 2)
        .background(s.color.opacity(0.18), in: Capsule())
        .foregroundStyle(s.color)
    }

    /// Shows the AI score when available, a "Scoring…" spinner while pending,
    /// or the local upload status before upload.
    private func displayStatus(_ rec: Recording) -> (text: String, color: Color, spinner: Bool) {
        guard let s = scores[rec.id.uuidString.lowercased()] else {
            if rec.status == .uploaded { return ("Scoring…", .appAmber, true) }
            return (rec.status.label, rec.status.color, false)
        }
        if s.isScoring { return ("Scoring…", .appAmber, true) }
        if let value = s.score {
            let passed = s.success ?? (value >= 5)
            return ("\(passed ? "✓" : "✗") \(scoreText(value))/10",
                    passed ? .appCollector : .appDanger, false)
        }
        return (rec.status.label, rec.status.color, false)
    }

    private func scoreText(_ value: Double) -> String {
        value == value.rounded() ? String(Int(value)) : String(format: "%.1f", value)
    }

    private var hasPendingScores: Bool {
        recordings.contains { rec in
            rec.status == .uploaded && (scores[rec.id.uuidString.lowercased()]?.isScoring ?? true)
        }
    }

    private func loadScores() async {
        let map = (try? await ScoringService.scores(taskId: job.id.uuidString)) ?? [:]
        await MainActor.run { scores = map }
    }

    /// Load once, then poll every few seconds while anything is still scoring.
    private func pollScores() async {
        await loadScores()
        while !Task.isCancelled && hasPendingScores {
            try? await Task.sleep(for: .seconds(4))
            await loadScores()
        }
    }
}

/// Wrapper so a recording can drive `.sheet(item:)`.
private struct DetailItem: Identifiable {
    let id = UUID()
    let recording: Recording
}
