import SwiftUI

/// Browse open jobs. Tap one to record data for it.
struct JobsView: View {
    @State private var jobs: [Job] = []
    @State private var loading = true
    @State private var error: String?

    var body: some View {
        NavigationStack {
            Group {
                if loading {
                    ProgressView("Loading jobs…")
                } else if let error {
                    ContentUnavailableView("Couldn't load jobs",
                                           systemImage: "wifi.slash",
                                           description: Text(error))
                } else if jobs.isEmpty {
                    ContentUnavailableView("No open jobs",
                                           systemImage: "tray",
                                           description: Text("Check back later."))
                } else {
                    List(jobs) { job in
                        NavigationLink {
                            JobDetailView(job: job)
                        } label: {
                            JobRow(job: job)
                        }
                    }
                    .scrollContentBackground(.hidden)
                }
            }
            .navigationTitle("Jobs")
            .background(Color.appBackground.ignoresSafeArea())
            .refreshable { await load() }
            .task { await load() }
        }
    }

    private func load() async {
        if jobs.isEmpty { loading = true }
        do {
            jobs = try await JobsService.fetchOpenJobs()
            error = nil
        } catch {
            self.error = error.localizedDescription
        }
        loading = false
    }
}

private struct JobRow: View {
    let job: Job

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text(job.title).font(.headline)
                Spacer()
                if let amount = job.bountyAmount {
                    Text("$\(amount, specifier: "%.0f")")
                        .font(.headline).foregroundStyle(Color.appCollector)
                }
            }
            if let desc = job.description, !desc.isEmpty {
                Text(desc).font(.subheadline).foregroundStyle(.secondary).lineLimit(2)
            }
            HStack(spacing: 6) {
                if let dataType = job.dataType { tag(dataType) }
                ForEach(job.requiredCapabilities ?? [], id: \.self) { tag($0) }
            }
            if let needed = job.quantityNeeded {
                Text("\(job.quantityFilled ?? 0)/\(needed) collected")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }

    private func tag(_ text: String) -> some View {
        Text(text)
            .font(.caption2.weight(.medium))
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(Color.appBorder, in: Capsule())
    }
}
