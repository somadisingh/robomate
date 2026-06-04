import SwiftUI

/// Lab tab bar: My Tasks + Profile.
struct LabHomeView: View {
    var body: some View {
        TabView {
            MyTasksView()
                .tabItem { Label("My Tasks", systemImage: "tray.full") }
            ProfileView()
                .tabItem { Label("Profile", systemImage: "person.crop.circle") }
        }
        .tint(.appAccent)
    }
}

// MARK: - My tasks

struct MyTasksView: View {
    @State private var tasks: [LabTask] = []
    @State private var loading = true
    @State private var error: String?
    @State private var showCreate = false

    var body: some View {
        NavigationStack {
            Group {
                if loading {
                    ProgressView("Loading tasks…")
                } else if let error {
                    ContentUnavailableView("Couldn't load tasks", systemImage: "wifi.slash",
                                           description: Text(error))
                } else if tasks.isEmpty {
                    ContentUnavailableView("No tasks yet", systemImage: "tray",
                                           description: Text("Tap + to create your first task."))
                } else {
                    List {
                        Section { statsRow }
                        Section("Tasks") {
                            ForEach(tasks) { task in
                                NavigationLink { LabTaskDetailView(task: task) } label: { taskRow(task) }
                            }
                        }
                    }
                    .scrollContentBackground(.hidden)
                }
            }
            .navigationTitle("Your Tasks")
            .background(Color.appBackground.ignoresSafeArea())
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showCreate = true } label: { Image(systemName: "plus") }
                }
            }
            .sheet(isPresented: $showCreate, onDismiss: { Task { await load() } }) {
                CreateTaskView()
            }
            .refreshable { await load() }
            .task { await load() }
        }
    }

    private var statsRow: some View {
        HStack {
            stat("\(tasks.count)", "Tasks")
            Divider()
            stat("\(tasks.reduce(0) { $0 + ($1.quantityFilled ?? 0) })", "Collected")
            Divider()
            stat(money(tasks.reduce(0) { $0 + ($1.bountyAmount ?? 0) * Double($1.quantityFilled ?? 0) }), "Spent")
        }
    }

    private func stat(_ value: String, _ label: String) -> some View {
        VStack {
            Text(value).font(.headline)
            Text(label).font(.caption2).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }

    private func taskRow(_ task: LabTask) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(task.title).font(.headline)
                Spacer()
                if let s = task.status {
                    Text(s).font(.caption2.weight(.semibold))
                        .padding(.horizontal, 8).padding(.vertical, 2)
                        .background(Color.appAccent.opacity(0.18), in: Capsule())
                        .foregroundStyle(Color.appAccent)
                }
            }
            HStack(spacing: 8) {
                if let b = task.bountyAmount {
                    Text("$\(b, specifier: "%.0f") / submission").font(.caption).foregroundStyle(.secondary)
                }
                Text("\(task.quantityFilled ?? 0)/\(task.quantityNeeded ?? 0) collected")
                    .font(.caption).foregroundStyle(.secondary)
            }
            ProgressView(value: progress(task)).tint(Color.appAccent)
        }
        .padding(.vertical, 4)
    }

    private func progress(_ t: LabTask) -> Double {
        guard let need = t.quantityNeeded, need > 0 else { return 0 }
        return min(1, Double(t.quantityFilled ?? 0) / Double(need))
    }

    private func load() async {
        if tasks.isEmpty { loading = true }
        do { tasks = try await LabTasksService.myTasks(); error = nil }
        catch { self.error = error.localizedDescription }
        loading = false
    }

    private func money(_ v: Double) -> String { String(format: "$%.0f", v) }
}

// MARK: - Create task

struct CreateTaskView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var title = ""
    @State private var description = ""
    @State private var capabilities: Set<String> = []
    @State private var bounty = "5"
    @State private var quantity = 24
    @State private var busy = false
    @State private var error: String?

    private let allCapabilities = ["outdoor", "indoor", "motion", "monochrome"]

    var body: some View {
        NavigationStack {
            Form {
                Section("Brief") {
                    TextField("Title", text: $title)
                    TextField("Description", text: $description, axis: .vertical).lineLimit(3...6)
                }
                Section("Requirements") {
                    ForEach(allCapabilities, id: \.self) { cap in
                        Button {
                            if capabilities.contains(cap) { capabilities.remove(cap) }
                            else { capabilities.insert(cap) }
                        } label: {
                            HStack {
                                Text(cap.capitalized)
                                Spacer()
                                if capabilities.contains(cap) {
                                    Image(systemName: "checkmark").foregroundStyle(Color.appAccent)
                                }
                            }
                        }
                        .foregroundStyle(.primary)
                    }
                }
                Section("Payout") {
                    HStack {
                        Text("Bounty $")
                        TextField("5", text: $bounty)
                            .keyboardType(.decimalPad)
                            .multilineTextAlignment(.trailing)
                    }
                    Stepper("Submissions: \(quantity)", value: $quantity, in: 1...1000)
                }
                if let error { Text(error).foregroundStyle(.red).font(.footnote) }
            }
            .scrollContentBackground(.hidden)
            .background(Color.appBackground.ignoresSafeArea())
            .navigationTitle("New Task")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Post") { submit() }
                        .disabled(busy || title.isEmpty || description.isEmpty)
                }
            }
        }
    }

    private func submit() {
        busy = true
        error = nil
        let amount = Double(bounty) ?? 0
        Task {
            do {
                try await LabTasksService.createTask(
                    title: title, description: description,
                    capabilities: Array(capabilities), dataType: "video",
                    bountyAmount: amount, quantityNeeded: quantity
                )
                await MainActor.run { dismiss() }
            } catch {
                await MainActor.run { self.error = error.localizedDescription; busy = false }
            }
        }
    }
}

// MARK: - Task detail

struct LabTaskDetailView: View {
    let task: LabTask
    @State private var referencePath: String?
    @State private var loadingRef = true

    var body: some View {
        List {
            Section {
                VStack(alignment: .leading, spacing: 8) {
                    if let b = task.bountyAmount {
                        Text("$\(b, specifier: "%.0f") / submission")
                            .font(.title3.bold()).foregroundStyle(Color.appCollector)
                    }
                    if let d = task.description, !d.isEmpty { Text(d).font(.subheadline) }
                    Text("\(task.quantityFilled ?? 0)/\(task.quantityNeeded ?? 0) collected")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
            }
            Section("Reference video") {
                if loadingRef {
                    HStack { ProgressView(); Text("Checking…").foregroundStyle(.secondary) }
                } else if referencePath != nil {
                    ReferenceVideoView(taskId: task.id.uuidString)
                    NavigationLink {
                        RecordView(referenceTaskId: task.id.uuidString,
                                   referenceTaskTitle: task.title,
                                   referenceTaskDescription: task.description)
                    } label: {
                        Label("Re-record reference", systemImage: "arrow.triangle.2.circlepath")
                    }
                } else {
                    Text("Record an example clip in-app so collectors — and the scoring model — know what a good capture looks like.")
                        .font(.caption).foregroundStyle(.secondary)
                    NavigationLink {
                        RecordView(referenceTaskId: task.id.uuidString,
                                   referenceTaskTitle: task.title,
                                   referenceTaskDescription: task.description)
                    } label: {
                        Label("Record reference video", systemImage: "video.badge.plus")
                    }
                }
            }
        }
        .scrollContentBackground(.hidden)
        .background(Color.appBackground.ignoresSafeArea())
        .navigationTitle(task.title)
        .navigationBarTitleDisplayMode(.inline)
        .task { await loadReference() }
    }

    private func loadReference() async {
        let path = try? await LabTasksService.referencePath(taskId: task.id.uuidString)
        await MainActor.run { referencePath = path ?? nil; loadingRef = false }
    }
}
