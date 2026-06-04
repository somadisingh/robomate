import SwiftUI
import SwiftData
import Supabase

/// Account info, progress, password change, and sign out.
struct ProfileView: View {
    @EnvironmentObject private var auth: AuthManager
    @Query private var recordings: [Recording]

    @State private var newPassword = ""
    @State private var message: String?
    @State private var isError = false
    @State private var busy = false
    @State private var earnings: [Earning] = []

    private var email: String { Backend.supabase.auth.currentUser?.email ?? "—" }
    private var uploadedCount: Int { recordings.filter { $0.status == .uploaded }.count }
    private var totalEarned: Double { earnings.reduce(0) { $0 + $1.amount } }
    private var pendingEarned: Double { earnings.filter { $0.status == "pending" }.reduce(0) { $0 + $1.amount } }
    private var paidOut: Double { totalEarned - pendingEarned }

    var body: some View {
        NavigationStack {
            List {
                Section("Account") {
                    LabeledContent("Email", value: email)
                    LabeledContent("Role", value: auth.role?.label ?? "Collector")
                }

                // Collector-only sections.
                if auth.role != .lab {
                    Section("Progress") {
                        LabeledContent("Recordings captured", value: "\(recordings.count)")
                        LabeledContent("Uploaded", value: "\(uploadedCount)")
                    }

                    Section("Earnings") {
                        HStack {
                            Text("Total earned")
                            Spacer()
                            Text(money(totalEarned)).fontWeight(.semibold).foregroundStyle(Color.appCollector)
                        }
                        LabeledContent("Pending", value: money(pendingEarned))
                        LabeledContent("Paid out", value: money(paidOut))
                    }
                }

                Section("Change password") {
                    SecureField("New password (min 6 characters)", text: $newPassword)
                        .textContentType(.newPassword)
                    Button {
                        changePassword()
                    } label: {
                        if busy { ProgressView() } else { Text("Update password") }
                    }
                    .disabled(busy || newPassword.count < 6)
                    if let message {
                        Text(message)
                            .font(.footnote)
                            .foregroundStyle(isError ? Color.appDanger : Color.appCollector)
                    }
                }

                Section("Experimental") {
                    NavigationLink {
                        LiveCoachView()
                    } label: {
                        Label("Live Coach (beta)", systemImage: "sparkles")
                    }
                }

                Section {
                    Button("Sign Out", role: .destructive) {
                        Task { await auth.signOut() }
                    }
                }
            }
            .navigationTitle("Profile")
            .scrollContentBackground(.hidden)
            .background(Color.appBackground.ignoresSafeArea())
            .task { await loadEarnings() }
        }
    }

    private func changePassword() {
        busy = true
        message = nil
        let pw = newPassword
        Task {
            do {
                _ = try await Backend.supabase.auth.update(user: UserAttributes(password: pw))
                await MainActor.run {
                    message = "Password updated."
                    isError = false
                    newPassword = ""
                    busy = false
                }
            } catch {
                await MainActor.run {
                    message = error.localizedDescription
                    isError = true
                    busy = false
                }
            }
        }
    }

    private func loadEarnings() async {
        let result = (try? await EarningsService.fetch()) ?? []
        await MainActor.run { earnings = result }
    }

    private func money(_ value: Double) -> String {
        String(format: "$%.2f", value)
    }
}
