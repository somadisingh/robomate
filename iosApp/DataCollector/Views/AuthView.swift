import SwiftUI

/// Email + password sign in / sign up, with a role picker (Lab / Collector) on sign-up.
struct AuthView: View {
    @EnvironmentObject private var auth: AuthManager
    @State private var email = ""
    @State private var password = ""
    @State private var name = ""
    @State private var role: UserRole = .collector
    @State private var isSignUp = false
    @State private var error: String?
    @State private var busy = false

    var body: some View {
        VStack(spacing: 20) {
            Spacer()
            Image(systemName: "video.badge.waveform")
                .font(.system(size: 56))
                .foregroundStyle(.tint)
            Text("Physical Data Collector")
                .font(.title2.weight(.bold))

            VStack(spacing: 12) {
                if isSignUp {
                    Picker("I am a", selection: $role) {
                        Text("Collector").tag(UserRole.collector)
                        Text("Lab").tag(UserRole.lab)
                    }
                    .pickerStyle(.segmented)

                    TextField(role == .lab ? "Lab name" : "Your name", text: $name)
                        .textContentType(.name)
                }
                TextField("Email", text: $email)
                    .textContentType(.emailAddress)
                    .keyboardType(.emailAddress)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                SecureField("Password", text: $password)
                    .textContentType(isSignUp ? .newPassword : .password)
            }
            .textFieldStyle(.roundedBorder)

            if let error {
                Text(error).font(.footnote).foregroundStyle(.red)
                    .multilineTextAlignment(.center)
            }

            Button(action: submit) {
                if busy {
                    ProgressView().frame(maxWidth: .infinity)
                } else {
                    Text(isSignUp ? "Sign Up" : "Sign In").frame(maxWidth: .infinity)
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(busy || email.isEmpty || password.isEmpty || (isSignUp && name.isEmpty))

            Button(isSignUp ? "Have an account? Sign In" : "New here? Sign Up") {
                isSignUp.toggle(); error = nil
            }
            .font(.footnote)

            Spacer()
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.appBackground.ignoresSafeArea())
    }

    private func submit() {
        busy = true
        error = nil
        Task {
            do {
                if isSignUp {
                    try await auth.signUp(email: email, password: password, displayName: name, role: role)
                } else {
                    try await auth.signIn(email: email, password: password)
                }
                if !auth.isAuthenticated {
                    error = "Account created — check your email to confirm, then sign in."
                    isSignUp = false
                }
            } catch {
                self.error = error.localizedDescription
            }
            busy = false
        }
    }
}
