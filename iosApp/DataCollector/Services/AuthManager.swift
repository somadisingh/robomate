import Foundation
import Supabase

/// Auth state over Supabase Auth (email + password) with the account role.
@MainActor
final class AuthManager: ObservableObject {
    @Published private(set) var isAuthenticated: Bool
    @Published private(set) var role: UserRole?

    init() {
        isAuthenticated = Backend.supabase.auth.currentSession != nil
        role = Self.currentRole()
    }

    func signIn(email: String, password: String) async throws {
        _ = try await Backend.supabase.auth.signIn(email: email, password: password)
        refresh()
    }

    /// `role` = lab or collector; `displayName` = the lab's name or the person's name.
    /// Sent as user metadata so the `handle_new_user` trigger fills `profiles`.
    func signUp(email: String, password: String, displayName: String, role: UserRole) async throws {
        _ = try await Backend.supabase.auth.signUp(
            email: email,
            password: password,
            data: [
                "role": .string(role.rawValue),
                "display_name": .string(displayName),
                "full_name": .string(displayName),
                "name": .string(displayName),
            ]
        )
        refresh()
    }

    func signOut() async {
        try? await Backend.supabase.auth.signOut()
        refresh()
    }

    private func refresh() {
        isAuthenticated = Backend.supabase.auth.currentSession != nil
        role = Self.currentRole()
    }

    /// Reads the role from the signed-in user's metadata (set at sign-up, like the web).
    private static func currentRole() -> UserRole? {
        guard let meta = Backend.supabase.auth.currentUser?.userMetadata,
              case let .string(value)? = meta["role"] else { return nil }
        return UserRole(rawValue: value)
    }
}
