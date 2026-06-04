import SwiftUI

/// App color palette — mirrors the web app's dark theme (web/src/app/globals.css).
extension Color {
    init(hex: UInt) {
        self.init(.sRGB,
                  red: Double((hex >> 16) & 0xFF) / 255,
                  green: Double((hex >> 8) & 0xFF) / 255,
                  blue: Double(hex & 0xFF) / 255,
                  opacity: 1)
    }

    static let appBackground          = Color(hex: 0x0F0F0F)
    static let appSurface             = Color(hex: 0x151515)
    static let appSurfaceElevated     = Color(hex: 0x1B1B1B)
    static let appForeground          = Color(hex: 0xF5F7FB)
    static let appForegroundSecondary = Color(hex: 0xA4ADBA)
    static let appBorder              = Color.white.opacity(0.1)
    static let appAccent              = Color(hex: 0x3B5BDB) // lab blue (brand accent)
    static let appCollector           = Color(hex: 0x2F9E44) // green (collector)
    static let appAmber               = Color(hex: 0xD8A347)
    static let appDanger              = Color(hex: 0xD26464)
}
