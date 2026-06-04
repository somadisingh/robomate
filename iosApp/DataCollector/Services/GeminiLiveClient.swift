import Foundation

/// Minimal client for the Gemini Live API (BidiGenerateContent over WebSocket).
/// Streams JPEG frames as realtime video input and surfaces the model's short
/// text "coaching tips". Everything stays on the main actor so the UI can bind
/// directly; the WebSocket callbacks hop back to main.
@MainActor
final class GeminiLiveClient: ObservableObject {
    enum Status: Equatable {
        case idle, connecting, ready, closed
        case error(String)
    }

    @Published private(set) var status: Status = .idle
    @Published private(set) var latestTip: String = ""
    @Published private(set) var log: [String] = []   // raw events, for debugging

    var isReady: Bool { if case .ready = status { return true }; return false }

    private var socket: URLSessionWebSocketTask?
    private var session: URLSession?
    private var partial = ""          // accumulates a turn's text (streamed to latestTip)
    private var isOpen = false
    private var awaitingSince: Date?  // set when a tip is requested, cleared on turnComplete
    private var frameInFlight = false // drop frames while a send is still pending
    private var pendingSystemInstruction: String?   // sent once the socket opens
    private let wsDelegate = WSDelegate()

    init() { wsDelegate.client = self }

    // MARK: - Lifecycle

    func connect(systemInstruction: String) {
        let key = SupabaseConfig.geminiAPIKey
        guard !key.isEmpty else {
            status = .error("No Gemini key. Paste one into SupabaseConfig.swift (geminiAPIKey).")
            return
        }
        status = .connecting
        partial = ""
        latestTip = ""

        let host = "generativelanguage.googleapis.com"
        let path = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        guard let url = URL(string: "wss://\(host)\(path)?key=\(key)") else {
            status = .error("Bad WebSocket URL"); return
        }
        pendingSystemInstruction = systemInstruction
        let session = URLSession(configuration: .default, delegate: wsDelegate, delegateQueue: nil)
        let socket = session.webSocketTask(with: url)
        self.session = session
        self.socket = socket
        isOpen = true
        socket.resume()
        addLog("connecting…")
        // receiveLoop + setup start once the socket actually opens (see socketDidOpen).
    }

    // MARK: - Connection events (called by WSDelegate)

    func socketDidOpen() {
        addLog("ws open")
        startReceiving()
        if let si = pendingSystemInstruction {
            sendSetup(systemInstruction: si)
            pendingSystemInstruction = nil
        }
    }

    func socketDidClose(code: Int, reason: String) {
        addLog("ws closed code=\(code)\(reason.isEmpty ? "" : " reason=\(reason)")")
        if isOpen { status = .error("closed (code \(code))\(reason.isEmpty ? "" : ": \(reason)")") }
    }

    func taskDidComplete(status code: Int?, errorText: String?) {
        if let code { addLog("http \(code)") }
        if let errorText { addLog("task error: \(errorText)") }
        if isOpen, let code, code >= 400 { status = .error("handshake HTTP \(code)") }
    }

    func disconnect() {
        guard isOpen else { return }
        isOpen = false
        socket?.cancel(with: .goingAway, reason: nil)
        socket = nil
        session?.invalidateAndCancel()
        session = nil
        awaitingSince = nil
        frameInFlight = false
        status = .closed
        addLog("closed")
    }

    // MARK: - Sending

    private func sendSetup(systemInstruction: String) {
        // The available Live models only output AUDIO, so we request AUDIO and turn
        // on output transcription to get the spoken tip back as text (audio ignored).
        let setup: [String: Any] = [
            "setup": [
                "model": SupabaseConfig.geminiLiveModel,
                "generationConfig": ["responseModalities": ["AUDIO"]],
                "systemInstruction": ["parts": [["text": systemInstruction]]],
                "outputAudioTranscription": [String: String](),
            ]
        ]
        send(setup, label: "setup")
    }

    /// Send one downscaled JPEG frame as realtime video input (keep to ≤1 FPS).
    /// Drops the frame if a previous send hasn't completed, so a slow uplink can't
    /// pile up requests (the cause of "request timed out").
    func sendFrame(_ jpeg: Data) {
        guard isOpen, isReady, !frameInFlight, let socket else { return }
        let payload: [String: Any] = [
            "realtimeInput": ["video": ["data": jpeg.base64EncodedString(), "mimeType": "image/jpeg"]]
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload),
              let text = String(data: data, encoding: .utf8) else { return }
        frameInFlight = true
        socket.send(.string(text)) { [weak self] error in
            Task { @MainActor in
                self?.frameInFlight = false
                if let error { self?.addLog("frame send err: \(error.localizedDescription)") }
            }
        }
    }

    /// Nudge the model to produce one short tip from the frames seen so far — but
    /// only if we're not already waiting on a turn (with a 6s safety cap so a
    /// dropped turn can't wedge it). Keeps cadence tight without overlapping.
    func requestTip(_ prompt: String) {
        guard isOpen, isReady else { return }
        if let since = awaitingSince, Date().timeIntervalSince(since) < 6 { return }
        awaitingSince = Date()
        send(["realtimeInput": ["text": prompt]], label: "ask")
    }

    private func send(_ json: [String: Any], label: String?) {
        guard let socket,
              let data = try? JSONSerialization.data(withJSONObject: json),
              let text = String(data: data, encoding: .utf8) else { return }
        socket.send(.string(text)) { [weak self] error in
            guard let error else {
                if let label { Task { @MainActor in self?.addLog("→ \(label)") } }
                return
            }
            Task { @MainActor in self?.addLog("send error: \(error.localizedDescription)") }
        }
    }

    // MARK: - Receiving

    /// Small, Sendable result of parsing one server message off the main thread.
    private struct Parsed: Sendable {
        var setupComplete = false
        var text = ""
        var turnComplete = false
        var note: String?
        /// Worth hopping to main for? Empty audio chunks are not.
        var hasContent: Bool { setupComplete || !text.isEmpty || turnComplete || note != nil }
    }

    /// Parse a server message OFF the main actor. AUDIO output streams a flood of
    /// large messages, so doing this JSON work on main would stall the camera/UI.
    nonisolated private static func parse(_ data: Data) -> Parsed {
        var out = Parsed()
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            out.note = "← non-JSON (\(data.count)b)"
            return out
        }
        if obj["setupComplete"] != nil { out.setupComplete = true; return out }
        if let server = obj["serverContent"] as? [String: Any] {
            if let trans = server["outputTranscription"] as? [String: Any],
               let text = trans["text"] as? String { out.text += text }
            if let modelTurn = server["modelTurn"] as? [String: Any],
               let parts = modelTurn["parts"] as? [[String: Any]] {
                for part in parts { if let text = part["text"] as? String { out.text += text } }
            }
            if (server["turnComplete"] as? Bool) == true { out.turnComplete = true }
            return out
        }
        out.note = "← \(String(data: data, encoding: .utf8)?.prefix(180) ?? "?")"
        return out
    }

    private func startReceiving() {
        guard let socket else { return }
        receiveNext(on: socket)
    }

    /// Runs entirely on URLSession's background queue and re-arms itself there, so a
    /// flood of (audio) messages never saturates the main actor — which during
    /// recording is busy with the camera pipeline. Only meaningful updates hop to main.
    nonisolated private func receiveNext(on socket: URLSessionWebSocketTask) {
        socket.receive { [weak self] result in
            switch result {
            case .failure(let error):
                let desc = error.localizedDescription
                Task { @MainActor in self?.handleRecvError(desc) }   // stop (no re-arm)
            case .success(let message):
                let data: Data
                switch message {
                case .data(let d):   data = d
                case .string(let s): data = Data(s.utf8)
                @unknown default:    data = Data()
                }
                let parsed = GeminiLiveClient.parse(data)
                if parsed.hasContent { Task { @MainActor in self?.apply(parsed) } }
                self?.receiveNext(on: socket)   // re-arm on this background queue
            }
        }
    }

    private func handleRecvError(_ desc: String) {
        addLog("recv error: \(desc)")
        awaitingSince = nil
        if isOpen { status = .error(desc) }
    }

    private func apply(_ p: Parsed) {
        if p.setupComplete {
            status = .ready
            addLog("ready")
            return
        }
        if !p.text.isEmpty {
            partial += p.text
            let s = partial.trimmingCharacters(in: .whitespacesAndNewlines)
            if !s.isEmpty { latestTip = s }   // render as it arrives
        }
        if p.turnComplete {
            let tip = partial.trimmingCharacters(in: .whitespacesAndNewlines)
            if !tip.isEmpty { addLog("tip: \(tip)") }
            partial = ""
            awaitingSince = nil
        }
        if let note = p.note { addLog(note) }
    }

    private func addLog(_ line: String) {
        log.append(line)
        if log.count > 50 { log.removeFirst(log.count - 50) }
    }
}

/// Captures WebSocket lifecycle so handshake failures surface a real reason
/// (HTTP status, close code) instead of a bare "socket is not connected".
private final class WSDelegate: NSObject, URLSessionWebSocketDelegate {
    weak var client: GeminiLiveClient?

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didOpenWithProtocol proto: String?) {
        let c = client
        Task { @MainActor in c?.socketDidOpen() }
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        let c = client
        let code = closeCode.rawValue
        let text = reason.flatMap { String(data: $0, encoding: .utf8) } ?? ""
        Task { @MainActor in c?.socketDidClose(code: code, reason: text) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        let c = client
        let code = (task.response as? HTTPURLResponse)?.statusCode
        let desc = error?.localizedDescription
        Task { @MainActor in c?.taskDidComplete(status: code, errorText: desc) }
    }
}
