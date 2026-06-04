import Foundation
import ARKit
import AVFoundation
import CoreImage
import QuartzCore
import UIKit
import simd

/// Orchestrates a single recording using ARKit as the capture source:
/// RGB video (encoded to .mp4 via AVAssetWriter) + 6DoF camera pose + camera
/// intrinsics + IMU + GPS, all into one bundle on disk.
///
/// ARKit's `ARWorldTrackingConfiguration` fuses camera + IMU (visual-inertial
/// odometry) to give a ground-truth camera trajectory — the data differentiator.
final class Recorder: NSObject, ObservableObject {
    enum Phase: Equatable { case idle, ready, recording, finishing, denied }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var elapsed: TimeInterval = 0
    @Published var live = LiveMetrics()

    /// Real-time capture signals for on-device coaching (preview + recording).
    struct LiveMetrics {
        var luminance: Double = 0.5      // 0...1 average brightness
        var motionDegPerSec: Double = 0  // camera rotational speed
        var distanceM: Double? = nil     // LiDAR depth at center
        var hasData = false
    }

    private weak var arSession: ARSession?
    private let delegateQueue = DispatchQueue(label: "recorder.ar.delegate")
    private let motion = MotionRecorder()
    private let location = LocationProvider()

    // Video writer (set up lazily on the first frame, once we know its size).
    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var audioInput: AVAssetWriterInput?
    private var pixelAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var writerStarted = false
    private var videoCodec = "h264"

    // Per-frame pose stream + one-shot intrinsics.
    private var poseHandle: FileHandle?
    private var wroteIntrinsics = false

    // LiDAR depth stream (throttled to keep file size sane).
    private var depthHandle: FileHandle?
    private var lastDepthWrite: TimeInterval = 0
    private var depthWidth = 0
    private var depthHeight = 0
    private let depthInterval: TimeInterval = 1.0 / 10.0   // ~10 fps

    // Live coaching metrics (preview + recording).
    private var lastLiveUpdate: TimeInterval = 0
    private var prevTransform: simd_float4x4?
    private var prevTransformTime: TimeInterval = 0
    private var motionEMA: Double = 0

    // Optional JPEG frame feed for semantic coaching (e.g. Gemini Live), ~1 fps.
    // Conversion runs on its own queue with drop-if-busy so it never stalls the
    // AR delegate queue (which would make ARKit retain frames and freeze).
    private let ciContext = CIContext(options: nil)
    private let coachingQueue = DispatchQueue(label: "recorder.coaching.jpeg", qos: .utility)
    private var lastCoachingEmit: TimeInterval = 0
    private var coachingBusy = false   // touched only on delegateQueue
    var onCoachingFrame: ((Data) -> Void)?

    private var bundle: RecordingBundle?
    private var startWall = Date()
    private var startMono: TimeInterval = 0
    private var firstFrameTime: TimeInterval?
    private var lastFrameTime: TimeInterval = 0
    private var isRecording = false
    private var timer: Timer?

    private let deviceInfo: RecordingMetadata.Device = {
        let d = UIDevice.current
        return .init(model: d.model, systemName: d.systemName, systemVersion: d.systemVersion)
    }()

    /// Invoked on the main thread once a recording is fully written to disk.
    var onFinished: ((Recording) -> Void)?

    /// The job/bounty this recording fulfils. Set before recording starts.
    var bountyId: String?

    private func setPhase(_ newPhase: Phase) {
        DispatchQueue.main.async { self.phase = newPhase }
    }

    // MARK: - Setup

    func configureIfNeeded() {
        guard phase == .idle else { return }
        location.requestPermission()
        AVAudioApplication.requestRecordPermission { _ in }   // mic, for audio + transcript
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            setPhase(.ready)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                self?.setPhase(granted ? .ready : .denied)
            }
        default:
            setPhase(.denied)
        }
    }

    /// Called by the preview (`CameraPreviewView`) with the ARSCNView's session.
    /// We become its delegate to receive frames, and start world tracking.
    func attach(to session: ARSession) {
        guard arSession !== session else { return }
        arSession = session
        session.delegate = self
        session.delegateQueue = delegateQueue
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        // Smaller files: prefer the lowest frame-rate, then lowest-resolution format.
        if let format = ARWorldTrackingConfiguration.supportedVideoFormats.min(by: {
            $0.framesPerSecond != $1.framesPerSecond
                ? $0.framesPerSecond < $1.framesPerSecond
                : $0.imageResolution.width * $0.imageResolution.height
                    < $1.imageResolution.width * $1.imageResolution.height
        }) {
            config.videoFormat = format
        }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth) // LiDAR depth (Pro devices)
        }
        config.providesAudioData = true   // mic audio → video track + transcript
        session.run(config)
    }

    /// Pause the AR session (e.g. when leaving the record screen) to free the camera.
    func pause() {
        arSession?.pause()
    }

    // MARK: - Record / stop

    func toggle() {
        switch phase {
        case .ready:     start()
        case .recording: stop()
        default:         break
        }
    }

    private func start() {
        guard let bundle = try? RecordingStore.makeBundle() else { return }
        self.bundle = bundle
        startWall = Date()
        startMono = CACurrentMediaTime()
        firstFrameTime = nil
        lastFrameTime = 0
        wroteIntrinsics = false
        writerStarted = false
        depthWidth = 0
        depthHeight = 0
        lastDepthWrite = 0
        elapsed = 0

        FileManager.default.createFile(atPath: bundle.posesURL.path, contents: nil)
        poseHandle = try? FileHandle(forWritingTo: bundle.posesURL)

        location.start()
        motion.start(writingTo: bundle.imuURL)
        isRecording = true
        setPhase(.recording)

        // Timer is scheduled on the main run loop, so it fires on the main thread.
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self else { return }
            self.elapsed = CACurrentMediaTime() - self.startMono
        }
    }

    private func stop() {
        guard isRecording else { return }
        isRecording = false
        setPhase(.finishing)
        timer?.invalidate(); timer = nil
        motion.stop()
        location.stop()

        delegateQueue.async { [weak self] in
            guard let self else { return }
            try? self.poseHandle?.close()
            self.poseHandle = nil
            try? self.depthHandle?.close()
            self.depthHandle = nil

            guard let writer = self.assetWriter, let input = self.videoInput, self.writerStarted else {
                self.completeRecording()
                return
            }
            input.markAsFinished()
            self.audioInput?.markAsFinished()
            writer.finishWriting { [weak self] in
                self?.completeRecording()
            }
        }
    }

    // MARK: - Finalize

    private func completeRecording() {
        assetWriter = nil; videoInput = nil; audioInput = nil; pixelAdaptor = nil; writerStarted = false
        guard let bundle else { setPhase(.ready); return }

        let durationMs = Int((lastFrameTime - (firstFrameTime ?? lastFrameTime)) * 1000)
        let gps = location.lastLocation.map {
            RecordingMetadata.GPS(lat: $0.coordinate.latitude,
                                  lon: $0.coordinate.longitude,
                                  accuracyM: $0.horizontalAccuracy)
        }
        // Only list files that actually got written — plus transcript.json, which
        // Whisper produces asynchronously below (the uploader skips it if absent).
        var streams = bundle.streamFilenames.filter {
            FileManager.default.fileExists(atPath: bundle.folderURL.appendingPathComponent($0).path)
        }
        if !streams.contains("transcript.json") { streams.append("transcript.json") }
        let metadata = RecordingMetadata(
            recordingId: bundle.id.uuidString,
            bountyId: bountyId,
            startedAt: ISO8601DateFormatter().string(from: startWall),
            startMonotonic: startMono,
            durationMs: durationMs,
            device: deviceInfo,
            video: .init(file: bundle.videoURL.lastPathComponent, container: "mp4", codec: videoCodec),
            gps: gps,
            depth: depthWidth > 0
                ? .init(file: bundle.depthURL.lastPathComponent,
                        width: depthWidth, height: depthHeight, dtype: "float32",
                        layout: "per-frame: float64 timestamp (LE) then width*height row-major float32 metres")
                : nil,
            streams: streams,
            schemaVersion: 3
        )
        try? RecordingStore.writeMetadata(metadata, to: bundle.metadataURL)
        RecordingStore.excludeFromBackup(bundle.folderURL)

        let size = RecordingStore.directorySize(bundle.folderURL)
        let id = bundle.id
        let createdAt = startWall
        let videoURL = bundle.videoURL
        let transcriptURL = bundle.transcriptURL
        // Re-read streams now that metadata.json exists (+ transcript.json, async).
        var finalStreams = bundle.streamFilenames.filter {
            FileManager.default.fileExists(atPath: bundle.folderURL.appendingPathComponent($0).path)
        }
        if !finalStreams.contains("transcript.json") { finalStreams.append("transcript.json") }
        self.bundle = nil

        // On-device Whisper transcription — runs now that the camera is off, so no
        // camera/GPU conflict. Writes transcript.json when done (first run downloads
        // the model, so it can take a bit).
        Task.detached(priority: .utility) {
            // Keep running ~30s even if the user leaves the app (e.g. opens Files).
            let bg = await MainActor.run { UIApplication.shared.beginBackgroundTask(withName: "whisper") }
            let (segments, status) = await WhisperTranscriber.transcribe(videoURL: videoURL)
            try? RecordingStore.writeTranscript(segments, status: status, to: transcriptURL)
            await MainActor.run { if bg != .invalid { UIApplication.shared.endBackgroundTask(bg) } }
        }

        DispatchQueue.main.async {
            let recording = Recording(
                id: id,
                bountyId: self.bountyId,
                createdAt: createdAt,
                durationMs: durationMs,
                sizeBytes: size,
                gpsLat: gps?.lat,
                gpsLon: gps?.lon,
                gpsAccuracyM: gps?.accuracyM,
                folderName: id.uuidString,
                streams: finalStreams,
                status: .local
            )
            self.onFinished?(recording)
            self.phase = .ready
        }
    }

    // MARK: - Per-frame writers

    private func setupWriter(for pixelBuffer: CVPixelBuffer, to url: URL, startTime: TimeInterval) {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard let writer = try? AVAssetWriter(outputURL: url, fileType: .mp4) else { return }
        writer.shouldOptimizeForNetworkUse = true // moov atom at front → web-streamable

        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264, // universal browser playback (Chrome/Firefox can't do HEVC)
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: 6_000_000, // ~6 Mbps cap → smaller video
            ],
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = true
        input.transform = .identity // record in landscape (record screen is locked to landscape)

        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: nil)
        if writer.canAdd(input) { writer.add(input) }

        // Audio track (AAC) so playback has sound. The writer transcodes ARKit's
        // mic audio to AAC.
        let audioSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVNumberOfChannelsKey: 1,
            AVSampleRateKey: 44_100.0,
            AVEncoderBitRateKey: 64_000,
        ]
        let audio = AVAssetWriterInput(mediaType: .audio, outputSettings: audioSettings)
        audio.expectsMediaDataInRealTime = true
        if writer.canAdd(audio) { writer.add(audio) }

        writer.startWriting()
        writer.startSession(atSourceTime: CMTime(seconds: startTime, preferredTimescale: 1_000_000))

        assetWriter = writer
        videoInput = input
        audioInput = audio
        pixelAdaptor = adaptor
        firstFrameTime = startTime
        writerStarted = (writer.status == .writing)
    }

    private func appendPose(_ frame: ARFrame, at t: TimeInterval) {
        guard let handle = poseHandle else { return }
        let m = frame.camera.transform
        let p = m.columns.3
        let q = simd_quatf(m)
        let row: [String: Double] = [
            "t": t,
            "px": Double(p.x), "py": Double(p.y), "pz": Double(p.z),
            "qw": Double(q.real), "qx": Double(q.imag.x), "qy": Double(q.imag.y), "qz": Double(q.imag.z),
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: row) else { return }
        handle.write(data)
        handle.write(Data("\n".utf8))
    }

    /// Appends one LiDAR depth frame to depth.bin:
    /// [Float64 timestamp (LE)] + [width*height row-major Float32 metres].
    // MARK: - Live coaching metrics

    private func updateLiveMetrics(_ frame: ARFrame) {
        let t = frame.timestamp
        if let prev = prevTransform, t > prevTransformTime {
            let q1 = simd_quatf(prev)
            let q2 = simd_quatf(frame.camera.transform)
            let dot = Double(min(1, abs(simd_dot(q1.vector, q2.vector))))
            let deg = 2.0 * acos(dot) * 180.0 / Double.pi
            let dt = t - prevTransformTime
            motionEMA = motionEMA * 0.8 + (deg / dt) * 0.2
        }
        prevTransform = frame.camera.transform
        prevTransformTime = t

        guard t - lastLiveUpdate >= 0.25 else { return }   // ~4 Hz UI updates
        lastLiveUpdate = t

        let lum = averageLuminance(frame.capturedImage)
        let dist = centerDepth(frame)
        let motion = motionEMA
        DispatchQueue.main.async {
            self.live = LiveMetrics(luminance: lum, motionDegPerSec: motion, distanceM: dist, hasData: true)
        }
    }

    /// Emit a downscaled JPEG frame for a semantic coach (throttled to ~1 fps).
    /// Runs on the AR delegate queue, so the conversion stays off the main thread.
    private func emitCoachingFrame(_ frame: ARFrame) {
        guard let consumer = onCoachingFrame else { return }
        let t = frame.timestamp
        guard t - lastCoachingEmit >= 1.0, !coachingBusy else { return }   // 1 fps, skip if still converting
        lastCoachingEmit = t
        coachingBusy = true
        // Retain just the image buffer (not the whole ARFrame) and convert off-queue.
        let pixelBuffer = frame.capturedImage
        coachingQueue.async { [weak self] in
            guard let self else { return }
            let data = self.jpeg(from: pixelBuffer, maxWidth: 640)
            self.delegateQueue.async { self.coachingBusy = false }
            if let data { consumer(data) }
        }
    }

    private func jpeg(from pixelBuffer: CVPixelBuffer, maxWidth: CGFloat) -> Data? {
        var image = CIImage(cvPixelBuffer: pixelBuffer)
        let width = image.extent.width
        if width > maxWidth {
            let scale = maxWidth / width
            image = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        }
        guard let cg = ciContext.createCGImage(image, from: image.extent) else { return nil }
        return UIImage(cgImage: cg).jpegData(compressionQuality: 0.6)
    }

    /// Average luma of the Y plane (sampled), 0...1.
    private func averageLuminance(_ pixelBuffer: CVPixelBuffer) -> Double {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) else { return 0.5 }
        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
        let bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
        let ptr = base.assumingMemoryBound(to: UInt8.self)
        let stepX = max(1, width / 24), stepY = max(1, height / 24)
        var sum = 0, count = 0
        for y in stride(from: 0, to: height, by: stepY) {
            let row = ptr + y * bytesPerRow
            for x in stride(from: 0, to: width, by: stepX) {
                sum += Int(row[x]); count += 1
            }
        }
        return count > 0 ? Double(sum) / Double(count) / 255.0 : 0.5
    }

    /// LiDAR depth (metres) at the center of the frame, if available.
    private func centerDepth(_ frame: ARFrame) -> Double? {
        guard let depth = frame.sceneDepth?.depthMap else { return nil }
        CVPixelBufferLockBaseAddress(depth, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
        let width = CVPixelBufferGetWidth(depth)
        let height = CVPixelBufferGetHeight(depth)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(depth)
        guard let base = CVPixelBufferGetBaseAddress(depth) else { return nil }
        let row = base.advanced(by: (height / 2) * bytesPerRow).assumingMemoryBound(to: Float32.self)
        let value = row[width / 2]
        return value.isFinite && value > 0 ? Double(value) : nil
    }

    private func appendDepth(_ depthMap: CVPixelBuffer, at t: TimeInterval, to url: URL) {
        if depthHandle == nil {
            FileManager.default.createFile(atPath: url.path, contents: nil)
            depthHandle = try? FileHandle(forWritingTo: url)
        }
        guard let handle = depthHandle else { return }
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthMap)
        guard let base = CVPixelBufferGetBaseAddress(depthMap) else { return }
        depthWidth = width
        depthHeight = height

        let rowBytes = width * MemoryLayout<Float32>.size
        var data = Data(capacity: MemoryLayout<Double>.size + rowBytes * height)
        var ts = t
        withUnsafeBytes(of: &ts) { data.append(contentsOf: $0) }
        for row in 0..<height {
            data.append(Data(bytes: base.advanced(by: row * bytesPerRow), count: rowBytes))
        }
        handle.write(data)
    }

    private func writeIntrinsics(_ frame: ARFrame, to url: URL) {
        let k = frame.camera.intrinsics
        let res = frame.camera.imageResolution
        let dict: [String: Double] = [
            "fx": Double(k.columns.0.x),
            "fy": Double(k.columns.1.y),
            "cx": Double(k.columns.2.x),
            "cy": Double(k.columns.2.y),
            "width": Double(res.width),
            "height": Double(res.height),
        ]
        if let data = try? JSONSerialization.data(withJSONObject: dict, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: url)
        }
    }
}

extension Recorder: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        updateLiveMetrics(frame)
        emitCoachingFrame(frame)
        guard isRecording, let bundle else { return }
        let t = frame.timestamp
        lastFrameTime = t

        if !wroteIntrinsics {
            wroteIntrinsics = true
            writeIntrinsics(frame, to: bundle.intrinsicsURL)
        }
        if assetWriter == nil {
            setupWriter(for: frame.capturedImage, to: bundle.videoURL, startTime: t)
        }
        if writerStarted, let adaptor = pixelAdaptor, let input = videoInput, input.isReadyForMoreMediaData {
            adaptor.append(frame.capturedImage,
                           withPresentationTime: CMTime(seconds: t, preferredTimescale: 1_000_000))
        }
        appendPose(frame, at: t)

        if let sceneDepth = frame.sceneDepth, t - lastDepthWrite >= depthInterval {
            lastDepthWrite = t
            appendDepth(sceneDepth.depthMap, at: t, to: bundle.depthURL)
        }
    }

    /// ARKit's captured microphone audio → the video's audio track (Whisper later
    /// transcribes that audio post-recording).
    func session(_ session: ARSession, didOutputAudioSampleBuffer audioSampleBuffer: CMSampleBuffer) {
        guard isRecording, writerStarted,
              let audioInput, audioInput.isReadyForMoreMediaData else { return }
        audioInput.append(audioSampleBuffer)
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        setPhase(.denied)
    }
}
