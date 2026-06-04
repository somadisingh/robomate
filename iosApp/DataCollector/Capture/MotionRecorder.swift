import Foundation
import CoreMotion

/// Streams fused device motion (IMU) to a JSON-Lines file at ~100 Hz while
/// recording. One JSON object per line, each stamped with the device-motion
/// timestamp so the cloud can align it to the video frames.
final class MotionRecorder {
    private let motion = CMMotionManager()
    private let queue = OperationQueue()
    private var handle: FileHandle?
    private(set) var sampleCount = 0

    func start(writingTo url: URL) {
        guard motion.isDeviceMotionAvailable else { return }
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try? FileHandle(forWritingTo: url)
        sampleCount = 0
        motion.deviceMotionUpdateInterval = 1.0 / 100.0
        queue.maxConcurrentOperationCount = 1
        motion.startDeviceMotionUpdates(to: queue) { [weak self] motion, _ in
            guard let self, let m = motion, let handle = self.handle else { return }
            let sample: [String: Double] = [
                "t":  m.timestamp,
                "ax": m.userAcceleration.x, "ay": m.userAcceleration.y, "az": m.userAcceleration.z,
                "gx": m.rotationRate.x,     "gy": m.rotationRate.y,     "gz": m.rotationRate.z,
                "mx": m.magneticField.field.x, "my": m.magneticField.field.y, "mz": m.magneticField.field.z,
                "qw": m.attitude.quaternion.w, "qx": m.attitude.quaternion.x,
                "qy": m.attitude.quaternion.y, "qz": m.attitude.quaternion.z,
            ]
            guard let data = try? JSONSerialization.data(withJSONObject: sample) else { return }
            handle.write(data)
            handle.write(Data("\n".utf8))
            self.sampleCount += 1
        }
    }

    func stop() {
        motion.stopDeviceMotionUpdates()
        try? handle?.close()
        handle = nil
    }
}
