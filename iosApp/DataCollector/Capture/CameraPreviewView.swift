import SwiftUI
import ARKit

/// Live ARKit camera preview. Hands its `ARSession` to the recorder, which then
/// becomes the session delegate to capture frames, 6DoF pose and intrinsics.
struct CameraPreviewView: UIViewRepresentable {
    let recorder: Recorder

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.automaticallyUpdatesLighting = true
        recorder.attach(to: view.session)
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}
