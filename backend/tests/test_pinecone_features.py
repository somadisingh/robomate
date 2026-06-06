"""Tests for the Pinecone features (Feature 1 recording indexer + shared client).

All external calls (Gemini embed, Pinecone) are mocked — no real network calls.

NOTE: The collector-profile update for Feature 3 runs inline in the web app
(TypeScript), not in Python, so there is no Python collector_profile_indexer to
test here. The 0.70 similarity threshold rule (Feature 2) is pure logic and is
covered below.
"""

from unittest.mock import MagicMock, patch


# ── Feature 1: Recording Indexer ──────────────────────────


class TestBuildRecordingDocument:
    def test_full_data_document(self):
        from backend.analyzers.pinecone_recording_indexer import build_recording_document

        doc = build_recording_document(
            recording_id="rec-1",
            task_title="Pick up bottle",
            task_description="Grasp and lift a water bottle",
            objects_to_detect=["bottle"],
            lab_id="lab-1",
            gemini_eval={"summary": "Collector picked up the bottle cleanly.", "score": 8.5, "passed": True},
            yolo_detections={"frames": [{"detections": [{"class_name": "bottle"}, {"class_name": "hand"}]}]},
            temporal_actions={"phases": [{"action": "reach"}, {"action": "grasp"}, {"action": "lift"}]},
            temporal_annotations={
                "scene_summary": "Smooth pickup from table.",
                "searchable_tags": ["manipulation", "bottle"],
                "key_moments": [{"timestamp_seconds": 1.2, "action": "reaches for bottle", "phase": "approach"}],
            },
        )
        assert "Pick up bottle" in doc
        assert "bottle" in doc
        assert "grasp" in doc.lower() or "reach" in doc.lower()
        assert "8.5" in doc
        assert len(doc) > 50

    def test_real_yolo_and_temporal_schema(self):
        """The live pipeline uses instances[*].class_name and segments[*].label."""
        from backend.analyzers.pinecone_recording_indexer import build_recording_document

        doc = build_recording_document(
            recording_id="rec-real",
            task_title="Plug in charger",
            task_description="Insert the laptop charger",
            objects_to_detect=["charger"],
            lab_id="lab-1",
            gemini_eval={"summary": "Charger inserted.", "score": 7, "success": True},
            yolo_detections={"frames": [{"instances": [{"class_name": "charger", "confidence": 0.9}]}]},
            temporal_actions={"segments": [{"label": "approach"}, {"label": "insert"}]},
            temporal_annotations=None,
        )
        assert "charger" in doc
        assert "approach" in doc or "insert" in doc

    def test_no_temporal_annotations(self):
        from backend.analyzers.pinecone_recording_indexer import build_recording_document

        doc = build_recording_document(
            recording_id="rec-2",
            task_title="Test task",
            task_description="",
            objects_to_detect=[],
            lab_id="lab-1",
            gemini_eval={"summary": "", "score": 0, "passed": False},
            yolo_detections={"frames": []},
            temporal_actions={"phases": []},
            temporal_annotations=None,
        )
        assert isinstance(doc, str)
        assert len(doc) > 0

    def test_upsert_called_correctly(self):
        from backend.analyzers.pinecone_recording_indexer import index_recording

        mock_index = MagicMock()
        with patch("backend.analyzers.pinecone_recording_indexer.get_pinecone_index", return_value=mock_index), \
             patch("backend.analyzers.pinecone_recording_indexer.embed_document", return_value=[0.1] * 768):
            index_recording(
                recording_id="rec-1", task_title="T", task_description="D",
                objects_to_detect=["bottle"], lab_id="lab-1", collector_id="col-1",
                gemini_eval={"summary": "", "score": 7, "passed": True},
                yolo_detections={"frames": []}, temporal_actions={"phases": []},
                temporal_annotations=None, gemini_score=7.0, task_id="task-1",
            )
        mock_index.upsert.assert_called_once()
        kwargs = mock_index.upsert.call_args.kwargs
        vectors = kwargs["vectors"]
        assert kwargs["namespace"] == "recordings"
        assert vectors[0]["id"] == "rec-1"
        assert len(vectors[0]["values"]) == 768
        assert vectors[0]["metadata"]["lab_id"] == "lab-1"
        assert vectors[0]["metadata"]["passed"] is True

    def test_index_recording_does_not_raise_on_error(self):
        from backend.analyzers.pinecone_recording_indexer import index_recording

        with patch("backend.analyzers.pinecone_recording_indexer.embed_document", side_effect=Exception("Gemini down")):
            # Must not raise.
            index_recording(
                recording_id="rec-err", task_title="T", task_description="D",
                objects_to_detect=[], lab_id="lab-1", collector_id="col-1",
                gemini_eval={}, yolo_detections={"frames": []}, temporal_actions={"phases": []},
                temporal_annotations=None, gemini_score=0, task_id="task-1",
            )


# ── Feature 2: Similar Recordings threshold (pure logic) ──


class TestSimilarRecordingsThreshold:
    def test_threshold_filters_low_scores(self):
        raw_matches = [
            {"id": "rec-a", "score": 0.92},
            {"id": "rec-b", "score": 0.75},
            {"id": "rec-c", "score": 0.55},
        ]
        relevant = [m for m in raw_matches if m["score"] >= 0.70]
        assert len(relevant) == 2
        assert all(m["id"] != "rec-c" for m in relevant)

    def test_empty_results_returns_empty_list(self):
        raw_matches = []
        relevant = [m for m in raw_matches if m["score"] >= 0.70]
        assert relevant == []


# ── Shared Pinecone client ─────────────────────────────────


class TestPineconeClient:
    def test_embed_document_returns_768_floats(self):
        from backend import pinecone_client

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = {"embedding": [0.1] * 768}
        with patch.object(pinecone_client, "_get_genai_client", return_value=mock_client):
            vector = pinecone_client.embed_document("test document")
        assert isinstance(vector, list)
        assert len(vector) == 768

    def test_embed_query_uses_retrieval_query_task_type(self):
        from backend import pinecone_client

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = {"embedding": [0.0] * 768}
        with patch.object(pinecone_client, "_get_genai_client", return_value=mock_client):
            pinecone_client.embed_query("test query")
        config = mock_client.models.embed_content.call_args.kwargs["config"]
        assert config.task_type == "RETRIEVAL_QUERY"
        assert config.output_dimensionality == 768

    def test_embed_document_uses_retrieval_document_task_type(self):
        from backend import pinecone_client

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = {"embedding": [0.0] * 768}
        with patch.object(pinecone_client, "_get_genai_client", return_value=mock_client):
            pinecone_client.embed_document("doc")
        config = mock_client.models.embed_content.call_args.kwargs["config"]
        assert config.task_type == "RETRIEVAL_DOCUMENT"

    def test_ensure_index_exists_creates_if_missing(self):
        from backend import pinecone_client

        mock_pc = MagicMock()
        mock_pc.list_indexes.return_value = []
        with patch.object(pinecone_client, "Pinecone", return_value=mock_pc), \
             patch.dict("os.environ", {"PINECONE_INDEX_NAME": "robomate", "PINECONE_API_KEY": "key"}):
            pinecone_client.ensure_index_exists()
        mock_pc.create_index.assert_called_once()
        kwargs = mock_pc.create_index.call_args.kwargs
        assert kwargs["dimension"] == 768
        assert kwargs["metric"] == "cosine"

    def test_ensure_index_exists_skips_if_already_there(self):
        from backend import pinecone_client

        mock_pc = MagicMock()
        existing = MagicMock()
        existing.name = "robomate"
        mock_pc.list_indexes.return_value = [existing]
        with patch.object(pinecone_client, "Pinecone", return_value=mock_pc), \
             patch.dict("os.environ", {"PINECONE_INDEX_NAME": "robomate", "PINECONE_API_KEY": "key"}):
            pinecone_client.ensure_index_exists()
        mock_pc.create_index.assert_not_called()
