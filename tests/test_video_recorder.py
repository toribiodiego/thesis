"""
Tests for VideoRecorder component.

Verifies:
- Frame capture and storage
- MP4 video encoding
- Grayscale and RGB frame handling
- Video file creation and metadata
- Integration with evaluate() function
"""

import numpy as np
import pytest

from src.models import DQN
from src.training import VideoRecorder, evaluate

# Check if cv2 is available
try:
    import cv2  # noqa: F401

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ============================================================================
# VideoRecorder Tests
# ============================================================================


@pytest.mark.skipif(not CV2_AVAILABLE, reason="cv2 not installed")
def test_video_recorder_basic():
    """Test VideoRecorder captures and saves frames."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "test_video.mp4")
        recorder = VideoRecorder(video_path, fps=30, export_gif=False)

        # Capture some test frames
        for i in range(10):
            frame = np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)
            recorder.capture_frame(frame)

        # Save video
        info = recorder.save()

        # Check video was created
        assert info is not None
        assert info["video_path"] == video_path
        assert info["num_frames"] == 10
        assert info["fps"] == 30
        assert os.path.exists(video_path)


@pytest.mark.skipif(not CV2_AVAILABLE, reason="cv2 not installed")
def test_video_recorder_grayscale():
    """Test VideoRecorder handles grayscale frames."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "test_gray.mp4")
        recorder = VideoRecorder(video_path, fps=15)

        # Capture grayscale frames
        for i in range(5):
            frame = np.random.randint(0, 255, (84, 84), dtype=np.uint8)
            recorder.capture_frame(frame)

        info = recorder.save()
        assert info is not None
        assert os.path.exists(video_path)


@pytest.mark.skipif(not CV2_AVAILABLE, reason="cv2 not installed")
def test_video_recorder_float_frames():
    """Test VideoRecorder handles float32 frames."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "test_float.mp4")
        recorder = VideoRecorder(video_path, fps=30)

        # Capture float32 frames (normalized to [0, 1])
        for i in range(5):
            frame = np.random.rand(84, 84, 3).astype(np.float32)
            recorder.capture_frame(frame)

        info = recorder.save()
        assert info is not None
        assert info["num_frames"] == 5
        assert os.path.exists(video_path)


@pytest.mark.skipif(not CV2_AVAILABLE, reason="cv2 not installed")
def test_video_recorder_empty():
    """Test VideoRecorder handles empty frame list gracefully."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "test_empty.mp4")
        recorder = VideoRecorder(video_path, fps=30)

        # Save without capturing frames
        info = recorder.save()

        # Should return None for empty recorder
        assert info is None


@pytest.mark.skipif(not CV2_AVAILABLE, reason="cv2 not installed")
def test_video_recorder_directory_creation():
    """Test VideoRecorder creates output directory if needed."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create path with nested directories
        video_path = os.path.join(tmpdir, "nested", "dir", "test_video.mp4")
        recorder = VideoRecorder(video_path, fps=30)

        # Capture frame
        frame = np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)
        recorder.capture_frame(frame)

        # Save video
        recorder.save()

        # Check directory was created
        assert os.path.exists(os.path.dirname(video_path))
        assert os.path.exists(video_path)


@pytest.mark.skipif(not CV2_AVAILABLE, reason="cv2 not installed")
def test_video_recorder_different_fps():
    """Test VideoRecorder with different frame rates."""
    import os
    import tempfile

    fps_values = [15, 30, 60]

    for fps in fps_values:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, f"test_fps_{fps}.mp4")
            recorder = VideoRecorder(video_path, fps=fps)

            # Capture frames
            for i in range(10):
                frame = np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)
                recorder.capture_frame(frame)

            info = recorder.save()
            assert info["fps"] == fps
            assert os.path.exists(video_path)


@pytest.mark.skipif(not CV2_AVAILABLE, reason="cv2 not installed")
def test_video_recorder_different_resolutions():
    """Test VideoRecorder with different frame sizes."""
    import os
    import tempfile

    resolutions = [(84, 84), (210, 160), (128, 128)]

    for height, width in resolutions:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, f"test_{width}x{height}.mp4")
            recorder = VideoRecorder(video_path, fps=30)

            # Capture frames with this resolution
            for i in range(5):
                frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                recorder.capture_frame(frame)

            info = recorder.save()
            assert info is not None
            assert os.path.exists(video_path)


# ============================================================================
# Integration with evaluate() Tests
# ============================================================================


@pytest.mark.skipif(not CV2_AVAILABLE, reason="cv2 not installed")
def test_evaluate_with_video_recording():
    """Test evaluate records video when requested."""
    import os
    import tempfile
    from unittest.mock import Mock

    with tempfile.TemporaryDirectory() as tmpdir:
        env = Mock()
        env.action_space.n = 6
        env.unwrapped = Mock()
        env.unwrapped.spec = Mock()
        env.unwrapped.spec.id = "TestEnv-v0"

        dummy_state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
        dummy_frame = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
        env.reset.return_value = (dummy_state, {})
        env.render.return_value = dummy_frame

        step_count = [0]

        def mock_step(action):
            step_count[0] += 1
            done = step_count[0] >= 5
            if done:
                step_count[0] = 0
            return (dummy_state, 1.0, done, False, {})

        env.step.side_effect = mock_step
        model = DQN(num_actions=6)

        # Evaluate with video recording
        results = evaluate(
            env,
            model,
            num_episodes=2,
            eval_epsilon=0.05,
            device="cpu",
            record_video=True,
            video_dir=tmpdir,
            step=250000,
        )

        # Check video info is included
        assert "video_info" in results
        assert results["video_info"] is not None
        assert "video_path" in results["video_info"]
        assert "num_frames" in results["video_info"]

        # Check video file exists
        video_path = results["video_info"]["video_path"]
        assert os.path.exists(video_path)
        assert video_path.endswith(".mp4")


def test_evaluate_without_video():
    """Test evaluate omits video when not requested."""
    from unittest.mock import Mock

    env = Mock()
    env.action_space.n = 6
    dummy_state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
    env.reset.return_value = (dummy_state, {})

    step_count = [0]

    def mock_step(action):
        step_count[0] += 1
        done = step_count[0] >= 5
        if done:
            step_count[0] = 0
        return (dummy_state, 1.0, done, False, {})

    env.step.side_effect = mock_step
    model = DQN(num_actions=6)

    # Evaluate without video
    results = evaluate(
        env, model, num_episodes=1, eval_epsilon=0.05, device="cpu", record_video=False
    )

    # Check video info is NOT included
    assert "video_info" not in results


@pytest.mark.skipif(not CV2_AVAILABLE, reason="cv2 not installed")
def test_evaluate_video_only_first_episode():
    """Test evaluate records video only for first episode."""
    import tempfile
    from unittest.mock import Mock

    with tempfile.TemporaryDirectory() as tmpdir:
        env = Mock()
        env.action_space.n = 6
        env.unwrapped = Mock()
        env.unwrapped.spec = Mock()
        env.unwrapped.spec.id = "TestEnv-v0"

        dummy_state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
        dummy_frame = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
        env.reset.return_value = (dummy_state, {})
        env.render.return_value = dummy_frame

        step_count = [0]
        render_calls = [0]

        def mock_step(action):
            step_count[0] += 1
            done = step_count[0] >= 5
            if done:
                step_count[0] = 0
            return (dummy_state, 1.0, done, False, {})

        def mock_render():
            render_calls[0] += 1
            return dummy_frame

        env.step.side_effect = mock_step
        env.render.side_effect = mock_render
        model = DQN(num_actions=6)

        # Evaluate multiple episodes with video
        results = evaluate(
            env,
            model,
            num_episodes=3,  # 3 episodes
            eval_epsilon=0.05,
            device="cpu",
            record_video=True,
            video_dir=tmpdir,
            step=250000,
        )

        # Video should only be from first episode
        # (5 steps per episode × 1 episode = 5 render calls)
        # Plus possible reset renders
        # Check that video exists and has reasonable frame count
        assert "video_info" in results
        assert results["video_info"]["num_frames"] > 0
        # Should be less than would be from all 3 episodes
        assert results["video_info"]["num_frames"] < 15
