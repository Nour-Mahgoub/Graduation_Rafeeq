#!/usr/bin/env python3

import threading
import numpy as np
import pyaudio
import librosa
import tensorflow as tf
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

SAMPLE_RATE = 16000
CHUNK = 1024


class SpeechRecognitionNode(Node):

    def __init__(self):
        super().__init__('rafeeq_speech_node')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('model_path', 'rafeeq_model.tflite')
        self.declare_parameter('labels_path', 'labels.txt')
        self.declare_parameter('volume_threshold', 0.06)
        self.declare_parameter('confidence_threshold', 0.70)
        self.declare_parameter('wake_word_threshold', 0.10)
        self.declare_parameter('duration', 1.5)

        model_path  = self.get_parameter('model_path').get_parameter_value().string_value
        labels_path = self.get_parameter('labels_path').get_parameter_value().string_value
        self.volume_threshold     = self.get_parameter('volume_threshold').get_parameter_value().double_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.wake_word_threshold  = self.get_parameter('wake_word_threshold').get_parameter_value().double_value
        self.duration             = self.get_parameter('duration').get_parameter_value().double_value
        self.samples_per_track    = int(SAMPLE_RATE * self.duration)

        # ── Publisher ────────────────────────────────────────────────────────
        self.nav_pub = self.create_publisher(String, 'navigation_goal', 10)

        # ── Load TFLite model ────────────────────────────────────────────────
        self.get_logger().info(f'Loading model from: {model_path}')
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # ── Load labels ──────────────────────────────────────────────────────
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f if line.strip()]
        self.get_logger().info(f'Loaded {len(self.labels)} commands: {self.labels}')

        # ── Start audio thread ───────────────────────────────────────────────
        self._running = True
        self._thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._thread.start()
        self.get_logger().info('Rafeeq speech node ready — listening for commands.')

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_rms(self, block: bytes) -> float:
        audio = np.frombuffer(block, dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(audio ** 2)))

    def _extract_features(self, audio_array: np.ndarray) -> np.ndarray:
        y, _ = librosa.effects.trim(audio_array, top_db=20)
        if len(y) > self.samples_per_track:
            y = y[:self.samples_per_track]
        else:
            y = np.pad(y, (0, self.samples_per_track - len(y)), 'constant')
        mfccs = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=13, n_fft=2048, hop_length=512)
        return np.float32(mfccs[np.newaxis, ..., np.newaxis])

    def _run_inference(self, audio_array: np.ndarray):
        features = self._extract_features(audio_array)
        self.interpreter.set_tensor(self.input_details[0]['index'], features)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        idx = int(np.argmax(output))
        return idx, float(output[idx]), self.labels[idx], output

    def _record_window(self, stream, duration_s: float) -> np.ndarray:
        """Record `duration_s` seconds and return a float32 audio array."""
        n_chunks = int(SAMPLE_RATE / CHUNK * duration_s)
        frames = [stream.read(CHUNK, exception_on_overflow=False) for _ in range(n_chunks)]
        raw = b''.join(frames)
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def _wait_for_voice(self, stream, timeout_s: float = 3.0):
        """
        Wait up to `timeout_s` seconds for a voice trigger.
        Returns a float32 audio array starting from the trigger chunk,
        or None if no voice is heard within the timeout.
        """
        n_timeout = int(SAMPLE_RATE / CHUNK * timeout_s)
        for _ in range(n_timeout):
            trigger_chunk = stream.read(CHUNK, exception_on_overflow=False)
            if self._get_rms(trigger_chunk) > self.volume_threshold:
                # Voice detected — capture the full window from this point
                n_window = int(SAMPLE_RATE / CHUNK * self.duration)
                window_frames = [trigger_chunk] + [
                    stream.read(CHUNK, exception_on_overflow=False)
                    for _ in range(n_window)
                ]
                raw = b''.join(window_frames)
                return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return None

    # ── Main audio loop ───────────────────────────────────────────────────────

    def _audio_loop(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        try:
            while self._running:
                # ════════════════════════════════════════════════════════════
                # STATE 1 — SLEEPING: wait indefinitely for wake word "rafeeq"
                # ════════════════════════════════════════════════════════════
                self.get_logger().info('Sleeping — say "rafeeq" to activate.')

                rafeeq_idx = self.labels.index('rafeeq')

                while self._running:
                    audio = self._wait_for_voice(stream, timeout_s=5.0)
                    if audio is None:
                        continue  # no sound — keep sleeping
                    idx, conf, word, scores = self._run_inference(audio)
                    rafeeq_score = float(scores[rafeeq_idx])
                    # Print all scores so we can tune the threshold
                    score_str = ', '.join(
                        f'{self.labels[i]}={scores[i]*100:.1f}%'
                        for i in np.argsort(scores)[::-1]
                    )
                    self.get_logger().info(f'[sleep] {score_str}')
                    if rafeeq_score >= self.wake_word_threshold:
                        self.get_logger().info(
                            f'Wake word detected! rafeeq={rafeeq_score*100:.1f}% — activated!'
                        )
                        break  # → AWAKE

                if not self._running:
                    break

                # ════════════════════════════════════════════════════════════
                # STATE 2 — RECORD COMMAND: immediately record a fixed window
                # ════════════════════════════════════════════════════════════
                self.get_logger().info('Recording command now — speak!')
                audio = self._record_window(stream, duration_s=self.duration)

                # ── Classify ──────────────────────────────────────────────
                idx, confidence, command, _ = self._run_inference(audio)
                self.get_logger().info(
                    f'Heard "{command}" ({confidence*100:.1f}%)'
                )

                # "stop" / "sleep" or low confidence → go back to sleep
                if command in ('stop', 'sleep'):
                    self.get_logger().info(f'"{command}" — going back to sleep.')
                    continue  # → SLEEPING

                if confidence < self.confidence_threshold:
                    self.get_logger().info(
                        f'Low confidence ({confidence*100:.1f}%) — going back to sleep.'
                    )
                    continue  # → SLEEPING

                # ════════════════════════════════════════════════════════════
                # STATE 3 — CONFIRM: wait for voice then verify
                # ════════════════════════════════════════════════════════════
                self.get_logger().warn(
                    f'Detected "{command}" ({confidence*100:.1f}%). '
                    f'Say it again to confirm (3 s)...'
                )

                confirm_audio = self._wait_for_voice(stream, timeout_s=3.0)

                if confirm_audio is None:
                    self.get_logger().warn(
                        f'No confirmation — "{command}" discarded, going back to sleep.'
                    )
                    continue  # → SLEEPING

                c_idx, c_conf, c_command, _ = self._run_inference(confirm_audio)

                if c_idx == idx and c_conf >= self.confidence_threshold:
                    self.get_logger().info(
                        f'CONFIRMED: "{command}" ({c_conf*100:.1f}%) '
                        f'— publishing to /navigation_goal'
                    )
                    msg = String()
                    msg.data = command
                    self.nav_pub.publish(msg)
                else:
                    self.get_logger().warn(
                        f'Mismatch: heard "{c_command}" ({c_conf*100:.1f}%) '
                        f'but expected "{command}" — discarded.'
                    )

                # Always return to sleep after one command cycle
                # → SLEEPING

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def destroy_node(self):
        self._running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SpeechRecognitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
