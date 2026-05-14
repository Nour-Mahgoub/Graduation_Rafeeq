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
        self.declare_parameter('volume_threshold', 0.02)      # matches working main.py (was 0.04)
        self.declare_parameter('confidence_threshold', 0.70)  # matches working main.py
        self.declare_parameter('wake_word_threshold', 0.50)   # min confidence to accept "rafeeq"
        self.declare_parameter('duration', 1.5)               # MUST stay 1.5 — model trained on this
        self.declare_parameter('confirm_timeout', 5.0)        # seconds to wait for confirmation

        model_path  = self.get_parameter('model_path').get_parameter_value().string_value
        labels_path = self.get_parameter('labels_path').get_parameter_value().string_value
        self.volume_threshold     = self.get_parameter('volume_threshold').get_parameter_value().double_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.wake_word_threshold  = self.get_parameter('wake_word_threshold').get_parameter_value().double_value
        self.duration             = self.get_parameter('duration').get_parameter_value().double_value
        self.confirm_timeout      = self.get_parameter('confirm_timeout').get_parameter_value().double_value
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
        """Matches main.py exactly — same feature extraction as training."""
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

    def _listen_and_classify(self, stream, timeout_s: float = None):
        """
        Mirrors the approach from the working main.py:
          - Poll chunks until voice is detected (RMS > volume_threshold)
          - Keep the trigger chunk — don't discard it — so the first
            phoneme of the word is captured (this was the original bug)
          - Record a full 1.5 s window from that point
          - Run inference and return (idx, confidence, label)

        Returns None if timeout_s is set and no voice is heard in time.
        """
        n_window = int(SAMPLE_RATE / CHUNK * self.duration)  # ~23 chunks for 1.5 s
        n_timeout = int(SAMPLE_RATE / CHUNK * timeout_s) if timeout_s is not None else None
        count = 0

        while self._running:
            chunk = stream.read(CHUNK, exception_on_overflow=False)

            if self._get_rms(chunk) > self.volume_threshold:
                # ✅ Include the trigger chunk in the recording window
                #    (main.py does exactly this with `frames = [data]`)
                frames = [chunk] + [
                    stream.read(CHUNK, exception_on_overflow=False)
                    for _ in range(n_window)
                ]
                raw = b''.join(frames)
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                idx, conf, label, _ = self._run_inference(audio)
                return idx, conf, label

            if n_timeout is not None:
                count += 1
                if count >= n_timeout:
                    return None  # timed out

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
                # STATE 1 — SLEEPING: wait for wake word "rafeeq"
                # Uses _listen_and_classify (same as main.py) so the trigger
                # chunk is included and the full word is captured.
                # ════════════════════════════════════════════════════════════
                self.get_logger().info('Sleeping — say "rafeeq" to activate.')

                while self._running:
                    result = self._listen_and_classify(stream, timeout_s=None)
                    if result is None:
                        break  # node is shutting down

                    idx, conf, label = result

                    if label == 'rafeeq' and conf >= self.wake_word_threshold:
                        self.get_logger().info(
                            f'Wake word "rafeeq" confirmed ({conf*100:.1f}%) — activated!'
                        )
                        break
                    else:
                        self.get_logger().debug(
                            f'Ignored: "{label}" ({conf*100:.1f}%)'
                        )

                if not self._running:
                    break

                # ════════════════════════════════════════════════════════════
                # STATE 2 — LISTEN FOR COMMAND
                # Same approach: wait for voice → record 1.5 s → classify.
                # ════════════════════════════════════════════════════════════
                self.get_logger().info('Activated! Speak your command now...')

                result = self._listen_and_classify(stream, timeout_s=5.0)

                if result is None:
                    self.get_logger().warn('No command heard — going back to sleep.')
                    continue

                idx, confidence, command = result
                self.get_logger().info(f'Heard "{command}" ({confidence*100:.1f}%)')

                if command in ('stop', 'sleep'):
                    self.get_logger().info(f'"{command}" — going back to sleep.')
                    continue

                if confidence < self.confidence_threshold:
                    self.get_logger().info(
                        f'Low confidence ({confidence*100:.1f}%) — going back to sleep.'
                    )
                    continue

                # ════════════════════════════════════════════════════════════
                # STATE 3 — CONFIRM
                # Same approach again: wait for voice → record 1.5 s → verify.
                # ════════════════════════════════════════════════════════════
                self.get_logger().warn(
                    f'Detected "{command}" ({confidence*100:.1f}%). '
                    f'Say it again to confirm ({self.confirm_timeout:.0f} s)...'
                )

                result = self._listen_and_classify(stream, timeout_s=self.confirm_timeout)

                if result is None:
                    self.get_logger().warn(
                        f'No confirmation — "{command}" discarded, going back to sleep.'
                    )
                    continue

                c_idx, c_conf, c_command = result

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
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 