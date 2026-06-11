#!/usr/bin/env python3

import os
import time
import subprocess
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
        self.declare_parameter('model_path',           'rafeeq_model.tflite')
        self.declare_parameter('labels_path',          'labels.txt')
        self.declare_parameter('records_path',         'command_records')  # dir with .wav files
        self.declare_parameter('volume_threshold',     0.02)   # matches working main.py
        self.declare_parameter('confidence_threshold', 0.70)   # matches working main.py
        self.declare_parameter('wake_word_threshold',  0.50)   # min confidence for "rafeeq"
        self.declare_parameter('duration',             1.5)    # MUST stay 1.5 — model trained on this
        self.declare_parameter('confirm_silence',      3.0)    # seconds of silence = user accepts

        model_path  = self.get_parameter('model_path').get_parameter_value().string_value
        labels_path = self.get_parameter('labels_path').get_parameter_value().string_value
        self.records_path         = self.get_parameter('records_path').get_parameter_value().string_value
        self.volume_threshold     = self.get_parameter('volume_threshold').get_parameter_value().double_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.wake_word_threshold  = self.get_parameter('wake_word_threshold').get_parameter_value().double_value
        self.duration             = self.get_parameter('duration').get_parameter_value().double_value
        self.confirm_silence      = self.get_parameter('confirm_silence').get_parameter_value().double_value
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
        self._thread  = threading.Thread(target=self._audio_loop, daemon=True)
        self._thread.start()
        self.get_logger().info('Rafeeq speech node ready — listening for commands.')

    # ── Audio helpers ─────────────────────────────────────────────────────────

    def _get_rms(self, block: bytes) -> float:
        audio = np.frombuffer(block, dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(audio ** 2)))

    @staticmethod
    def _pre_emphasis(y: np.ndarray, coeff: float = 0.97) -> np.ndarray:
        """
        First-order pre-emphasis FIR filter: H(z) = 1 - 0.97·z⁻¹
        (Darabkh et al. 2013, §II-B, Eq. 1)
        Boosts high-frequency consonants and flattens the spectral slope.
        MUST be applied before MFCC extraction at inference to match training.
        """
        return np.append(y[0], y[1:] - coeff * y[:-1]).astype(np.float32)

    def _extract_features(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Extract 3-channel feature map to match the retrained model.

        Pipeline (must be identical to model_generation_paper.ipynb):
          1. Trim silence
          2. Pad / truncate to 1.5 s
          3. Pre-emphasis (1 - 0.97·z⁻¹)
          4. MFCC (13 coefficients, ~47 frames)
          5. Δ  — 1st-order temporal derivative  (phoneme transitions)
          6. ΔΔ — 2nd-order temporal derivative  (onset / offset dynamics)
          7. Stack as 3 channels → shape (13, T, 3)
          8. Add batch dim        → shape (1,  13, T, 3)
        """
        y, _ = librosa.effects.trim(audio_array, top_db=20)
        if len(y) > self.samples_per_track:
            y = y[:self.samples_per_track]
        else:
            y = np.pad(y, (0, self.samples_per_track - len(y)), 'constant')

        # Pre-emphasis (paper approach)
        y = self._pre_emphasis(y, coeff=0.97)

        mfccs  = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=13,
                                       n_fft=2048, hop_length=512)   # (13, T)
        delta  = librosa.feature.delta(mfccs, order=1)               # (13, T)
        delta2 = librosa.feature.delta(mfccs, order=2)               # (13, T)

        # Stack: (13, T, 3) — channels: [MFCC, Δ, ΔΔ]
        features = np.stack([mfccs, delta, delta2], axis=-1)
        return np.float32(features[np.newaxis, ...])

    def _run_inference(self, audio_array: np.ndarray):
        features = self._extract_features(audio_array)
        self.interpreter.set_tensor(self.input_details[0]['index'], features)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        idx = int(np.argmax(output))
        return idx, float(output[idx]), self.labels[idx]

    def _record_until_silence(self, stream, trigger_chunk: bytes) -> np.ndarray:
        """
        Record from the trigger chunk until end-of-speech, then return audio.

        Rules:
          - Always record at least MIN_CHUNKS (0.5 s) before checking silence.
            This prevents brief noise spikes from producing a tiny recording.
          - After the minimum, stop when TRAILING_SILENT_CHUNKS (5 chunks,
            ~0.32 s) of consecutive silence is detected → speech ended.
          - Hard cap at 1.5 s (n_max) regardless.

        _extract_features pads to samples_per_track so stopping early is safe.
        """
        n_max                = int(SAMPLE_RATE / CHUNK * self.duration)  # 23 chunks = 1.5 s
        MIN_CHUNKS           = int(SAMPLE_RATE / CHUNK * 0.5)           # 7 chunks  = 0.5 s min
        TRAILING_SILENT_CHUNKS = 5                                       # ~0.32 s of silence
        frames  = [trigger_chunk]
        silent  = 0

        for i in range(n_max - 1):
            c = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(c)

            # Only start checking for end-of-speech after the minimum duration
            if i >= MIN_CHUNKS:
                if self._get_rms(c) < self.volume_threshold:
                    silent += 1
                    if silent >= TRAILING_SILENT_CHUNKS:
                        break       # speech finished
                else:
                    silent = 0      # still speaking

        raw = b''.join(frames)
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def _listen_and_classify(self, stream, timeout_s: float = None):
        """
        Poll until voice spike, then record until end-of-speech (not fixed 1.5 s).
        Returns (idx, confidence, label) or None on timeout / shutdown.
        """
        n_timeout = int(SAMPLE_RATE / CHUNK * timeout_s) if timeout_s is not None else None
        count = 0

        while self._running:
            chunk = stream.read(CHUNK, exception_on_overflow=False)

            if self._get_rms(chunk) > self.volume_threshold:
                audio = self._record_until_silence(stream, chunk)
                idx, conf, label = self._run_inference(audio)
                return idx, conf, label

            if n_timeout is not None:
                count += 1
                if count >= n_timeout:
                    return None  # timed out

        return None  # node shutting down

    def _wait_for_confirmation(self, stream, duration_s: float) -> bool:
        """
        Listen for duration_s seconds after the robot plays back the command.

        Cancel rule — ONLY "stop":
          - Noise / low confidence / any other command  → ignored, keep waiting
          - "stop" detected with confidence >= threshold → cancel

        Returns True  → window completed without "stop" → publish.
        Returns False → "stop" detected                 → cancel, back to sleep.

        ── Why wall-clock time instead of chunk counting ────────────────────
        The old implementation counted chunks to track elapsed time. But
        _record_until_silence() consumes up to 1.5 s of audio per call, and
        if background noise triggers it twice, the 3-second budget burns up
        instantly — making confirmation completely non-deterministic.
        time.monotonic() is always exactly duration_s of real time regardless
        of how many recordings happen inside the window.
        """
        deadline = time.monotonic() + duration_s

        while self._running and time.monotonic() < deadline:
            chunk = stream.read(CHUNK, exception_on_overflow=False)

            if self._get_rms(chunk) > self.volume_threshold:
                # Voice detected — record until end-of-speech, then check for "stop"
                audio          = self._record_until_silence(stream, chunk)
                _, conf, label = self._run_inference(audio)

                if label == 'stop' and conf >= self.confidence_threshold:
                    self.get_logger().warn(
                        f'"stop" detected ({conf*100:.1f}%) — command cancelled.'
                    )
                    return False  # explicit cancel

                # Anything else (noise, echo, other command) → ignored
                self.get_logger().debug(
                    f'Ignored during confirmation: "{label}" ({conf*100:.1f}%)'
                )

        return True  # window expired without "stop" → publish

    def _play_wav(self, command: str):
        """
        Play the pre-recorded WAV file using aplay (blocking).
        Using subprocess + aplay instead of PyAudio avoids conflicts with
        the open input stream (PyAudio p.terminate() can destabilise ALSA).
        """
        wav_path = os.path.join(self.records_path, f'{command}.wav')
        if not os.path.isfile(wav_path):
            self.get_logger().warn(f'No WAV file found for "{command}" at {wav_path}')
            return
        try:
            result = subprocess.run(
                ['aplay', '-q', wav_path],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                self.get_logger().warn(
                    f'aplay failed for "{command}": {result.stderr.strip()}'
                )
        except FileNotFoundError:
            self.get_logger().error(
                'aplay not found — install it with: sudo apt install alsa-utils'
            )
        except Exception as e:
            self.get_logger().warn(f'Could not play WAV for "{command}": {e}')

    def _flush_mic(self, stream, duration_s: float = 0.5):
        """
        Discard buffered microphone audio accumulated while the WAV was playing.
        Without this, _wait_for_confirmation reads the echoed WAV playback
        instead of fresh audio and may falsely cancel the command.
        """
        n_chunks = int(SAMPLE_RATE / CHUNK * duration_s)
        for _ in range(n_chunks):
            stream.read(CHUNK, exception_on_overflow=False)

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
                # STATE 1 — SLEEPING
                # Wait indefinitely for either:
                #   • "rafeeq"  → wake up and listen for a command
                #   • "stop"    → publish immediately (emergency stop, no
                #                 wake word required), stay sleeping
                # ════════════════════════════════════════════════════════════
                self.get_logger().info('Sleeping — say "rafeeq" to activate (or "stop" to stop).')

                while self._running:
                    result = self._listen_and_classify(stream, timeout_s=None)
                    if result is None:
                        break  # shutting down

                    idx, conf, label = result

                    # Emergency stop — publish without needing wake word
                    if label == 'stop' and conf >= self.confidence_threshold:
                        self.get_logger().info(
                            f'"stop" detected ({conf*100:.1f}%) — publishing immediately (from sleep).'
                        )
                        msg      = String()
                        msg.data = 'stop'
                        self.nav_pub.publish(msg)
                        # Stay in sleep state — don't break, keep listening

                    elif label == 'rafeeq' and conf >= self.wake_word_threshold:
                        self.get_logger().info(
                            f'Wake word "rafeeq" confirmed ({conf*100:.1f}%) — activated!'
                        )
                        break

                    else:
                        self.get_logger().debug(f'Ignored: "{label}" ({conf*100:.1f}%)')

                if not self._running:
                    break

                # Flush residual audio from wake word before listening for command
                self._flush_mic(stream, duration_s=0.3)

                # ════════════════════════════════════════════════════════════
                # STATE 2 — DETECT COMMAND
                # Listen for up to 5 s for a navigation command.
                # ════════════════════════════════════════════════════════════
                self.get_logger().info('Activated! Speak your command...')

                result = self._listen_and_classify(stream, timeout_s=5.0)

                if result is None:
                    self.get_logger().warn('No command heard — going back to sleep.')
                    continue

                idx, confidence, command = result
                self.get_logger().info(f'Heard "{command}" ({confidence*100:.1f}%)')

                # Low confidence → discard
                if confidence < self.confidence_threshold:
                    self.get_logger().info(
                        f'Low confidence ({confidence*100:.1f}%) — going back to sleep.'
                    )
                    continue

                # "sleep" is a mode command — go back to sleep, nothing published
                if command == 'sleep':
                    self.get_logger().info('"sleep" — going back to sleep.')
                    continue

                # "stop" publishes IMMEDIATELY — no playback, no confirmation window
                if command == 'stop':
                    self.get_logger().info(
                        f'"stop" detected ({confidence*100:.1f}%) — publishing immediately.'
                    )
                    msg      = String()
                    msg.data = 'stop'
                    self.nav_pub.publish(msg)
                    continue  # back to sleep

                # ════════════════════════════════════════════════════════════
                # STATE 3 — PLAY BACK + CONFIRM
                # Robot speaks the command using the pre-recorded WAV.
                # After playback, a confirmation window opens:
                #   - Silence for confirm_silence seconds → publish ✅
                #   - "stop" detected (>= threshold)     → cancel  ❌
                # ════════════════════════════════════════════════════════════
                self.get_logger().info(
                    f'Playing back "{command}" — stay silent to confirm, '
                    f'say "stop" to cancel.'
                )
                self._play_wav(command)
                self._flush_mic(stream, duration_s=0.5)  # discard echo buffered during playback

                accepted = self._wait_for_confirmation(stream, duration_s=self.confirm_silence)

                if accepted:
                    self.get_logger().info(
                        f'CONFIRMED: "{command}" — publishing to /navigation_goal'
                    )
                    msg      = String()
                    msg.data = command
                    self.nav_pub.publish(msg)
                else:
                    self.get_logger().warn(
                        f'Cancelled — "{command}" discarded, going back to sleep.'
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
