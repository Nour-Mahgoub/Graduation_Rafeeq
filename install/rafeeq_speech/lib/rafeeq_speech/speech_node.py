#!/usr/bin/env python3

import os
import wave
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
        self.declare_parameter('model_path',          'rafeeq_model.tflite')
        self.declare_parameter('labels_path',         'labels.txt')
        self.declare_parameter('records_path',        'command_records')   # dir with .wav files
        self.declare_parameter('volume_threshold',    0.02)   # matches working main.py
        self.declare_parameter('confidence_threshold', 0.70)  # matches working main.py
        self.declare_parameter('wake_word_threshold', 0.50)   # min confidence for "rafeeq"
        self.declare_parameter('duration',            1.5)    # MUST stay 1.5 — model trained on this
        self.declare_parameter('confirm_silence',     3.0)    # seconds of silence = user accepts

        model_path   = self.get_parameter('model_path').get_parameter_value().string_value
        labels_path  = self.get_parameter('labels_path').get_parameter_value().string_value
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
        self._thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._thread.start()
        self.get_logger().info('Rafeeq speech node ready — listening for commands.')

    # ── Audio helpers ─────────────────────────────────────────────────────────

    def _get_rms(self, block: bytes) -> float:
        audio = np.frombuffer(block, dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(audio ** 2)))

    def _extract_features(self, audio_array: np.ndarray) -> np.ndarray:
        """Identical to working main.py — must match training exactly."""
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
        return idx, float(output[idx]), self.labels[idx]

    def _listen_and_classify(self, stream, timeout_s: float = None):
        """
        Mirrors main.py approach:
          - Poll until voice spike (RMS > volume_threshold)
          - Include trigger chunk so no audio is wasted
          - Record full 1.5 s window from that point
          - Run inference and return (idx, confidence, label)
        Returns None on timeout or shutdown.
        """
        n_window  = int(SAMPLE_RATE / CHUNK * self.duration)
        n_timeout = int(SAMPLE_RATE / CHUNK * timeout_s) if timeout_s is not None else None
        count = 0

        while self._running:
            chunk = stream.read(CHUNK, exception_on_overflow=False)

            if self._get_rms(chunk) > self.volume_threshold:
                # Include the trigger chunk — same as main.py's `frames = [data]`
                frames = [chunk] + [
                    stream.read(CHUNK, exception_on_overflow=False)
                    for _ in range(n_window)
                ]
                raw   = b''.join(frames)
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                idx, conf, label = self._run_inference(audio)
                return idx, conf, label

            if n_timeout is not None:
                count += 1
                if count >= n_timeout:
                    return None  # timed out

        return None  # node shutting down

    def _wait_for_confirmation(self, stream, duration_s: float) -> bool:
        """
        Listen for `duration_s` seconds after the robot plays back the command.

        Cancel rule — ONLY if the model confidently detects "stop":
          - Noise / low confidence / any other command → ignored, keep waiting
          - "stop" detected with confidence >= confidence_threshold → cancel

        This prevents false cancellations from background noise, TV,
        or the speaker echo re-triggering the mic.

        Returns True  → window completed without "stop" → publish.
        Returns False → "stop" detected                → cancel, back to sleep.
        """
        n_window  = int(SAMPLE_RATE / CHUNK * self.duration)
        n_timeout = int(SAMPLE_RATE / CHUNK * duration_s)
        count = 0

        while self._running and count < n_timeout:
            chunk = stream.read(CHUNK, exception_on_overflow=False)
            count += 1

            if self._get_rms(chunk) > self.volume_threshold:
                # Voice detected — run the model to check if it's "stop"
                frames = [chunk] + [
                    stream.read(CHUNK, exception_on_overflow=False)
                    for _ in range(n_window)
                ]
                count += n_window  # account for the chunks we just consumed
                raw   = b''.join(frames)
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                _, conf, label = self._run_inference(audio)

                if label == 'stop' and conf >= self.confidence_threshold:
                    self.get_logger().warn(
                        f'"stop" detected ({conf*100:.1f}%) — command cancelled.'
                    )
                    return False  # explicit cancel

                # Anything else (noise, echo, other command) → ignore
                self.get_logger().debug(
                    f'Ignored during confirmation: "{label}" ({conf*100:.1f}%)'
                )

        return True  # window expired without "stop" → accept

    def _play_wav(self, command: str):
        """
        Play the pre-recorded WAV file for this command (blocking).
        Files live in records_path/<command>.wav
        """
        wav_path = os.path.join(self.records_path, f'{command}.wav')
        if not os.path.isfile(wav_path):
            self.get_logger().warn(f'No WAV file found for "{command}" at {wav_path}')
            return

        try:
            with wave.open(wav_path, 'rb') as wf:
                p = pyaudio.PyAudio()
                out_stream = p.open(
                    format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                )
                data = wf.readframes(CHUNK)
                while data:
                    out_stream.write(data)
                    data = wf.readframes(CHUNK)
                out_stream.stop_stream()
                out_stream.close()
                p.terminate()
        except Exception as e:
            self.get_logger().warn(f'Could not play WAV for "{command}": {e}')

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
                # ════════════════════════════════════════════════════════════
                self.get_logger().info('Sleeping — say "rafeeq" to activate.')

                while self._running:
                    result = self._listen_and_classify(stream, timeout_s=None)
                    if result is None:
                        break

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
                # ════════════════════════════════════════════════════════════
                self.get_logger().info('Activated! Speak your command...')

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
                # STATE 3 — CONFIRM via playback + silence
                #
                # Robot speaks the command aloud using the pre-recorded WAV.
                # If the user stays silent → accepted → publish.
                # If the user says anything → cancelled → back to sleep.
                # ═══════════════════════════════════════════�