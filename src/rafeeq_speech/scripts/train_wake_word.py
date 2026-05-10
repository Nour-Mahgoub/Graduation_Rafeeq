#!/usr/bin/env python3
"""
Train a small binary TFLite wake-word model: rafeeq vs. background/other.

Usage:
    python3 train_wake_word.py \
        --rafeeq ~/rafeeq_samples \
        --out    src/rafeeq_speech/models/wake_word.tflite

The script generates background samples automatically from mic silence.
"""

import argparse
import os
import glob
import time
import numpy as np
import librosa
import pyaudio
import wave
import tensorflow as tf
from sklearn.model_selection import train_test_split

SAMPLE_RATE  = 16000
DURATION     = 1.5
N_MFCC       = 13
N_FFT        = 2048
HOP_LENGTH   = 512
SAMPLES      = int(SAMPLE_RATE * DURATION)


def extract_features(audio: np.ndarray) -> np.ndarray:
    y, _ = librosa.effects.trim(audio, top_db=20)
    if len(y) > SAMPLES:
        y = y[:SAMPLES]
    else:
        y = np.pad(y, (0, SAMPLES - len(y)), 'constant')
    mfccs = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE,
                                  n_mfcc=N_MFCC,
                                  n_fft=N_FFT,
                                  hop_length=HOP_LENGTH)
    return mfccs.astype(np.float32)


def load_wav(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
    return audio.astype(np.float32)


def record_background(n_samples: int, out_dir: str):
    print(f'Recording {n_samples} background silence samples ...')
    print('Please stay quiet (background noise only).')
    os.makedirs(out_dir, exist_ok=True)
    p = pyaudio.PyAudio()
    n_chunks = int(SAMPLE_RATE / CHUNK * DURATION)
    CHUNK = 1024

    stream = p.open(format=pyaudio.paInt16, channels=1,
                    rate=SAMPLE_RATE, input=True,
                    frames_per_buffer=CHUNK)
    for i in range(n_samples):
        frames = [stream.read(CHUNK, exception_on_overflow=False)
                  for _ in range(n_chunks)]
        path = os.path.join(out_dir, f'bg_{i:04d}.wav')
        with wave.open(path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames))
        print(f'  [{i+1}/{n_samples}]', end='\r')
    stream.stop_stream(); stream.close(); p.terminate()
    print(f'\nBackground samples saved to {out_dir}')


def build_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, out)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rafeeq', required=True, help='Dir with rafeeq .wav files')
    ap.add_argument('--bg',     default=None,  help='Dir with background .wav files (auto-recorded if missing)')
    ap.add_argument('--out',    required=True, help='Output .tflite path')
    ap.add_argument('--epochs', type=int, default=40)
    args = ap.parse_args()

    # ── Load rafeeq samples ───────────────────────────────────────────────────
    rafeeq_files = glob.glob(os.path.join(args.rafeeq, '*.wav'))
    if not rafeeq_files:
        raise SystemExit(f'No .wav files in {args.rafeeq}. Run collect_rafeeq.py first.')
    print(f'Rafeeq samples: {len(rafeeq_files)}')

    # ── Background samples ────────────────────────────────────────────────────
    bg_dir = args.bg or os.path.join(args.rafeeq, '../background_samples')
    bg_dir = os.path.abspath(bg_dir)
    bg_files = glob.glob(os.path.join(bg_dir, '*.wav'))
    if not bg_files:
        record_background(len(rafeeq_files), bg_dir)
        bg_files = glob.glob(os.path.join(bg_dir, '*.wav'))

    print(f'Background samples: {len(bg_files)}')

    # ── Extract features ──────────────────────────────────────────────────────
    X, y = [], []
    for f in rafeeq_files:
        X.append(extract_features(load_wav(f)))
        y.append(1)
    for f in bg_files:
        X.append(extract_features(load_wav(f)))
        y.append(0)

    X = np.array(X)[..., np.newaxis]   # (N, n_mfcc, frames, 1)
    y = np.array(y, dtype=np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'Train: {len(X_train)}  Val: {len(X_val)}')

    # ── Train ─────────────────────────────────────────────────────────────────
    model = build_model(X_train.shape[1:])
    model.summary()
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=args.epochs,
              batch_size=16)

    # ── Convert to TFLite ─────────────────────────────────────────────────────
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'wb') as f:
        f.write(tflite_model)
    print(f'\nSaved wake word model → {args.out}')


if __name__ == '__main__':
    main()
