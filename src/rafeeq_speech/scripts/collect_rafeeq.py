#!/usr/bin/env python3
"""
Record audio samples of the wake word "rafeeq" for retraining.
Usage:
    python3 collect_rafeeq.py --out ~/rafeeq_samples --count 80

Press ENTER to record each sample, Ctrl+C to stop early.
"""

import argparse
import os
import time
import numpy as np
import pyaudio
import wave

SAMPLE_RATE = 16000
CHUNK       = 1024
DURATION    = 1.5   # seconds per sample


def record_one(p, out_path: str):
    stream = p.open(format=pyaudio.paInt16, channels=1,
                    rate=SAMPLE_RATE, input=True,
                    frames_per_buffer=CHUNK)
    n_chunks = int(SAMPLE_RATE / CHUNK * DURATION)
    print(f'  Recording {DURATION}s ...', end=' ', flush=True)
    frames = [stream.read(CHUNK, exception_on_overflow=False)
              for _ in range(n_chunks)]
    stream.stop_stream()
    stream.close()
    print('done')

    with wave.open(out_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',   default=os.path.expanduser('~/rafeeq_samples'),
                    help='Output directory')
    ap.add_argument('--count', type=int, default=80,
                    help='Number of samples to collect')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    existing = len([f for f in os.listdir(args.out) if f.endswith('.wav')])
    print(f'Saving to: {args.out}  (already have {existing} samples)')
    print(f'Will record {args.count} more samples.')
    print('Say "rafeeq" clearly each time. Press ENTER to record, Ctrl+C to quit.\n')

    p = pyaudio.PyAudio()
    try:
        for i in range(existing, existing + args.count):
            input(f'[{i+1}/{existing+args.count}] Press ENTER then say "rafeeq" ...')
            path = os.path.join(args.out, f'rafeeq_{i:04d}.wav')
            record_one(p, path)
            print(f'  Saved: {path}')
    except KeyboardInterrupt:
        print('\nStopped early.')
    finally:
        p.terminate()

    total = len([f for f in os.listdir(args.out) if f.endswith('.wav')])
    print(f'\nTotal samples collected: {total}')


if __name__ == '__main__':
    main()
