import soundfile as sf
from pathlib import Path

"""
To accomodate for class imbalance for ambient hydrophone audio class, I have downloaded
some hydrophone recording to derive ambient samples to balance classes at least a little
bit better to assist in ML learning

Although it must be noted for ocean acoustics, a class imbalance is common thinking about
the proportions of acoustic signals relevant to each of the classes, regardless
it can be worked with with cross entropy weighted loss

Although the nature of the ambience as different conditions (hydrophones are in different
geographic locations at different depths and such) they still convey the underlying
aquatic noises of ambient recordings (that being recordings with no other phenomenon such
as biological, seismic, and so on)
"""


# AUDIO_PATH = Path("data/raw/ambient_long/fieldrecording_hydrophone_01.mp3")
AUDIO_PATH = Path("data/raw/ambient_long/fieldrecording_hydrophone_02.mp3")

def main():
    with sf.SoundFile(AUDIO_PATH) as f:
        samplerate = f.samplerate
        channels = f.channels
        frames = len(f)

    duration_seconds = frames / samplerate
    duration_minutes = duration_seconds / 60

    print("Audio inspection:")
    print(f"  Path: {AUDIO_PATH}")
    print(f"  Sample rate: {samplerate} Hz")
    print(f"  Channels: {channels}")
    print(f"  Total frames: {frames}")
    print(f"  Duration: {duration_seconds:.1f} seconds ({duration_minutes:.1f} minutes)")

    clips_5s = int(duration_seconds // 5)
    print(f"  Possible 5-second clips: {clips_5s}")

if __name__ == "__main__":
    main()
