from pathlib import Path
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from termcolor import colored

"""
Using the IRIS FDSN seisemolgy standard to download a manageable amount of 
seismic waveform data. These would be low frequency acoustic signatures and
will be the samples of the seismic class

I think this is acceptable since I am doing a more overarching and vague
classification of ocean acoustics to possible be extended later into more
spefic subclassifications (e.g. bioliglcal -> whales, dolphins, sea lions) given
that in the future a more robust source of oceanographic acoustics samples is found

Note this data is not actual audio, however it should work just fine for classification
with consistent representation as a time series

obspy is the client by which the seismic data is retrieved
"""

OUT_DIR = Path("data/raw/seismic")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "seismic_30min_IU_ANMO_BHZ.mseed"

def main():
    client = Client("IRIS")

    # Station: IU.ANMO (Albuquerque, New Mexico), should be a reliable station
    network = "IU"
    station = "ANMO"
    location = "00"
    channel = "BHZ"  # broadband, vertical component

    # Using fixed time window (30 minutes) so deterministic for reproducibility
    start = UTCDateTime("2011-03-11T05:46:00")  # time of the Tohoku quake (UTC)
    end = start + 30 * 60

    print(colored("Requesting waveform from IRIS FDSN...", "light_magenta"))
    print(f"  {network}.{station}.{location}.{channel}")
    print(f"  Start: {start}  End: {end}")

    st = client.get_waveforms(
        network=network,
        station=station,
        location=location,
        channel=channel,
        starttime=start,
        endtime=end,
        attach_response=False,
    )

    # Interpolated merge to address NAs
    st.merge(fill_value="interpolate")

    # Inspect retrieved data
    tr = st[0]
    duration = tr.stats.endtime - tr.stats.starttime
    print("\nDownloaded trace:")
    print("  id:", tr.id)
    print("  start:", tr.stats.starttime)
    print("  end:", tr.stats.endtime)
    print("  duration (s):", float(duration))
    print("  sampling_rate (Hz):", tr.stats.sampling_rate)
    print("  npts:", tr.stats.npts)

    # Save
    st.write(str(OUT_FILE), format="MSEED")
    print("\nSaved to:", OUT_FILE.resolve())

if __name__ == "__main__":
    main()
