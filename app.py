import streamlit as st
import mne
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="NeuroClean Pro | Auto-EEG Preprocessor",
    page_icon="🧠",
    layout="wide"
)

# --- BACKEND FUNCTIONS ---

def load_pretrained_model():
    """
    Replace with actual loading of your pretrained model if available
    """
    # from tensorflow.keras.models import load_model
    # return load_model("pretrained_eegnet.h5")
    return None  # placeholder if no model available

def process_eeg(file_path, l_freq, h_freq, notch_freq, use_model=False):
    """
    Full pipeline: Load -> Filter -> ICA -> (optional model) -> Clean
    """
    # 1. Load file
    try:
        if file_path.endswith('.edf'):
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        elif file_path.endswith('.fif'):
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
        elif file_path.endswith('.set'):
            raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        else:
            raw = mne.io.read_raw(file_path, preload=True, verbose=False)
    except Exception as e:
        return None, f"Error reading file: {str(e)}"

    # Standard montage
    try:
        raw.set_montage('standard_1020', on_missing='ignore')
    except:
        pass

    # 2. Filtering
    raw_filtered = raw.copy()
    raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    raw_filtered.notch_filter(freqs=notch_freq, verbose=False)

    # 3. ICA - only EEG channels
    eeg_picks = mne.pick_types(raw_filtered.info, eeg=True, meg=False, stim=False, eog=False)
    n_channels = len(eeg_picks)
    n_components = min(15, n_channels)  # ICA components <= # of EEG channels
    ica = ICA(n_components=n_components, max_iter='auto', random_state=97, method='picard')
    ica.fit(raw_filtered, picks=eeg_picks, verbose=False)

    # 4. Artifact removal (eye, heart, muscle) using heuristic
    exclude_idx = []  # If no model, leave empty
    if not use_model:
        # Simple automatic exclusion: remove components with high variance (naive approach)
        # For a real AI, replace with ICLabel or pretrained model
        var = np.var(ica.get_sources(raw_filtered).get_data(), axis=1)
        exclude_idx = [i for i, v in enumerate(var) if v > np.percentile(var, 90)]
    
    ica.exclude = exclude_idx

    # Apply cleaning
    raw_cleaned = raw_filtered.copy()
    ica.apply(raw_cleaned, verbose=False)

    return {
        "original": raw,
        "cleaned": raw_cleaned,
        "ica": ica,
        "exclude": exclude_idx
    }, None

def plot_psd_comparison(raw_orig, raw_clean):
    """Frequency domain comparison"""
    fig, ax = plt.subplots(figsize=(10, 4))
    psd_orig = raw_orig.compute_psd(fmax=60)
    psd_clean = raw_clean.compute_psd(fmax=60)
    psd_orig.plot(axes=ax, color='red', alpha=0.5, show=False, average=True, spatial_colors=False)
    psd_clean.plot(axes=ax, color='green', alpha=0.8, show=False, average=True, spatial_colors=False)
    ax.set_title("Frequency Spectrum (PSD)")
    ax.legend(["Original", "Cleaned"])
    plt.tight_layout()
    return fig

def plot_time_series(raw_orig, raw_clean, channel, start_time, duration=4.0):
    """Time domain comparison"""
    fig, ax = plt.subplots(figsize=(12, 4))
    data_orig, times = raw_orig.get_data(picks=channel, tmin=start_time, tmax=start_time+duration, return_times=True)
    data_clean, _ = raw_clean.get_data(picks=channel, tmin=start_time, tmax=start_time+duration, return_times=True)
    scale = 1e6
    ax.plot(times, data_orig[0]*scale, color='red', alpha=0.4, label='Original')
    ax.plot(times, data_clean[0]*scale, color='green', alpha=0.9, linewidth=1.5, label='Cleaned')
    ax.set_title(f"Channel: {channel} | Time: {start_time}-{start_time+duration}s")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

# --- FRONTEND UI ---
st.title("🧠 NeuroClean Pro")
st.markdown("Automated EEG Artifact Rejection with **Before/After Visualization**")

st.sidebar.header("Settings")
low_cut = st.sidebar.number_input("Low Cut (Hz)", value=1.0)
high_cut = st.sidebar.number_input("High Cut (Hz)", value=40.0)
notch = st.sidebar.selectbox("Notch Filter (Hz)", [50, 60], index=0)
use_model = st.sidebar.checkbox("Use Pretrained Model (if available)", value=False)

uploaded_file = st.file_uploader("Upload EEG (.edf, .fif, .set)", type=['edf','fif','set'])

if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False

if uploaded_file:
    if not st.session_state.data_processed:
        if st.button("Run Auto-Preprocessing", type="primary"):
            with st.spinner("Processing EEG..."):
                suffix = f".{uploaded_file.name.split('.')[-1]}"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                results, error = process_eeg(tmp_path, low_cut, high_cut, notch, use_model)
                os.remove(tmp_path)

                if error:
                    st.error(error)
                else:
                    st.session_state.results = results
                    st.session_state.data_processed = True
                    st.rerun()

    if st.session_state.data_processed:
        results = st.session_state.results
        st.divider()
        st.success("Preprocessing Complete!")

        # Time Domain
        st.subheader("1. Time Domain Inspection")
        all_chans = results['cleaned'].ch_names
        default_idx = all_chans.index('Fp1') if 'Fp1' in all_chans else 0
        selected_chan = st.selectbox("Select Channel", all_chans, index=default_idx)
        max_time = int(results['cleaned'].times[-1])
        start_time = st.slider("Time Window (seconds)", 0, max_time, 0)
        fig_time = plot_time_series(results['original'], results['cleaned'], selected_chan, start_time)
        st.pyplot(fig_time)

        # Frequency Domain
        st.subheader("2. Frequency Domain Inspection (PSD)")
        fig_psd = plot_psd_comparison(results['original'], results['cleaned'])
        st.pyplot(fig_psd)

        # Details & Download
        st.subheader("3. Details & Download")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Components Removed: {len(results['exclude'])}")
            st.write(f"Indices: {results['exclude']}")
        with col2:
            clean_path = os.path.join(tempfile.gettempdir(), "clean_eeg.fif")
            results['cleaned'].save(clean_path, overwrite=True, verbose=False)
            with open(clean_path, "rb") as f:
                st.download_button("Download Clean EEG (.FIF)", f, "clean_eeg.fif")

        if st.button("Reset / Upload New File"):
            st.session_state.data_processed = False
            st.rerun()
