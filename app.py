#!/usr/bin/env python3
"""
Streamlit Application for GPU-Accelerated Beacon Detection
Interactive visualization and analysis interface
"""

import streamlit as st
import subprocess
import json
import os
import tempfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import cupy as cp
import cusignal
from scapy.all import PcapReader, IP
import psutil

# Page config
st.set_page_config(
    page_title="GPU Beacon Detection",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling - FIXED COLORS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #262730;
    }
    .info-box b {
        color: #1f77b4;
    }
    .beacon-indicator {
        background-color: #ff6b6b;
        color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    .dsp-explanation {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        color: #262730;
    }
</style>
""", unsafe_allow_html=True)

def get_gpu_stats():
    """Get GPU statistics using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu", 
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            return {
                'memory_used_mb': int(parts[0]),
                'memory_total_mb': int(parts[1]),
                'gpu_util_percent': int(parts[2]),
                'temperature_c': int(parts[3])
            }
    except:
        pass
    return None

def kill_project_processes():
    """Kill all Python processes related to this project."""
    killed_count = 0
    project_scripts = ["gpu_detector.py", "visualize.py", "data_curation.py"]
    current_pid = os.getpid()
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline:
                        # Check if any project script is in the command line
                        for script in project_scripts:
                            if any(script in str(arg) for arg in cmdline):
                                pid = proc.info['pid']
                                # Don't kill current process
                                if pid != current_pid:
                                    try:
                                        proc_obj = psutil.Process(pid)
                                        proc_obj.terminate()
                                        # Wait a bit, then force kill if needed
                                        try:
                                            proc_obj.wait(timeout=2)
                                        except psutil.TimeoutExpired:
                                            proc_obj.kill()
                                        killed_count += 1
                                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception as e:
        st.warning(f"Error killing processes: {e}")
    
    return killed_count

def run_detection(pcap_path: str) -> dict:
    """Run GPU detector on uploaded PCAP."""
    import time
    
    # Check if stop was requested
    if st.session_state.get('stop_requested', False):
        st.session_state['stop_requested'] = False
        return None
    
    # Store process reference for potential termination
    proc = subprocess.Popen(
        ["python", "gpu_detector.py", "--pcap", pcap_path, "--dtype", "float32", "--top-n", "50"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    st.session_state['detection_process'] = proc
    
    # Use a placeholder to show progress
    status_placeholder = st.empty()
    status_placeholder.info("🔄 Running GPU-accelerated detection (optimized: ~2-5 seconds)...")
    
    try:
        # Poll process with timeout, checking for stop request
        start_time = time.time()
        timeout = 300  # 5 minutes
        poll_interval = 1.0  # Check every second
        
        while proc.poll() is None:
            # Check if stop was requested
            if st.session_state.get('stop_requested', False):
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                st.session_state['stop_requested'] = False
                status_placeholder.warning("⚠️ Detection stopped by user")
                if 'detection_process' in st.session_state:
                    del st.session_state['detection_process']
                return None
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                proc.kill()
                status_placeholder.error("❌ Detection timed out after 5 minutes")
                if 'detection_process' in st.session_state:
                    del st.session_state['detection_process']
                return None
            
            # Small sleep to avoid busy-waiting
            time.sleep(poll_interval)
        
        # Process finished, get output
        stdout, stderr = proc.communicate()
        
        if proc.returncode != 0:
            status_placeholder.error(f"❌ Detection failed: {stderr[:500] if stderr else 'Unknown error'}")
            if 'detection_process' in st.session_state:
                del st.session_state['detection_process']
            return None
        
        status_placeholder.success("✅ Detection completed!")
        time.sleep(0.5)  # Brief pause to show success
        status_placeholder.empty()
        
        # Load results
        results_file = "detection_results.json"
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
                if 'detection_process' in st.session_state:
                    del st.session_state['detection_process']
                return results
    except Exception as e:
        if proc.poll() is None:
            proc.kill()
        status_placeholder.error(f"❌ Detection error: {e}")
        if 'detection_process' in st.session_state:
            del st.session_state['detection_process']
        return None
    
    if 'detection_process' in st.session_state:
        del st.session_state['detection_process']
    return None

def create_plots_directly(results: list, pcap_path: str, visualizer):
    """Create plots directly using plotly instead of HTML files."""
    import time
    total_start = time.time()
    plots = {}
    
    # Pre-load PCAP cache once (shared across all plots)
    print("[*] Pre-loading PCAP cache for visualizations...")
    cache_start = time.time()
    visualizer._load_pcap_cache(pcap_path)
    print(f"[*] PCAP cache loaded in {time.time() - cache_start:.2f}s")
    
    # 2D Plots
    try:
        plot_start = time.time()
        top_results = results[:5]
        fig_2d = make_subplots(
            rows=len(top_results),
            cols=2,
            subplot_titles=[
                f"{r['src_ip']} → {r['dst_ip']} (Score: {r['score']:.2f})"
                for r in top_results
                for _ in [0, 1]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        for idx, result in enumerate(top_results, 1):
            time_axis, event_rate = visualizer.create_time_series_from_pcap(
                pcap_path, result['src_ip'], result['dst_ip']
            )
            if time_axis is not None:
                fig_2d.add_trace(
                    go.Scatter(x=time_axis, y=event_rate, mode='lines', name=f"Event Rate {idx}",
                              line=dict(width=1), showlegend=False),
                    row=idx, col=1
                )
                frequencies, magnitude = visualizer.compute_fft_spectrum(event_rate)
                periods = 1.0 / frequencies[1:]
                magnitude_period = magnitude[1:]
                fig_2d.add_trace(
                    go.Scatter(x=periods, y=magnitude_period, mode='lines', name=f"FFT {idx}",
                              line=dict(width=2), showlegend=False),
                    row=idx, col=2
                )
        
        for i in range(1, len(top_results) + 1):
            fig_2d.update_xaxes(title_text="Time (s)", row=i, col=1)
            fig_2d.update_yaxes(title_text="Event Rate", row=i, col=1)
            fig_2d.update_xaxes(title_text="Period (s)", row=i, col=2)
            fig_2d.update_yaxes(title_text="Magnitude", row=i, col=2)
        
        fig_2d.update_layout(height=300 * len(top_results), showlegend=False,
                            title="Beacon Detection: Event Rate and FFT Spectrum")
        plots['2d'] = fig_2d
        print(f"[*] 2D plots created in {time.time() - plot_start:.2f}s")
    except Exception as e:
        st.warning(f"Could not create 2D plots: {e}")
    
    # PSD Comparison
    try:
        plot_start = time.time()
        fig_psd = go.Figure()
        cache = visualizer._load_pcap_cache(pcap_path)
        beacon_pairs = {(r['src_ip'], r['dst_ip']) for r in results[:3]}
        
        # Find benign pair
        benign_timestamps = None
        for pair_key, timestamps in cache.items():
            if pair_key not in beacon_pairs and len(timestamps) > 100:
                benign_timestamps = timestamps[:1000]
                break
        
        if benign_timestamps is not None:
            min_time = benign_timestamps.min()
            max_time = benign_timestamps.max()
            num_bins = int((max_time - min_time) / 1.0) + 1
            bin_indices = ((benign_timestamps - min_time) / 1.0).astype(int)
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)
            benign_event_rate = np.bincount(bin_indices, minlength=num_bins).astype(float)
            periods_benign, psd_benign = visualizer.compute_psd(benign_event_rate)
            fig_psd.add_trace(go.Scatter(
                x=periods_benign, y=psd_benign, mode='lines', name='Benign Background',
                line=dict(color='gray', width=1, dash='dash'), opacity=0.5
            ))
        
        colors = ['red', 'orange', 'yellow']
        for idx, result in enumerate(results[:3]):
            time_axis, event_rate = visualizer.create_time_series_from_pcap(
                pcap_path, result['src_ip'], result['dst_ip']
            )
            if time_axis is not None:
                periods, psd = visualizer.compute_psd(event_rate)
                fig_psd.add_trace(go.Scatter(
                    x=periods, y=psd, mode='lines',
                    name=f"Beacon #{idx+1}: {result['src_ip']} → {result['dst_ip']}<br>Score: {result['score']:.2f}",
                    line=dict(color=colors[idx % len(colors)], width=2)
                ))
                fig_psd.add_vline(x=result['period'], line=dict(color=colors[idx % len(colors)], width=1, dash='dot'),
                                 annotation_text=f"{result['period']:.1f}s")
        
        fig_psd.update_layout(
            title="Power Spectral Density Comparison: Beacons vs. Benign Traffic",
            xaxis_title="Period (seconds)", yaxis_title="Power Spectral Density (dB)",
            xaxis_type="log", height=600, hovermode='x unified'
        )
        plots['psd'] = fig_psd
        print(f"[*] PSD plot created in {time.time() - plot_start:.2f}s")
    except Exception as e:
        st.warning(f"Could not create PSD plot: {e}")
    
    # 3D Spectrogram (skip if too slow - make it optional)
    try:
        plot_start = time.time()
        if results:
            src_ip = results[0]['src_ip']
            dst_ip = results[0]['dst_ip']
            time_axis, event_rate = visualizer.create_time_series_from_pcap(pcap_path, src_ip, dst_ip)
            if time_axis is not None:
                # Optimize 3D spectrogram - use larger step to reduce computation
                window_size = 200  # Larger windows = fewer windows to process
                overlap = 100
                step = window_size - overlap
                time_windows = []
                frequency_axis = None
                magnitude_list = []
                
                # Limit number of windows for performance
                max_windows = 200
                window_count = 0
                
                for i in range(0, len(event_rate) - window_size, step):
                    if window_count >= max_windows:
                        break
                    
                    window_data = event_rate[i:i + window_size]
                    window_time = time_axis[i + window_size // 2]
                    windowed = np.hanning(len(window_data))
                    windowed_series = window_data * windowed
                    detrended_gpu = cusignal.detrend(cp.asarray(windowed_series, dtype=cp.float32), type='linear')
                    fft_result_gpu = cp.fft.rfft(detrended_gpu)
                    fft_magnitude = cp.abs(fft_result_gpu).get()
                    
                    if frequency_axis is None:
                        frequencies = cp.fft.rfftfreq(len(window_data), 1.0).get()
                        frequency_axis = 1.0 / frequencies[1:]
                        expected_size = len(frequency_axis)
                    
                    fft_magnitude_no_dc = fft_magnitude[1:]
                    if len(fft_magnitude_no_dc) == expected_size:
                        time_windows.append(window_time)
                        magnitude_list.append(fft_magnitude_no_dc)
                        window_count += 1
                
                time_windows = np.array(time_windows)
                magnitude_matrix = np.array(magnitude_list)
                magnitude_matrix_log = np.log10(magnitude_matrix + 1e-10)
                
                fig_3d = go.Figure(data=[go.Surface(
                    x=time_windows, y=frequency_axis, z=magnitude_matrix_log.T,
                    colorscale='Viridis', showscale=True, colorbar=dict(title="Log10(Magnitude)")
                )])
                fig_3d.update_layout(
                    title=f"3D Spectrogram (Log-Scale): {src_ip} → {dst_ip}",
                    scene=dict(xaxis_title="Time (s)", yaxis_title="Period (s)", zaxis_title="Log10(Magnitude)"),
                    width=1200, height=800
                )
                plots['3d'] = fig_3d
                print(f"[*] 3D spectrogram created in {time.time() - plot_start:.2f}s")
    except Exception as e:
        st.warning(f"Could not create 3D spectrogram: {e}")
    
    total_time = time.time() - total_start
    print(f"[✓] All visualizations created in {total_time:.2f}s")
    return plots

def create_beacon_summary_table(results: list):
    """Create a summary table of detected beacons."""
    if not results:
        return None
    
    df_data = []
    for idx, r in enumerate(results, 1):
        df_data.append({
            "Rank": idx,
            "Score": f"{r['score']:.2f}",
            "Period (s)": f"{r['period']:.2f}",
            "Source IP": r['src_ip'],
            "Destination IP": r['dst_ip'],
            "Packets": r['packet_count']
        })
    
    return pd.DataFrame(df_data)

def main():
    st.markdown('<div class="main-header">🔍 GPU-Accelerated Beacon Detection</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Upload PCAP File")
        st.markdown("""
        <div class="info-box">
        <b>Instructions:</b><br>
        1. Upload your modified PCAP file<br>
        2. Detection runs automatically<br>
        3. View results and visualizations
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose PCAP file",
            type=['pcap'],
            help="Upload the modified PCAP file with synthetic beacons"
        )
        
        if uploaded_file is not None:
            # Clean up old HTML files
            for old_file in ["beacon_detection_2d.html", "beacon_psd_comparison.html", "beacon_spectrogram_3d.html"]:
                if os.path.exists(old_file):
                    os.remove(old_file)
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pcap') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            st.session_state['pcap_path'] = tmp_path
            st.session_state['pcap_name'] = uploaded_file.name
            st.session_state['detection_results'] = None  # Reset to trigger new detection
            st.session_state['stop_requested'] = False  # Clear stop flag on new upload
            # Clear any existing detection process and cached plots
            if 'detection_process' in st.session_state:
                del st.session_state['detection_process']
            if 'visualizer' in st.session_state:
                del st.session_state['visualizer']
            if 'cached_plots' in st.session_state:
                del st.session_state['cached_plots']  # Clear cached plots on new upload
            if 'beacon_ranks' in st.session_state:
                del st.session_state['beacon_ranks']  # Clear rank cache on new upload
            st.success(f"✅ Uploaded: {uploaded_file.name}")
        
        st.markdown("---")
        st.header("🛑 Process Control")
        
        # Check if detection is running
        detection_running = 'detection_process' in st.session_state and st.session_state.get('detection_process') is not None
        if detection_running:
            proc = st.session_state['detection_process']
            if proc.poll() is None:  # Still running
                st.warning("⚠️ Detection is running...")
            else:
                detection_running = False
        
        # Stop button
        if st.button("🛑 Stop All Background Processes", type="primary", use_container_width=True):
            # Set stop flag first
            st.session_state['stop_requested'] = True
            
            killed = kill_project_processes()
            
            # Also kill stored process if exists
            if 'detection_process' in st.session_state:
                proc = st.session_state.get('detection_process')
                if proc and proc.poll() is None:
                    try:
                        proc.terminate()
                        proc.wait(timeout=2)
                    except:
                        try:
                            proc.kill()
                        except:
                            pass
                if 'detection_process' in st.session_state:
                    del st.session_state['detection_process']
            
            # Reset detection state
            if 'detection_results' in st.session_state:
                st.session_state['detection_results'] = None
            if 'visualizer' in st.session_state:
                del st.session_state['visualizer']
            
            if killed > 0:
                st.success(f"✅ Stopped {killed} background process(es)")
            else:
                st.info("ℹ️ Stopped detection process")
            
            st.rerun()
        
        st.markdown("""
        <div class="info-box" style="font-size: 0.85em;">
        <b>Note:</b> This will kill all running Python processes for gpu_detector.py, visualize.py, and data_curation.py
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    if 'pcap_path' not in st.session_state:
        st.info("👈 Please upload a PCAP file using the sidebar to begin analysis.")
        st.markdown("""
        ### What is Beacon Detection?
        
        **Beaconing** is a pattern where infected computers periodically contact command-and-control (C2) servers.
        This tool uses **GPU-accelerated frequency analysis** to detect these periodic patterns in network traffic.
        
        ### How It Works:
        1. **Upload PCAP**: Your network capture file
        2. **GPU Analysis**: Fast Fourier Transform (FFT) finds periodic patterns
        3. **Visualization**: See beacons highlighted in interactive charts
        4. **Reporting**: Get precise locations and details of detected beacons
        """)
        return
    
    # Run detection if not already done
    if 'detection_results' not in st.session_state or st.session_state['detection_results'] is None:
        # Check if process is still running
        if 'detection_process' in st.session_state:
            proc = st.session_state.get('detection_process')
            if proc and proc.poll() is None:
                st.warning("⚠️ Detection is still running. Please wait or use Stop button.")
                return
        
        # Run detection (stop_requested is checked inside run_detection)
        results = run_detection(st.session_state['pcap_path'])
        
        # Check if stop was requested during detection
        if st.session_state.get('stop_requested', False):
            st.info("ℹ️ Detection was stopped. Upload a new file to start detection.")
            st.session_state['stop_requested'] = False
            return
        
        if results:
            st.session_state['detection_results'] = results
            # Clear stop flag on successful detection
            st.session_state['stop_requested'] = False
            # Initialize visualizer for direct plotting
            from visualize import BeaconVisualizer
            st.session_state['visualizer'] = BeaconVisualizer()
        else:
            # Only show error if not stopped by user
            if not st.session_state.get('stop_requested', False):
                st.error("Failed to run detection. Please check the console for errors.")
            return
    
    results = st.session_state['detection_results']
    visualizer = st.session_state.get('visualizer')
    
    # Header with stats - WITH TOOLTIPS
    st.markdown(f"### 📁 File: {st.session_state.get('pcap_name', 'Unknown')}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📊 Total Beacons", len(results),
                 help="Total number of detected periodic communication patterns. Each beacon represents an IP pair showing regular, repeating traffic that may indicate C2 communication.")
    with col2:
        avg_score = np.mean([r['score'] for r in results]) if results else 0
        st.metric("📈 Avg Score", f"{avg_score:.1f}",
                 help="Average periodicity score across all detected beacons. Higher scores indicate stronger periodic patterns. Score = Signal Strength / Noise Floor.")
    with col3:
        total_packets = sum([r['packet_count'] for r in results])
        st.metric("📦 Total Packets", f"{total_packets:,}",
                 help="Total number of network packets analyzed across all detected beacons. This represents the volume of periodic traffic identified.")
    with col4:
        unique_ips = len(set([r['src_ip'] for r in results] + [r['dst_ip'] for r in results]))
        st.metric("🌐 Unique IPs", unique_ips,
                 help="Number of unique IP addresses involved in detected beacon communications. Includes both source and destination IPs.")
    with col5:
        high_confidence = len([r for r in results if r['score'] > 50])
        st.metric("🔴 High Confidence", high_confidence, 
                 delta=f"{high_confidence/len(results)*100:.0f}%" if results else "0%",
                 help="Number of beacons with high confidence scores (>50). These show very strong periodic patterns and are most likely to be malicious C2 traffic.")
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Detection Results", "📈 Visualizations", "🔍 Beacon Details", "⚡ GPU Performance", "ℹ️ About"])
    
    with tab1:
        st.header("Detection Results Summary")
        st.markdown("""
        <div class="info-box">
        <b>What you're seeing:</b> This table shows all detected beacons ranked by their periodicity score.
        Higher scores indicate stronger periodic patterns (more likely to be C2 beacons).
        </div>
        """, unsafe_allow_html=True)
        
        df = create_beacon_summary_table(results)
        if df is not None:
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="beacon_detection_results.csv",
                mime="text/csv"
            )
    
    with tab2:
        st.header("Interactive Visualizations")
        
        # Cache plots in session state to avoid regenerating on every tab switch
        if 'cached_plots' not in st.session_state:
            if visualizer:
                with st.spinner("🔄 Generating visualizations (one-time, ~5 seconds)..."):
                    st.session_state['cached_plots'] = create_plots_directly(results, st.session_state['pcap_path'], visualizer)
            else:
                st.session_state['cached_plots'] = {}
        
        plots = st.session_state.get('cached_plots', {})
        
        # 2D Plots
        if '2d' in plots:
            st.subheader("📊 Event Rate & FFT Spectrum")
            st.plotly_chart(plots['2d'], use_container_width=True)
            st.markdown("""
            <div class="dsp-explanation">
            <b>DSP Explanation:</b> This visualization uses <b>Fast Fourier Transform (FFT)</b> to convert time-domain signals (packet arrival times) into frequency-domain representations. 
            The left plot shows the <b>event rate</b> - when packets arrived over time. The right plot shows the <b>FFT spectrum</b> - revealing hidden periodic patterns. 
            <b>Peaks in the FFT spectrum</b> indicate the frequency (and thus period) of beacon communications. This is the core DSP technique: transforming temporal data to expose periodicities that are invisible in raw time-series data.
            </div>
                """, unsafe_allow_html=True)
        
        # PSD Comparison
        if 'psd' in plots:
            st.subheader("📈 Power Spectral Density Comparison")
            st.plotly_chart(plots['psd'], use_container_width=True)
            st.markdown("""
            <div class="dsp-explanation">
            <b>DSP Explanation:</b> <b>Power Spectral Density (PSD)</b> measures signal power across different frequencies. This plot compares detected beacons (colored lines) against benign background traffic (gray line). 
            The <b>signal-to-noise ratio (SNR)</b> is visible as the gap between beacon lines and the noise floor. Higher peaks indicate stronger periodic signals. 
            This uses <b>Welch's method</b> for PSD estimation, which provides better frequency resolution than simple FFT magnitude. The log scale helps visualize both strong and weak signals simultaneously.
            </div>
                """, unsafe_allow_html=True)
        
        # 3D Spectrogram
        if '3d' in plots:
            st.subheader("🌐 3D Spectrogram")
            st.plotly_chart(plots['3d'], use_container_width=True)
            st.markdown("""
            <div class="dsp-explanation">
            <b>DSP Explanation:</b> A <b>spectrogram</b> shows how frequency content changes over time. This 3D visualization uses <b>Short-Time Fourier Transform (STFT)</b> - applying FFT to sliding time windows. 
            Each slice represents the frequency spectrum at a specific time. <b>Bright peaks</b> indicate strong periodic patterns at specific periods. The log-scale magnitude (Z-axis) helps reveal weak signals hidden in noise. 
            This technique is essential for detecting non-stationary periodic signals - beacons that may vary slightly over time but maintain their core periodicity.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("### 🎮 How to Use 3D View")
            st.markdown("""
            - **Rotate**: Click and drag the plot
            - **Zoom**: Scroll mouse wheel
            - **Pan**: Right-click and drag
            - **Reset**: Double-click the plot
            - **Look for**: Bright yellow/green peaks = strong beacon patterns
            """)
    
    with tab3:
        st.header("🔍 Detailed Beacon Analysis")
        
        if results:
            # IP selection with search
            st.markdown("### Select Beacon for Detailed Analysis")
            
            # Search/filter options
            col_search, col_filter = st.columns(2)
            with col_search:
                search_term = st.text_input("🔍 Search by IP:", placeholder="e.g., 147.32.84.165")
            with col_filter:
                min_score = st.slider("Minimum Score Filter:", 0.0, 200.0, 0.0, 1.0)
            
            # Cache rank mapping once (avoid expensive index() calls)
            if 'beacon_ranks' not in st.session_state:
                st.session_state['beacon_ranks'] = {
                    f"{r['src_ip']}_{r['dst_ip']}": idx + 1 
                    for idx, r in enumerate(results)
                }
            
            # Filter results (fast - just list comprehensions)
            filtered_results = results
            if search_term:
                filtered_results = [r for r in results if search_term in r['src_ip'] or search_term in r['dst_ip']]
            filtered_results = [r for r in filtered_results if r['score'] >= min_score]
            
            if not filtered_results:
                st.warning("No beacons match your search criteria.")
            else:
                # Pre-compute display strings for selectbox (faster)
                if 'filtered_display_strings' not in st.session_state or st.session_state.get('last_filter_hash') != hash((search_term, min_score)):
                    ip_pairs = [f"{r['src_ip']} → {r['dst_ip']}" for r in filtered_results]
                    display_strings = [
                        f"#{r['score']:.1f} score | {ip_pairs[i]} | {r['packet_count']} packets"
                        for i, r in enumerate(filtered_results)
                    ]
                    st.session_state['filtered_display_strings'] = display_strings
                    st.session_state['last_filter_hash'] = hash((search_term, min_score))
                else:
                    display_strings = st.session_state['filtered_display_strings']
                
                selected_idx = st.selectbox(
                    "Select Beacon:",
                    range(len(filtered_results)),
                    format_func=lambda x: display_strings[x]
                )
                
                selected_beacon = filtered_results[selected_idx]
                
                # Fast rank lookup using cached mapping
                beacon_key = f"{selected_beacon['src_ip']}_{selected_beacon['dst_ip']}"
                original_rank = st.session_state['beacon_ranks'].get(beacon_key, selected_idx + 1)
                
                # Export option for all records with selected IPs (lazy loading - only when clicked)
                st.markdown("### 📥 Export Records for Selected IPs")
                st.markdown("""
                <div class="info-box">
                Export all detection records that involve the selected IP addresses (source or destination) to CSV.
                This avoids loading large datasets into the page for better performance.
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📥 Export to CSV", key="export_ip_records"):
                    # Only compute when export is clicked (lazy loading)
                    selected_ip_records = [r for r in results if r['src_ip'] == selected_beacon['src_ip'] or 
                                         r['dst_ip'] == selected_beacon['dst_ip'] or
                                         r['src_ip'] == selected_beacon['dst_ip'] or
                                         r['dst_ip'] == selected_beacon['src_ip']]
                    if selected_ip_records:
                        # Use cached ranks (already created above)
                        ranks = st.session_state.get('beacon_ranks', {})
                        
                        ip_df = pd.DataFrame([{
                            "Rank": ranks.get(f"{r['src_ip']}_{r['dst_ip']}", 0),
                            "Score": f"{r['score']:.2f}",
                            "Period (s)": f"{r['period']:.2f}",
                            "Source IP": r['src_ip'],
                            "Destination IP": r['dst_ip'],
                            "Packets": r['packet_count']
                        } for r in selected_ip_records])
                        
                        csv = ip_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"beacon_records_{selected_beacon['src_ip'].replace('.', '_')}_{selected_beacon['dst_ip'].replace('.', '_')}.csv",
                            mime="text/csv",
                            key="download_ip_csv"
                        )
                        st.success(f"✅ Found {len(selected_ip_records)} records. Click download to save.")
                    else:
                        st.info("ℹ️ No additional records found for these IPs.")
                
                # Beacon details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="beacon-indicator">
                    <h3>🚨 Beacon #{original_rank} (Ranked)</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.metric("Periodicity Score", f"{selected_beacon['score']:.2f}", 
                             delta=f"{'Very High' if selected_beacon['score'] > 50 else 'High' if selected_beacon['score'] > 20 else 'Medium'}")
                    st.metric("Detected Period", f"{selected_beacon['period']:.2f} seconds",
                             help="How often this beacon repeats")
                    st.metric("Total Packets", f"{selected_beacon['packet_count']:,}",
                             help="Number of packets in this communication")
                    st.metric("Peak Magnitude", f"{selected_beacon['peak_magnitude']:.2e}",
                             help="FFT peak strength")
                
                with col2:
                    st.markdown("### 📍 IP Addresses")
                    st.code(f"Source IP:     {selected_beacon['src_ip']}")
                    st.code(f"Destination:   {selected_beacon['dst_ip']}")
                    
                    st.markdown("### 🎯 Confidence Level")
                    if selected_beacon['score'] > 50:
                        st.error("🔴 **HIGH CONFIDENCE**: Very strong periodic pattern - likely C2 beacon")
                    elif selected_beacon['score'] > 20:
                        st.warning("🟡 **MEDIUM CONFIDENCE**: Moderate periodic pattern - suspicious")
                    else:
                        st.info("🟢 **LOW CONFIDENCE**: Weak periodic pattern - investigate further")
                
                # Methodology section
                st.markdown("### 🔬 Detection Methodology")
                st.markdown("""
                <div class="dsp-explanation">
                <b>DSP Techniques Used:</b>
                <ul>
                <li><b>Fast Fourier Transform (FFT):</b> Converts packet timestamps into frequency domain to identify periodic patterns. 
                The FFT reveals hidden periodicities by decomposing the time-series signal into its frequency components.</li>
                <li><b>Hanning Window:</b> Applied before FFT to reduce spectral leakage - a phenomenon where energy from one frequency 
                "leaks" into adjacent frequencies, making detection less accurate.</li>
                <li><b>Linear Detrending:</b> Removes long-term trends from the signal, ensuring we detect true periodic patterns 
                rather than gradual changes over time.</li>
                <li><b>Power Spectral Density (PSD):</b> Calculates signal power at each frequency, providing a robust measure of 
                periodicity strength that's less sensitive to noise.</li>
                <li><b>Periodicity Score:</b> Computed as Peak Magnitude / Median Noise Floor. This ratio quantifies how much stronger 
                the periodic signal is compared to background noise - the higher the score, the more confident we are it's a beacon.</li>
                </ul>
                <b>Why FFT and not Wavelet?</b> FFT is ideal for detecting constant-period beacons (which is the case for most C2 traffic). 
                Wavelet transforms are better for signals with varying frequencies over time, but beacon periods are typically stable. 
                FFT also provides better frequency resolution for periodic signals and is computationally faster on GPU.
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                ### 📊 What This Means:
                
                **Pattern Detection:**
                - This IP pair communicates every **{selected_beacon['period']:.1f} seconds**
                - Pattern strength: **{selected_beacon['score']:.2f}x** stronger than random noise
                - Total observations: **{selected_beacon['packet_count']:,} packets**
                
                **Security Implication:**
                - Periodic communication suggests automated beaconing
                - Higher scores = more likely to be malicious C2 traffic
                - Investigate source IP for potential infection
                """)
                
                # Pattern visualization
                st.markdown("### 📈 Pattern Visualization")
                st.markdown(f"""
                **Repeating Pattern:**
                ```
                Time:  0s    {selected_beacon['period']:.1f}s    {selected_beacon['period']*2:.1f}s    {selected_beacon['period']*3:.1f}s
                Packets:  █        █              █              █
                ```
                This shows packets arriving at regular intervals, indicating beaconing behavior.
                """)
    
    with tab4:
        st.header("⚡ GPU Performance & Statistics")
        
        # Get GPU stats
        gpu_stats = get_gpu_stats()
        if gpu_stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("GPU Memory Used", f"{gpu_stats['memory_used_mb']} MB",
                         help="Current GPU memory usage")
            with col2:
                st.metric("GPU Memory Total", f"{gpu_stats['memory_total_mb']} MB",
                         help="Total available GPU memory")
            with col3:
                st.metric("GPU Utilization", f"{gpu_stats['gpu_util_percent']}%",
                         help="Current GPU compute utilization percentage")
            with col4:
                st.metric("GPU Temperature", f"{gpu_stats['temperature_c']}°C",
                         help="Current GPU temperature")
        
        st.markdown("### 🔧 GPU/CUDA Functions Used")
        st.markdown("""
        <div class="info-box">
        <b>RAPIDS Stack Functions:</b>
        <ul>
        <li><b>cuDF:</b> GPU-accelerated DataFrames for loading and processing PCAP data</li>
        <li><b>CuPy:</b> GPU-accelerated NumPy operations (array operations, FFT)</li>
        <li><b>cuSignal:</b> GPU-accelerated signal processing (detrending, windowing)</li>
        <li><b>cuFFT:</b> CUDA Fast Fourier Transform library for batched FFT operations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if results:
            st.markdown("### 📊 Processing Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("IP Pairs Processed", len(results),
                         help="Number of unique IP pairs analyzed for periodic patterns")
            with col2:
                total_pairs = len(set([(r['src_ip'], r['dst_ip']) for r in results]))
                st.metric("Unique IP Pairs", total_pairs,
                         help="Number of unique source-destination IP combinations")
            with col3:
                st.metric("Batched Operations", "1",
                         help="All IP pairs processed in a single batched GPU operation for maximum throughput")
            
            st.markdown("### 💾 Memory Operations")
            st.markdown("""
            <div class="info-box">
            <b>GPU Memory Operations:</b>
            <ul>
            <li><b>Read:</b> PCAP data loaded into cuDF DataFrame on GPU memory</li>
            <li><b>Write:</b> Time-series matrices, FFT results, and scores computed on GPU</li>
            <li><b>Batching:</b> All IP pairs processed simultaneously in a single 2D matrix operation</li>
            <li><b>Memory Pool:</b> CuPy memory pool manages GPU memory allocation (8GB limit for RTX 5090)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab5:
        st.header("ℹ️ About This Tool")
        
        st.markdown("""
        ### What is Beacon Detection?
        
        **Beaconing** is a communication pattern where malware-infected computers periodically contact 
        command-and-control (C2) servers to receive instructions or exfiltrate data.
        
        ### How Detection Works
        
        1. **Time-Series Conversion**: Network packets are converted into time-series data
        2. **Frequency Analysis**: Fast Fourier Transform (FFT) identifies periodic patterns
        3. **Scoring**: Each IP pair gets a "periodicity score" = Signal Strength / Noise Floor
        4. **Ranking**: Higher scores = stronger periodic patterns = likely beacons
        
        ### Understanding the Visualizations
        
        - **Event Rate Plot**: Shows when packets occurred over time
        - **FFT Spectrum**: Reveals hidden periodic patterns (peaks = beacon frequency)
        - **PSD Comparison**: Compares beacons vs. normal traffic (shows signal-to-noise ratio)
        - **3D Spectrogram**: Time × Frequency × Strength (shows how patterns evolve)
        
        ### Performance
        
        - **GPU-Accelerated**: Uses NVIDIA RTX 5090 for fast processing
        - **Throughput**: 37,000+ IP pairs/second
        - **Technology**: RAPIDS (cuDF, CuPy, cuSignal)
        """)
        
        st.markdown("---")
        st.markdown("**Project**: GPU-Accelerated Frequency Domain Analysis for C2 Detection")

if __name__ == "__main__":
    main()
