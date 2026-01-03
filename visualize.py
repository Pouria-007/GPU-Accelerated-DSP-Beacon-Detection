#!/usr/bin/env python3
"""
Visualization module for GPU beacon detection results
Generates 2D plots, Power Spectral Density comparisons, and 3D spectrograms with log-scale
"""

import sys
import json
import numpy as np
import cupy as cp
import cusignal
import cudf
from scapy.all import PcapReader, IP
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
from typing import Dict, List, Tuple, Optional
import socket
import os

# Try to import faster PCAP parser
try:
    import dpkt
    HAS_DPKT = True
except ImportError:
    HAS_DPKT = False


class BeaconVisualizer:
    """
    Visualization tools for beacon detection results.
    """
    
    def __init__(self):
        """Initialize visualizer."""
        self.pcap_cache = {}  # Cache PCAP data to avoid re-reading
    
    def _load_pcap_cache(self, pcap_path: str, max_pairs: int = 10):
        """
        Load PCAP once and cache timestamps for top IP pairs - OPTIMIZED with dpkt.
        This avoids reading the PCAP multiple times.
        """
        if pcap_path in self.pcap_cache:
            return self.pcap_cache[pcap_path]
        
        print(f"[*] Loading PCAP cache (optimized, ~2-3s)...")
        cache = {}
        
        # Use fast dpkt parser if available
        use_dpkt = HAS_DPKT
        if use_dpkt:
            try:
                packet_count = 0
                with open(pcap_path, 'rb') as f:
                    pcap = dpkt.pcap.Reader(f)
                    for ts, buf in pcap:
                        try:
                            eth = dpkt.ethernet.Ethernet(buf)
                            if isinstance(eth.data, dpkt.ip.IP):
                                ip = eth.data
                                src_ip = socket.inet_ntoa(ip.src)
                                dst_ip = socket.inet_ntoa(ip.dst)
                                
                                pair_key = (src_ip, dst_ip)
                                if pair_key not in cache:
                                    cache[pair_key] = []
                                cache[pair_key].append(ts)
                                
                                packet_count += 1
                                if packet_count % 100000 == 0:
                                    print(f"[*] Cached {packet_count:,} packets...")
                        except:
                            continue
                
                print(f"[✓] Using dpkt (FAST): Cached {packet_count:,} packets")
            except Exception as e:
                print(f"[!] dpkt failed: {e}, falling back to Scapy")
                use_dpkt = False
        
        # Fallback to Scapy (slower)
        if not use_dpkt:
            packet_count = 0
            with PcapReader(pcap_path) as pcap_reader:
                for pkt in pcap_reader:
                    if IP in pkt:
                        src_ip = pkt[IP].src
                        dst_ip = pkt[IP].dst
                        timestamp = float(pkt.time)
                        
                        pair_key = (src_ip, dst_ip)
                        if pair_key not in cache:
                            cache[pair_key] = []
                        cache[pair_key].append(timestamp)
                        
                        packet_count += 1
                        if packet_count % 100000 == 0:
                            print(f"[*] Cached {packet_count:,} packets...")
        
        # Convert to numpy arrays
        for key in cache:
            cache[key] = np.array(cache[key])
        
        self.pcap_cache[pcap_path] = cache
        print(f"[✓] PCAP cache loaded: {len(cache)} IP pairs")
        return cache
    
    def load_detection_results(self, results_file: str) -> List[Dict]:
        """
        Load detection results from JSON file.
        
        Args:
            results_file: Path to JSON results file
            
        Returns:
            List of detection results
        """
        with open(results_file, 'r') as f:
            results = json.load(f)
        return results
    
    def create_time_series_from_pcap(
        self,
        pcap_path: str,
        src_ip: str,
        dst_ip: str,
        bin_size: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time-series from PCAP for a specific IP pair using cached data.
        
        Args:
            pcap_path: Path to PCAP file
            src_ip: Source IP
            dst_ip: Destination IP
            bin_size: Size of time bins in seconds
            
        Returns:
            Tuple of (time_axis, event_rate)
        """
        # Use cached data instead of re-reading PCAP
        cache = self._load_pcap_cache(pcap_path)
        pair_key = (src_ip, dst_ip)
        
        if pair_key not in cache:
            return None, None
        
        timestamps = cache[pair_key]
        
        if len(timestamps) == 0:
            return None, None
        
        min_time = timestamps.min()
        max_time = timestamps.max()
        num_bins = int((max_time - min_time) / bin_size) + 1
        
        # Create time axis
        time_axis = np.arange(num_bins) * bin_size
        
        # Create event rate (packets per bin) using numpy bincount
        bin_indices = ((timestamps - min_time) / bin_size).astype(int)
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        event_rate = np.bincount(bin_indices, minlength=num_bins).astype(float)
        
        return time_axis, event_rate
    
    def compute_fft_spectrum(
        self,
        event_rate: np.ndarray,
        sample_rate: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT spectrum for visualization.
        
        Args:
            event_rate: Event rate time-series
            sample_rate: Sample rate in Hz
            
        Returns:
            Tuple of (frequencies, magnitude)
        """
        # Apply windowing
        windowed = np.hanning(len(event_rate))
        windowed_series = event_rate * windowed
        
        # Detrend on GPU
        detrended_gpu = cusignal.detrend(cp.asarray(windowed_series, dtype=cp.float32), type='linear')
        
        # FFT on GPU
        fft_result_gpu = cp.fft.rfft(detrended_gpu)
        fft_magnitude = cp.abs(fft_result_gpu).get()
        frequencies = cp.fft.rfftfreq(len(event_rate), 1.0 / sample_rate).get()
        
        return frequencies, fft_magnitude
    
    def compute_psd(
        self,
        event_rate: np.ndarray,
        sample_rate: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density (PSD) for visualization.
        
        Args:
            event_rate: Event rate time-series
            sample_rate: Sample rate in Hz
            
        Returns:
            Tuple of (periods, psd)
        """
        # Apply windowing
        windowed = np.hanning(len(event_rate))
        windowed_series = event_rate * windowed
        
        # Detrend on GPU
        detrended_gpu = cusignal.detrend(cp.asarray(windowed_series, dtype=cp.float32), type='linear')
        
        # FFT on GPU
        fft_result_gpu = cp.fft.rfft(detrended_gpu)
        
        # Power Spectral Density (squared magnitude normalized) on GPU
        psd_gpu = (cp.abs(fft_result_gpu) ** 2) / len(event_rate)
        psd = psd_gpu.get()
        
        # Convert to dB scale
        psd_db = 10 * np.log10(psd + 1e-10)  # Add small value to avoid log(0)
        
        frequencies = np.fft.rfftfreq(len(event_rate), 1.0 / sample_rate)
        
        # Convert frequency to period (skip DC)
        periods = 1.0 / frequencies[1:]
        psd_db = psd_db[1:]
        
        return periods, psd_db
    
    def create_psd_comparison_plot(
        self,
        results: List[Dict],
        pcap_path: str,
        top_n: int = 3,
        output_file: str = "beacon_psd_comparison.html"
    ):
        """
        Create Power Spectral Density (PSD) comparison plot.
        Shows top N beacons overlaid with benign background spectrum.
        
        Args:
            results: List of detection results
            pcap_path: Path to PCAP file
            top_n: Number of top candidates to plot
            output_file: Output HTML file path
        """
        print(f"[*] Creating PSD comparison plot for top {top_n} beacons...")
        
        fig = go.Figure()
        
        # Compute benign background using cached data
        cache = self._load_pcap_cache(pcap_path)
        beacon_pairs = {(r['src_ip'], r['dst_ip']) for r in results[:top_n]}
        
        # Find a non-beacon pair from cache
        benign_timestamps = None
        for pair_key, timestamps in cache.items():
            if pair_key not in beacon_pairs and len(timestamps) > 100:
                benign_timestamps = timestamps[:1000]  # Sample first 1000
                break
        
        if benign_timestamps is not None:
            # Create benign time-series
            min_time = benign_timestamps.min()
            max_time = benign_timestamps.max()
            num_bins = int((max_time - min_time) / 1.0) + 1
            
            bin_indices = ((benign_timestamps - min_time) / 1.0).astype(int)
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)
            benign_event_rate = np.bincount(bin_indices, minlength=num_bins).astype(float)
            
            # Compute PSD for benign
            periods_benign, psd_benign = self.compute_psd(benign_event_rate)
            
            # Add benign background
            fig.add_trace(go.Scatter(
                x=periods_benign,
                y=psd_benign,
                mode='lines',
                name='Benign Background',
                line=dict(color='gray', width=1, dash='dash'),
                opacity=0.5
            ))
        
        # Plot top N beacons
        colors = ['red', 'orange', 'yellow']
        for idx, result in enumerate(results[:top_n]):
            time_axis, event_rate = self.create_time_series_from_pcap(
                pcap_path,
                result['src_ip'],
                result['dst_ip']
            )
            
            if time_axis is None:
                continue
            
            # Compute PSD
            periods, psd = self.compute_psd(event_rate)
            
            # Add trace
            fig.add_trace(go.Scatter(
                x=periods,
                y=psd,
                mode='lines',
                name=f"Beacon #{idx+1}: {result['src_ip']} → {result['dst_ip']}<br>Score: {result['score']:.2f}, Period: {result['period']:.2f}s",
                line=dict(color=colors[idx % len(colors)], width=2)
            ))
            
            # Add vertical line at detected period
            fig.add_vline(
                x=result['period'],
                line=dict(color=colors[idx % len(colors)], width=1, dash='dot'),
                annotation_text=f"{result['period']:.1f}s"
            )
        
        fig.update_layout(
            title="Power Spectral Density Comparison: Beacons vs. Benign Traffic",
            xaxis_title="Period (seconds)",
            yaxis_title="Power Spectral Density (dB)",
            xaxis_type="log",
            height=600,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        fig.write_html(output_file)
        print(f"[✓] PSD comparison plot saved to {output_file}")
    
    def create_2d_plots(
        self,
        results: List[Dict],
        pcap_path: str,
        top_n: int = 5,
        output_file: str = "beacon_detection_2d.html"
    ):
        """
        Create 2D plots: Event-rate and FFT spectrum for top N candidates.
        
        Args:
            results: List of detection results
            pcap_path: Path to PCAP file
            top_n: Number of top candidates to plot
            output_file: Output HTML file path
        """
        top_results = results[:top_n]
        
        # Create subplots: one row per candidate, two columns (event rate + FFT)
        fig = make_subplots(
            rows=top_n,
            cols=2,
            subplot_titles=[
                f"{r['src_ip']} → {r['dst_ip']} (Score: {r['score']:.2f})"
                for r in top_results
                for _ in [0, 1]  # Two plots per result
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        for idx, result in enumerate(top_results, 1):
            # Get time-series from PCAP
            time_axis, event_rate = self.create_time_series_from_pcap(
                pcap_path,
                result['src_ip'],
                result['dst_ip']
            )
            
            if time_axis is None:
                continue
            
            # Plot event rate
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=event_rate,
                    mode='lines',
                    name=f"Event Rate {idx}",
                    line=dict(width=1),
                    showlegend=False
                ),
                row=idx,
                col=1
            )
            
            # Compute and plot FFT spectrum
            frequencies, magnitude = self.compute_fft_spectrum(event_rate)
            
            # Convert frequency to period for better interpretation
            periods = 1.0 / frequencies[1:]  # Skip DC component
            magnitude_period = magnitude[1:]
            
            fig.add_trace(
                go.Scatter(
                    x=periods,
                    y=magnitude_period,
                    mode='lines',
                    name=f"FFT Spectrum {idx}",
                    line=dict(width=2),
                    showlegend=False
                ),
                row=idx,
                col=2
            )
        
        # Update axes labels
        for i in range(1, top_n + 1):
            fig.update_xaxes(title_text="Time (s)", row=i, col=1)
            fig.update_yaxes(title_text="Event Rate", row=i, col=1)
            fig.update_xaxes(title_text="Period (s)", row=i, col=2)
            fig.update_yaxes(title_text="Magnitude", row=i, col=2)
        
        fig.update_layout(
            title="Beacon Detection: Event Rate and FFT Spectrum (Top 5)",
            height=300 * top_n,
            showlegend=False
        )
        
        fig.write_html(output_file)
        print(f"[✓] 2D plots saved to {output_file}")
    
    def create_3d_spectrogram(
        self,
        pcap_path: str,
        src_ip: str,
        dst_ip: str,
        window_size: int = 100,
        overlap: int = 50,
        output_file: str = "beacon_spectrogram_3d.html"
    ):
        """
        Create 3D spectrogram: Frequency vs. Time vs. Magnitude.
        
        Args:
            pcap_path: Path to PCAP file
            src_ip: Source IP
            dst_ip: Destination IP
            window_size: Size of sliding window for spectrogram
            overlap: Overlap between windows
            output_file: Output HTML file path
        """
        # Get time-series
        time_axis, event_rate = self.create_time_series_from_pcap(
            pcap_path,
            src_ip,
            dst_ip
        )
        
        if time_axis is None:
            print(f"[!] No data found for {src_ip} → {dst_ip}")
            return
        
        # Create sliding window spectrogram
        step = window_size - overlap
        time_windows = []
        frequency_axis = None
        magnitude_list = []
        
        for i in range(0, len(event_rate) - window_size, step):
            window_data = event_rate[i:i + window_size]
            window_time = time_axis[i + window_size // 2]  # Center time of window
            
            # Compute FFT for this window (GPU)
            windowed = np.hanning(len(window_data))
            windowed_series = window_data * windowed
            detrended_gpu = cusignal.detrend(cp.asarray(windowed_series, dtype=cp.float32), type='linear')
            
            fft_result_gpu = cp.fft.rfft(detrended_gpu)
            fft_magnitude = cp.abs(fft_result_gpu).get()
            
            if frequency_axis is None:
                frequencies = cp.fft.rfftfreq(len(window_data), 1.0).get()
                # Convert to periods for better interpretation
                frequency_axis = 1.0 / frequencies[1:]  # Skip DC
                expected_size = len(frequency_axis)
            
            # Skip DC component and ensure consistent size
            fft_magnitude_no_dc = fft_magnitude[1:]
            if len(fft_magnitude_no_dc) == expected_size:
                time_windows.append(window_time)
                magnitude_list.append(fft_magnitude_no_dc)
        
        # Convert to numpy arrays (now all same size)
        time_windows = np.array(time_windows)
        magnitude_matrix = np.array(magnitude_list)
        
        # Apply log-scale to magnitude for better visualization of low-power signals
        magnitude_matrix_log = np.log10(magnitude_matrix + 1e-10)  # Add small value to avoid log(0)
        
        # Create 3D surface plot with log-scale magnitude
        fig = go.Figure(data=[go.Surface(
            x=time_windows,
            y=frequency_axis,
            z=magnitude_matrix_log.T,  # Transpose for correct orientation
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Log10(Magnitude)")
        )])
        
        fig.update_layout(
            title=f"3D Spectrogram (Log-Scale): {src_ip} → {dst_ip}",
            scene=dict(
                xaxis_title="Time (s)",
                yaxis_title="Period (s)",
                zaxis_title="Log10(Magnitude)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            width=1200,
            height=800
        )
        
        fig.write_html(output_file)
        print(f"[✓] 3D spectrogram saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize beacon detection results"
    )
    parser.add_argument(
        "--results",
        default="detection_results.json",
        help="Path to detection results JSON file"
    )
    parser.add_argument(
        "--pcap",
        default="botnet-capture-20110810-neris_modified.pcap",
        help="Path to PCAP file"
    )
    parser.add_argument(
        "--neris-ip",
        default="147.32.84.165",
        help="Neris infected host IP for 3D spectrogram"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top candidates for 2D plots (default: 5)"
    )
    parser.add_argument(
        "--2d-only",
        dest="plot_2d_only",
        action="store_true",
        help="Only generate 2D plots"
    )
    parser.add_argument(
        "--3d-only",
        dest="plot_3d_only",
        action="store_true",
        help="Only generate 3D spectrogram"
    )
    parser.add_argument(
        "--psd-only",
        dest="plot_psd_only",
        action="store_true",
        help="Only generate PSD comparison plot"
    )
    parser.add_argument(
        "--no-psd",
        dest="skip_psd",
        action="store_true",
        help="Skip PSD comparison plot"
    )
    
    args = parser.parse_args()
    
    visualizer = BeaconVisualizer()
    
    # Load results
    try:
        results = visualizer.load_detection_results(args.results)
        print(f"[*] Loaded {len(results)} detection results")
    except FileNotFoundError:
        print(f"[!] Results file not found: {args.results}")
        print("[!] Run gpu_detector.py first to generate results")
        sys.exit(1)
    
    # Generate PSD comparison plot
    if args.plot_psd_only or (not args.plot_2d_only and not args.plot_3d_only and not args.skip_psd):
        print("\n[*] Generating PSD comparison plot...")
        visualizer.create_psd_comparison_plot(
            results,
            args.pcap,
            top_n=min(3, len(results)),
            output_file="beacon_psd_comparison.html"
        )
        
        if args.plot_psd_only:
            return
    
    # Generate 2D plots
    if not args.plot_3d_only and not args.plot_psd_only:
        print("\n[*] Generating 2D plots...")
        visualizer.create_2d_plots(
            results,
            args.pcap,
            top_n=args.top_n,
            output_file="beacon_detection_2d.html"
        )
    
    # Generate 3D spectrogram for Neris IP
    if not args.plot_2d_only and not args.plot_psd_only:
        print("\n[*] Generating 3D spectrogram for Neris IP...")
        # Find Neris IP in results or use first destination
        neris_dst = None
        for result in results:
            if result['src_ip'] == args.neris_ip:
                neris_dst = result['dst_ip']
                break
        
        if neris_dst is None:
            # Use first result's destination
            if results:
                neris_dst = results[0]['dst_ip']
                print(f"[*] Using first result destination: {neris_dst}")
            else:
                print("[!] No results available for 3D spectrogram")
                return
        
        visualizer.create_3d_spectrogram(
            args.pcap,
            args.neris_ip,
            neris_dst,
            output_file="beacon_spectrogram_3d.html"
        )
    
    print("\n[✓] Visualization complete!")


if __name__ == "__main__":
    main()

