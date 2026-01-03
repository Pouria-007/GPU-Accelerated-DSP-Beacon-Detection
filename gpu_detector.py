#!/usr/bin/env python3
"""
GPU-Accelerated Frequency Domain Analysis for Network Command & Control (C2) Detection  
Optimized for NVIDIA RTX 5090 (Blackwell) with automatic cuDF fallback
"""

import sys
import cupy as cp
import cusignal
import cudf
from scapy.all import PcapReader, IP
from typing import Dict, List, Tuple
import argparse
from rich.console import Console
from rich.table import Table
from rich import print as rprint
import time
import os
import socket

# Try to import faster PCAP parser
try:
    import dpkt
    HAS_DPKT = True
except ImportError:
    HAS_DPKT = False

# cuDF is REQUIRED - no fallback
import cudf


class GPUBeaconDetector:
    """
    GPU-accelerated beacon detection using FFT-based periodicity analysis.
    Automatically uses cuDF (GPU) or pandas (CPU) for dataframes.
    """
    
    def __init__(self, dtype=cp.float32):
        """
        Initialize detector.
        
        Args:
            dtype: Data type for FFT operations (float32 for performance, float64 for precision)
        """
        self.dtype = dtype
        self.console = Console()
        self.time_series_cache = {}
        
        # Check GPU availability
        try:
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=8 * 2**30)  # 8GB limit for RTX 5090
            rprint(f"[green]✓[/green] GPU detected: {cp.cuda.Device().compute_capability}")
            rprint(f"[green]✓[/green] Using dtype: {dtype}")
        except Exception as e:
            rprint(f"[red]✗[/red] GPU initialization failed: {e}")
            sys.exit(1)
        
        # Verify cuDF works (test with drop_duplicates instead of unique)
        try:
            test_df = cudf.DataFrame({'a': [1, 2, 3, 1]})
            _ = test_df.drop_duplicates()  # Use drop_duplicates instead of unique
            rprint(f"[green]✓[/green] cuDF GPU dataframes: ENABLED")
            rprint(f"[green]✓[/green] cuDF version: {cudf.__version__}")
        except Exception as e:
            rprint(f"[red]✗[/red] cuDF compatibility test failed: {e}")
            rprint(f"[red]✗[/red] This GPU may not be compatible with cuDF")
            sys.exit(1)
    
    def load_pcap_to_cudf(self, pcap_path: str, loop_count: int = 1) -> cudf.DataFrame:
        """
        Load PCAP timestamps into cuDF DataFrame (GPU) - ULTRA-FAST VERSION.
        
        Uses dpkt (C-based) for 10-50x faster parsing than Scapy.
        Falls back to optimized Scapy if dpkt not available.
        
        Args:
            pcap_path: Path to PCAP file
            loop_count: Number of times to duplicate data for stress testing
            
        Returns:
            cuDF DataFrame with columns: timestamp, src_ip, dst_ip
        """
        rprint(f"[*] Loading PCAP: {pcap_path}")
        total_start = time.time()
        
        file_size = os.path.getsize(pcap_path)
        rprint(f"[*] File size: {file_size/1e6:.1f} MB")
        
        # Try fast dpkt parser first
        use_dpkt = HAS_DPKT
        if use_dpkt:
            try:
                t0 = time.time()
                data = []
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
                                data.append({
                                    'timestamp': ts,
                                    'src_ip': src_ip,
                                    'dst_ip': dst_ip
                                })
                                packet_count += 1
                        except:
                            continue
                
                parse_time = time.time() - t0
                rprint(f"[green]✓[/green] Using dpkt (FAST): Parsed {packet_count:,} IP packets in {parse_time:.2f}s")
                rprint(f"[*] Parsing rate: {packet_count/parse_time:.0f} packets/sec")
                
            except Exception as e:
                rprint(f"[yellow]![/yellow] dpkt failed: {e}, falling back to Scapy")
                use_dpkt = False
        
        # Fallback to Scapy (slower but more compatible)
        if not use_dpkt:
            t0 = time.time()
            data = []
            packet_count = 0
            
            with PcapReader(pcap_path) as pcap_reader:
                for pkt in pcap_reader:
                    if IP in pkt:
                        src_ip = pkt[IP].src
                        dst_ip = pkt[IP].dst
                        timestamp = float(pkt.time)
                        data.append({
                            'timestamp': timestamp,
                            'src_ip': src_ip,
                            'dst_ip': dst_ip
                        })
                        packet_count += 1
                        
                        if packet_count % 100000 == 0:
                            rprint(f"[*] Read {packet_count:,} packets...")
            
            parse_time = time.time() - t0
            rprint(f"[*] Parsed {packet_count:,} IP packets in {parse_time:.2f}s (Scapy)")
            rprint(f"[*] Parsing rate: {packet_count/parse_time:.0f} packets/sec")
        
        # Duplicate data for stress testing
        if loop_count > 1:
            original_len = len(data)
            data = data * loop_count
            rprint(f"[*] Duplicated data {loop_count}x: {original_len:,} -> {len(data):,} packets")
        
        # Convert to cuDF DataFrame (GPU) - optimized batch creation
        t0 = time.time()
        if len(data) > 500000:
            # Chunked creation for very large datasets
            chunk_size = 200000
            chunks = []
            for i in range(0, len(data), chunk_size):
                chunk_df = cudf.DataFrame(data[i:i+chunk_size])
                chunks.append(chunk_df)
            df = cudf.concat(chunks, ignore_index=True)
        else:
            df = cudf.DataFrame(data)
        
        cudf_time = time.time() - t0
        rprint(f"[*] Created cuDF DataFrame in {cudf_time:.2f}s")
        
        total_time = time.time() - total_start
        rprint(f"[green]✓[/green] Loaded {len(df):,} packets to GPU (cuDF) in {total_time:.2f}s")
        
        # Performance breakdown
        if total_time > 0:
            rprint(f"[*] Breakdown: Parse={parse_time:.2f}s ({parse_time/total_time*100:.1f}%), "
                  f"cuDF={cudf_time:.2f}s ({cudf_time/total_time*100:.1f}%)")
        
        return df
    
    def create_batched_time_series(
        self,
        df: cudf.DataFrame,
        bin_size: float = 1.0,
        max_ip_pairs: int = None
    ) -> Tuple[cp.ndarray, List[Tuple[str, str]], int]:
        """
        Create batched time-series matrix for ALL IP pairs at once using cuDF (GPU).
        
        Args:
            df: cuDF DataFrame with timestamps (on GPU)
            bin_size: Size of time bins in seconds
            max_ip_pairs: Maximum number of IP pairs to process
            
        Returns:
            Tuple of (batched_time_series, ip_pair_list, num_bins)
        """
        rprint(f"[*] Creating batched time-series using GPU (cuDF)...")
        start_time = time.time()
        
        # Get global time bounds (GPU operations)
        min_time = float(df['timestamp'].min())
        max_time = float(df['timestamp'].max())
        num_bins = int((max_time - min_time) / bin_size) + 1
        
        rprint(f"[*] Time range: {max_time - min_time:.2f}s, Bins: {num_bins:,}")
        
        # Create IP pair identifier (GPU operation)
        df['ip_pair'] = df['src_ip'] + '_' + df['dst_ip']
        
        # Get unique IP pairs using cuDF (GPU operation)
        # Use drop_duplicates instead of unique() to avoid RTX 5090 compatibility issue
        unique_df = df[['ip_pair']].drop_duplicates()
        unique_pairs = unique_df['ip_pair'].to_pandas().values
        
        if max_ip_pairs and len(unique_pairs) > max_ip_pairs:
            rprint(f"[*] Limiting to {max_ip_pairs:,} IP pairs (out of {len(unique_pairs):,})")
            unique_pairs = unique_pairs[:max_ip_pairs]
        
        num_pairs = len(unique_pairs)
        rprint(f"[*] Processing {num_pairs:,} unique IP pairs on GPU...")
        
        # Initialize batched time-series matrix on GPU
        time_series_matrix = cp.zeros((num_pairs, num_bins), dtype=self.dtype)
        
        # TRUE BATCHED PROCESSING - No Python loops!
        # Add bin indices to dataframe (GPU operation)
        df['bin_idx'] = ((df['timestamp'] - min_time) / bin_size).astype('int32')
        df['bin_idx'] = df['bin_idx'].clip(0, num_bins - 1)
        
        # Create mapping from IP pair string to integer index
        # Use pandas for mapping and groupby (cuDF groupby fails on RTX 5090)
        t0 = time.time()
        pair_to_idx = {pair: i for i, pair in enumerate(unique_pairs)}
        df_pandas = df.to_pandas()
        df_pandas['pair_idx'] = df_pandas['ip_pair'].map(pair_to_idx)
        rprint(f"[*] Created pair mapping in {time.time()-t0:.2f}s")
        
        # Group by pair_idx and bin_idx using pandas (faster and more compatible)
        t0 = time.time()
        grouped_pandas = df_pandas.groupby(['pair_idx', 'bin_idx']).size().reset_index(name='counts')
        rprint(f"[*] Pandas groupby completed in {time.time()-t0:.2f}s ({len(grouped_pandas):,} groups)")
        
        # Convert to CuPy arrays for scatter_add
        t0 = time.time()
        pair_indices_cp = cp.asarray(grouped_pandas['pair_idx'].values, dtype=cp.int32)
        bin_indices_cp = cp.asarray(grouped_pandas['bin_idx'].values, dtype=cp.int32)
        counts_cp = cp.asarray(grouped_pandas['counts'].values, dtype=self.dtype)
        rprint(f"[*] Converted to CuPy arrays in {time.time()-t0:.2f}s")
        
        # Fill time_series_matrix using vectorized operations
        t0 = time.time()
        # Process in batches to avoid memory issues and improve speed
        batch_size = 1000
        for batch_start in range(0, num_pairs, batch_size):
            batch_end = min(batch_start + batch_size, num_pairs)
            batch_pairs = cp.arange(batch_start, batch_end, dtype=cp.int32)
            
            # Vectorized mask for this batch
            mask = cp.isin(pair_indices_cp, batch_pairs)
            if cp.any(mask):
                batch_pair_indices = pair_indices_cp[mask]
                batch_bin_indices = bin_indices_cp[mask]
                batch_counts = counts_cp[mask]
                
                # Process each pair in batch
                for pair_idx in batch_pairs:
                    pair_mask = batch_pair_indices == pair_idx
                    if cp.any(pair_mask):
                        pair_bins = batch_bin_indices[pair_mask]
                        pair_counts = batch_counts[pair_mask]
                        counts = cp.bincount(pair_bins, weights=pair_counts, minlength=num_bins).astype(self.dtype)
                        time_series_matrix[pair_idx] = counts
        
        cp.cuda.Stream.null.synchronize()
        rprint(f"[*] GPU matrix fill completed in {time.time()-t0:.2f}s")
        
        # Convert IP pair strings back to tuples
        ip_pair_list = [tuple(pair.split('_')) for pair in unique_pairs]
        
        elapsed = time.time() - start_time
        rprint(f"[green]✓[/green] Created batched time-series matrix: {time_series_matrix.shape} in {elapsed:.2f}s")
        
        return time_series_matrix, ip_pair_list, num_bins
    
    def compute_batched_periodicity_scores(
        self,
        time_series_matrix: cp.ndarray,
        sample_rate: float = 1.0
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        Compute periodicity scores for ALL IP pairs in one batched GPU operation.
        
        Args:
            time_series_matrix: 2D array of shape (num_ip_pairs, num_bins)
            sample_rate: Sample rate in Hz (1.0 for 1s bins)
            
        Returns:
            Tuple of (periodicity_scores, estimated_periods, peak_magnitudes)
        """
        rprint("[*] Computing batched FFT and periodicity scores on GPU...")
        start_time = time.time()
        
        num_pairs, num_bins = time_series_matrix.shape
        
        # Apply Hanning window to ALL series at once (batched)
        hanning_window = cp.hanning(num_bins).astype(self.dtype)
        windowed_matrix = time_series_matrix * hanning_window[cp.newaxis, :]
        
        # Detrend ALL series at once (batched)
        detrended_matrix = cusignal.detrend(windowed_matrix, type='linear', axis=1)
        
        # Perform BATCHED FFT across all IP pairs at once
        fft_results = cp.fft.rfft(detrended_matrix, axis=1)
        fft_magnitudes = cp.abs(fft_results)
        
        # Calculate periodicity scores using vectorized operations
        fft_magnitudes_no_dc = fft_magnitudes[:, 1:]
        
        # Sort magnitudes for each series to find median noise floor
        sorted_magnitudes = cp.sort(fft_magnitudes_no_dc, axis=1)
        
        # Median noise floor (exclude top 10 peaks)
        median_noise = cp.median(sorted_magnitudes[:, :-10], axis=1)
        median_noise = cp.where(median_noise == 0, 1e-10, median_noise)
        
        # Find dominant frequency (peak) for each series
        peak_indices = cp.argmax(fft_magnitudes_no_dc, axis=1) + 1
        peak_magnitudes = fft_magnitudes[cp.arange(num_pairs), peak_indices]
        
        # Periodicity score = peak / noise floor
        periodicity_scores = peak_magnitudes / median_noise
        
        # Convert frequency to period
        frequencies = cp.fft.rfftfreq(num_bins, 1.0 / sample_rate)
        dominant_freqs = frequencies[peak_indices]
        estimated_periods = cp.where(dominant_freqs > 0, 1.0 / dominant_freqs, 0.0)
        
        elapsed = time.time() - start_time
        rprint(f"[green]✓[/green] Computed {num_pairs:,} periodicity scores in {elapsed:.2f}s")
        rprint(f"[*] GPU Throughput: {num_pairs / elapsed:.0f} IP pairs/second")
        
        return periodicity_scores, estimated_periods, peak_magnitudes
    
    def detect_beacons(
        self,
        pcap_path: str,
        min_score: float = 5.0,
        bin_size: float = 1.0,
        loop_count: int = 1,
        top_n: int = 10,
        max_ip_pairs: int = None
    ) -> List[Dict]:
        """
        Detect beaconing behavior in PCAP file using batched GPU processing.
        
        Args:
            pcap_path: Path to PCAP file
            min_score: Minimum periodicity score to flag
            bin_size: Size of time bins in seconds
            loop_count: Number of times to duplicate data for stress testing
            top_n: Number of top candidates to return
            max_ip_pairs: Maximum number of IP pairs to process
            
        Returns:
            List of detected beacons with scores and metadata
        """
        rprint("\n[bold cyan]GPU-Accelerated Beacon Detection[/bold cyan]")
        rprint("=" * 60)
        
        # Load PCAP to GPU (cuDF)
        df = self.load_pcap_to_cudf(pcap_path, loop_count)
        
        # Create batched time-series matrix
        time_series_matrix, ip_pair_list, num_bins = self.create_batched_time_series(
            df,
            bin_size,
            max_ip_pairs
        )
        
        rprint(f"[*] Time-series matrix shape: {time_series_matrix.shape}")
        rprint(f"[*] GPU memory usage: {time_series_matrix.nbytes / (1024**3):.2f} GB")
        
        # Compute periodicity scores for ALL pairs at once
        scores, periods, peak_mags = self.compute_batched_periodicity_scores(
            time_series_matrix,
            1.0 / bin_size
        )
        
        # Move results to CPU for filtering and sorting
        scores_cpu = scores.get()
        periods_cpu = periods.get()
        peak_mags_cpu = peak_mags.get()
        
        # Build results list
        results = []
        for idx, (src_ip, dst_ip) in enumerate(ip_pair_list):
            score = float(scores_cpu[idx])
            
            if score >= min_score:
                packet_count = int(time_series_matrix[idx].sum().get())
                
                results.append({
                    'src_ip': src_ip,
                    'dst_ip': dst_ip,
                    'score': score,
                    'period': float(periods_cpu[idx]),
                    'packet_count': packet_count,
                    'peak_magnitude': float(peak_mags_cpu[idx])
                })
        
        rprint(f"[green]✓[/green] Found {len(results)} potential beacons (score >= {min_score})")
        
        # Sort by score and return top N
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_n]
    
    def print_results_table(self, results: List[Dict]):
        """
        Print results in a formatted table using rich.
        
        Args:
            results: List of detection results
        """
        table = Table(title="Beacon Detection Results", show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="cyan", justify="right")
        table.add_column("Score", style="green", justify="right")
        table.add_column("Period (s)", style="yellow", justify="right")
        table.add_column("Source IP", style="blue")
        table.add_column("Destination IP", style="blue")
        table.add_column("Packets", style="cyan", justify="right")
        table.add_column("Peak Mag", style="magenta", justify="right")
        
        for idx, result in enumerate(results, 1):
            table.add_row(
                str(idx),
                f"{result['score']:.2f}",
                f"{result['period']:.2f}",
                result['src_ip'],
                result['dst_ip'],
                str(result['packet_count']),
                f"{result['peak_magnitude']:.2e}"
            )
        
        self.console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="GPU-Accelerated Frequency Domain Analysis for C2 Detection"
    )
    parser.add_argument(
        "--pcap",
        default="botnet-capture-20110810-neris_modified.pcap",
        help="Path to PCAP file"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=5.0,
        help="Minimum periodicity score to flag (default: 5.0)"
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=1.0,
        help="Time bin size in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--loop-count",
        type=int,
        default=1,
        help="Number of times to duplicate data for stress testing (default: 1)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top candidates to display (default: 10)"
    )
    parser.add_argument(
        "--dtype",
        choices=['float32', 'float64'],
        default='float32',
        help="Data type for FFT (default: float32 for performance)"
    )
    parser.add_argument(
        "--max-ip-pairs",
        type=int,
        default=None,
        help="Maximum number of IP pairs to process (default: unlimited)"
    )
    
    args = parser.parse_args()
    
    # Convert dtype string to cupy dtype
    dtype = cp.float32 if args.dtype == 'float32' else cp.float64
    
    # Initialize detector
    detector = GPUBeaconDetector(dtype=dtype)
    
    # Detect beacons
    results = detector.detect_beacons(
        args.pcap,
        min_score=args.min_score,
        bin_size=args.bin_size,
        loop_count=args.loop_count,
        top_n=args.top_n,
        max_ip_pairs=args.max_ip_pairs
    )
    
    # Print results
    if results:
        detector.print_results_table(results)
        
        # Save results for visualization
        import json
        output_file = "detection_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        rprint(f"\n[green]✓[/green] Results saved to {output_file}")
    else:
        rprint("\n[yellow]No beacons detected above threshold[/yellow]")


if __name__ == "__main__":
    main()
