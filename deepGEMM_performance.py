#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import argparse
from pathlib import Path

def run_benchmark_and_collect_data():
    """Run the benchmark tests and collect data for visualization"""
    import torch
    import deep_gemm
    from deep_gemm import bench_kineto, calc_diff
    import random
    import tempfile
    import subprocess
    from deep_gemm.jit_kernels.gemm import get_best_configs
    from deep_gemm.utils import get_num_sms
    
    # Initialize results dictionary
    results = {
        'standard_gemm': [],
        'grouped_contiguous_gemm': [],
        'grouped_masked_gemm': [],
        'tuning_configs': []
    }
    
    # Helper functions for tensor construction (from test_core.py)
    def per_token_cast_to_fp8(x):
        assert x.dim() == 2 and x.size(1) % 128 == 0
        m, n = x.shape
        x_view = x.view(m, -1, 128)
        x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
        return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)

    def per_block_cast_to_fp8(x):
        assert x.dim() == 2
        m, n = x.shape
        x_padded = torch.zeros((((m + 127) // 128) * 128, ((n + 127) // 128) * 128), dtype=x.dtype, device=x.device)
        x_padded[:m, :n] = x
        x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
        x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
        x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
        return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))
    
    def get_col_major_tma_aligned_tensor(tensor):
        return deep_gemm.get_col_major_tma_aligned_tensor(tensor)
    
    def construct(m, k, n):
        x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
        y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
        out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        ref_out = x @ y.t()

        x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_block_cast_to_fp8(y)
        # Transpose earlier so that the testing will not trigger transposing kernels
        x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
        return x_fp8, y_fp8, out, ref_out
    
    def construct_grouped(num_groups, m, k, n, is_masked):
        x = torch.randn((num_groups, m, k), device='cuda', dtype=torch.bfloat16)
        y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
        out = torch.empty((num_groups, m, n), device='cuda', dtype=torch.bfloat16)
        ref_out = torch.einsum('gmk,gnk->gmn', x, y)

        assert m % 4 == 0, f'TMA alignment error: {m}'
        x_fp8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty((num_groups, m, k // 128), device='cuda', dtype=torch.float))
        y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, (n + 127) // 128, k // 128), device='cuda', dtype=torch.float))
        for i in range(num_groups):
            x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8(x[i])
            y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])

        # For non-masked input, we must merge the group and M dims
        if not is_masked:
            x_fp8 = (x_fp8[0].view(-1, k), per_token_cast_to_fp8(x.view(-1, k))[1])
            out, ref_out = out.view(-1, n), ref_out.view(-1, n)

        # Transpose earlier so that the testing will not trigger transposing kernels
        x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
        return x_fp8, y_fp8, out, ref_out

    print("Running benchmark tests to collect data...")
    # 1. Standard GEMM benchmarks
    for m in (64, 128, 4096):
        for k, n in [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)]:
            x_fp8, y_fp8, out, ref_out = construct(m, k, n)
            
            # noinspection PyShadowingNames
            def test_func():
                # Construct new tensors every time to avoid L2 cache acceleration
                x_fp8, y_fp8, out, ref_out = construct(m, k, n)
                deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)

            t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
            diff = calc_diff(out, ref_out)
            tflops = 2 * m * n * k / t / 1e12
            memory_bw = (m * k + k * n + m * n * 2) / 1e9 / t
            
            # Get tuning configs for this shape
            num_sms = get_num_sms()
            block_m, block_n, num_stages, num_tma_multicast, smem_size = get_best_configs(m, n, k, 1, num_sms)
            
            results['standard_gemm'].append({
                'shape': f"m={m}, n={n}, k={k}",
                'm': m,
                'n': n,
                'k': k,
                'execution_time_us': t * 1e6,
                'throughput_tflops': tflops,
                'memory_bandwidth_gbps': memory_bw,
                'numerical_diff': diff,
                'matmul_size_gb': (m * n * k * 2) / 8 / 1e9,
                'block_m': block_m,
                'block_n': block_n,
                'num_stages': num_stages,
                'num_tma_multicast': num_tma_multicast,
                'smem_size': smem_size
            })
            
            results['tuning_configs'].append({
                'shape': f"m={m}, n={n}, k={k}",
                'm': m,
                'n': n,
                'k': k,
                'block_m': block_m,
                'block_n': block_n,
                'num_stages': num_stages,
                'num_tma_multicast': num_tma_multicast,
                'smem_size': smem_size,
                'throughput_tflops': tflops
            })
    
    # 2. Grouped Contiguous GEMM benchmarks
    for num_groups, m, k, n in ((4, 8192, 7168, 4096), (4, 8192, 2048, 7168), (8, 4096, 7168, 4096), (8, 4096, 2048, 7168)):
        x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=False)
        
        # noinspection PyShadowingNames
        def test_func():
            # Construct new tensors every time to avoid L2 cache acceleration
            x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=False)
            m_indices = torch.arange(0, num_groups, device='cuda', dtype=torch.int)
            m_indices = m_indices.unsqueeze(-1).expand(num_groups, m).contiguous().view(-1)
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        diff = calc_diff(out, ref_out)
        tflops = 2 * num_groups * m * n * k / t / 1e12
        memory_bw = (num_groups * (m * k + k * n + m * n * 2)) / 1e9 / t
        
        results['grouped_contiguous_gemm'].append({
            'shape': f"ng={num_groups}, m={m}, n={n}, k={k}",
            'num_groups': num_groups,
            'm': m,
            'n': n,
            'k': k,
            'execution_time_us': t * 1e6,
            'throughput_tflops': tflops,
            'memory_bandwidth_gbps': memory_bw,
            'numerical_diff': diff,
            'matmul_size_gb': (num_groups * m * n * k * 2) / 8 / 1e9
        })
    
    # 3. Grouped Masked GEMM benchmarks
    for num_groups, m in ((1, 1024), (2, 512), (4, 256)):
        for k, n in ((7168, 4096), (2048, 7168)):
            x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=True)
            
            # noinspection PyShadowingNames
            def test_func():
                # Construct new tensors every time to avoid L2 cache acceleration
                x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=True)
                masked_m = torch.ones((num_groups, ), device='cuda', dtype=torch.int) * m
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x_fp8, y_fp8, out, masked_m, m)

            t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
            diff = calc_diff(out, ref_out)
            tflops = 2 * num_groups * m * n * k / t / 1e12
            memory_bw = (num_groups * (m * k + k * n + m * n * 2)) / 1e9 / t
            
            results['grouped_masked_gemm'].append({
                'shape': f"ng={num_groups}, m={m}, n={n}, k={k}",
                'num_groups': num_groups,
                'm': m,
                'n': n,
                'k': k,
                'execution_time_us': t * 1e6,
                'throughput_tflops': tflops,
                'memory_bandwidth_gbps': memory_bw,
                'numerical_diff': diff,
                'matmul_size_gb': (num_groups * m * n * k * 2) / 8 / 1e9
            })
    
    return results


def visualize_performance(data, output_dir='figures/performance'):
    """Create performance visualizations from benchmark data"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('ggplot')
    sns.set_context("talk")
    
    # 1. Performance comparison across different GEMMs
    def plot_tflops_comparison():
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Standard GEMM data
        std_shapes = [d['shape'] for d in data['standard_gemm']]
        std_tflops = [d['throughput_tflops'] for d in data['standard_gemm']]
        
        # Grouped Contiguous GEMM data  
        gc_shapes = [d['shape'] for d in data['grouped_contiguous_gemm']]
        gc_tflops = [d['throughput_tflops'] for d in data['grouped_contiguous_gemm']]
        
        # Grouped Masked GEMM data
        gm_shapes = [d['shape'] for d in data['grouped_masked_gemm']]
        gm_tflops = [d['throughput_tflops'] for d in data['grouped_masked_gemm']]
        
        # Create a wide dataframe for plotting
        import pandas as pd
        df = pd.DataFrame({
            'Standard GEMM': pd.Series(dict(zip(std_shapes, std_tflops))),
            'Grouped Contiguous GEMM': pd.Series(dict(zip(gc_shapes, gc_tflops))),
            'Grouped Masked GEMM': pd.Series(dict(zip(gm_shapes, gm_tflops)))
        })
        
        # Plot
        df.plot(kind='bar', ax=ax)
        ax.set_ylabel('Throughput (TFLOPS)')
        ax.set_xlabel('Matrix Dimensions')
        ax.set_title('DeepGEMM Performance Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/gemm_performance_comparison.png', dpi=300)
        plt.close()
    
    # 2. Scaling behavior visualization
    def plot_scaling_behavior():
        # Extract data for standard GEMM with different M dimensions
        m_sizes = sorted(list(set([d['m'] for d in data['standard_gemm']])))
        
        # Create figure with 2 y-axes (TFLOPS and memory bandwidth)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        
        # Plot throughput lines for each M size
        for m in m_sizes:
            # Filter data for this M size
            filtered_data = [d for d in data['standard_gemm'] if d['m'] == m]
            # Sort by matrix product size (m*n*k)
            filtered_data.sort(key=lambda x: x['m'] * x['n'] * x['k'])
            
            # Extract data points
            shapes = [f"n={d['n']}, k={d['k']}" for d in filtered_data]
            tflops = [d['throughput_tflops'] for d in filtered_data]
            memory_bw = [d['memory_bandwidth_gbps'] for d in filtered_data]
            
            # Plot lines
            line1 = ax1.plot(shapes, tflops, 'o-', label=f'm={m} (TFLOPS)')
            ax2.plot(shapes, memory_bw, 's--', color=line1[0].get_color(), alpha=0.6, label=f'm={m} (GB/s)')
        
        # Set labels and legend
        ax1.set_xlabel('Matrix Dimensions (n, k)')
        ax1.set_ylabel('Throughput (TFLOPS)')
        ax2.set_ylabel('Memory Bandwidth (GB/s)')
        ax1.set_title('DeepGEMM Scaling Behavior')
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/scaling_behavior.png', dpi=300)
        plt.close()
    
    # 3. Heatmap of performance across different dimensions
    def plot_performance_heatmap():
        # Create dictionaries to hold the performance data
        # We'll create a 2D grid of n vs k dimensions, with color showing TFLOPS
        std_data = data['standard_gemm']
        
        # Get unique n and k values
        n_values = sorted(list(set([d['n'] for d in std_data])))
        k_values = sorted(list(set([d['k'] for d in std_data])))
        
        # Create matrices for each m value
        for m in sorted(list(set([d['m'] for d in std_data]))):
            # Filter data for this m
            m_data = [d for d in std_data if d['m'] == m]
            
            # Create empty matrix
            performance_matrix = np.zeros((len(k_values), len(n_values)))
            
            # Fill the matrix
            for d in m_data:
                i = k_values.index(d['k'])
                j = n_values.index(d['n'])
                performance_matrix[i, j] = d['throughput_tflops']
            
            # Plot heatmap
            plt.figure(figsize=(12, 8))
            ax = sns.heatmap(performance_matrix, annot=True, fmt=".1f", 
                          xticklabels=[f"n={n}" for n in n_values],
                          yticklabels=[f"k={k}" for k in k_values],
                          cmap="YlGnBu")
            
            plt.title(f'DeepGEMM Performance Heatmap (m={m})')
            plt.xlabel('n dimension')
            plt.ylabel('k dimension')
            cbar = ax.collections[0].colorbar
            cbar.set_label('Throughput (TFLOPS)')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/performance_heatmap_m{m}.png', dpi=300)
            plt.close()
    
    # 4. Configuration impact visualization
    def plot_configuration_impact():
        # Extract configuration data
        block_m_values = sorted(list(set([d['block_m'] for d in data['tuning_configs']])))
        block_n_values = sorted(list(set([d['block_n'] for d in data['tuning_configs']])))
        
        # Group by block sizes
        block_groups = {}
        for d in data['tuning_configs']:
            key = f"block_m={d['block_m']}, block_n={d['block_n']}"
            if key not in block_groups:
                block_groups[key] = []
            block_groups[key].append(d)
        
        # Calculate average performance for each configuration
        avg_performance = {}
        for key, group in block_groups.items():
            avg_performance[key] = sum(d['throughput_tflops'] for d in group) / len(group)
        
        # Plot bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(avg_performance.keys(), avg_performance.values())
        plt.ylabel('Average Throughput (TFLOPS)')
        plt.xlabel('Block Configuration')
        plt.title('Impact of Block Sizes on Performance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/block_size_impact.png', dpi=300)
        plt.close()
        
        # Plot relationship between num_stages and performance
        stage_groups = {}
        for d in data['tuning_configs']:
            key = d['num_stages']
            if key not in stage_groups:
                stage_groups[key] = []
            stage_groups[key].append(d)
        
        avg_stage_performance = {}
        for key, group in stage_groups.items():
            avg_stage_performance[key] = sum(d['throughput_tflops'] for d in group) / len(group)
        
        plt.figure(figsize=(10, 6))
        stages = sorted(avg_stage_performance.keys())
        plt.bar([str(s) for s in stages], [avg_stage_performance[s] for s in stages])
        plt.ylabel('Average Throughput (TFLOPS)')
        plt.xlabel('Number of Pipeline Stages')
        plt.title('Impact of Pipeline Stages on Performance')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pipeline_stages_impact.png', dpi=300)
        plt.close()
    
    # 5. Memory bandwidth utilization
    def plot_memory_bandwidth():
        # Calculate theoretical peak bandwidth for Hopper
        # H100 has ~3TB/s memory bandwidth
        theoretical_peak_gbps = 3000
        
        # Prepare data
        std_data = data['standard_gemm']
        
        # Sort by matmul size
        std_data.sort(key=lambda x: x['matmul_size_gb'])
        
        # Extract data
        shapes = [d['shape'] for d in std_data]
        memory_bw = [d['memory_bandwidth_gbps'] for d in std_data]
        utilization = [bw / theoretical_peak_gbps * 100 for bw in memory_bw]
        
        # Plot
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()
        
        bar_width = 0.35
        x = np.arange(len(shapes))
        
        ax1.bar(x - bar_width/2, memory_bw, bar_width, label='Memory Bandwidth')
        ax2.bar(x + bar_width/2, utilization, bar_width, color='orange', label='Utilization %')
        
        ax1.set_xlabel('Matrix Dimensions')
        ax1.set_ylabel('Memory Bandwidth (GB/s)')
        ax2.set_ylabel('HBM Utilization (%)')
        ax1.set_title('Memory Bandwidth Utilization')
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(shapes, rotation=45, ha='right')
        
        # Add a horizontal line for theoretical peak
        ax1.axhline(y=theoretical_peak_gbps, color='r', linestyle='--', label='Theoretical Peak')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/memory_bandwidth.png', dpi=300)
        plt.close()
    
    # 6. Numerical accuracy vs. performance
    def plot_accuracy_vs_performance():
        plt.figure(figsize=(10, 6))
        
        # Prepare data - all gemm types
        all_data = data['standard_gemm'] + data['grouped_contiguous_gemm'] + data['grouped_masked_gemm']
        
        # Extract data
        tflops = [d['throughput_tflops'] for d in all_data]
        diff = [d['numerical_diff'] for d in all_data]
        labels = ['Standard' if 'num_groups' not in d else 
                  'Grouped Contiguous' if 'num_groups' in d and d in data['grouped_contiguous_gemm'] else 
                  'Grouped Masked' for d in all_data]
        
        # Create scatter plot
        for label in set(labels):
            indices = [i for i, l in enumerate(labels) if l == label]
            plt.scatter([tflops[i] for i in indices], [diff[i] for i in indices], label=label, alpha=0.7)
        
        plt.xlabel('Throughput (TFLOPS)')
        plt.ylabel('Numerical Difference')
        plt.title('Accuracy vs. Performance')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/accuracy_vs_performance.png', dpi=300)
        plt.close()
    
    # Execute all visualization functions
    print("Generating performance visualizations...")
    plot_tflops_comparison()
    plot_scaling_behavior()
    plot_performance_heatmap()
    plot_configuration_impact()
    plot_memory_bandwidth()
    plot_accuracy_vs_performance()
    print(f"Visualizations saved to {output_dir}/")


def save_benchmark_data(data, filename='benchmark_data.json'):
    """Save benchmark data to JSON file"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Benchmark data saved to {filename}")


def load_benchmark_data(filename='benchmark_data.json'):
    """Load benchmark data from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepGEMM Performance Visualization')
    parser.add_argument('--run-benchmark', action='store_true', help='Run benchmarks to collect fresh data')
    parser.add_argument('--input-file', type=str, default='benchmark_data.json', help='Input data file (if not running benchmarks)')
    parser.add_argument('--output-dir', type=str, default='figures/performance', help='Output directory for visualizations')
    args = parser.parse_args()
    
    if args.run_benchmark:
        # Run benchmarks and collect data
        benchmark_data = run_benchmark_and_collect_data()
        # Save data to file
        save_benchmark_data(benchmark_data)
    else:
        # Load existing data
        try:
            benchmark_data = load_benchmark_data(args.input_file)
        except FileNotFoundError:
            print(f"Error: Input file {args.input_file} not found. Run with --run-benchmark to generate data.")
            exit(1)
    
    # Generate visualizations
    visualize_performance(benchmark_data, args.output_dir)
