import streamlit as st
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="DeepGEMM Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_benchmark_data(filename='benchmark_data.json'):
    """Load benchmark data from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: Input file {filename} not found. Run visualize_performance.py with --run-benchmark first.")
        st.stop()


def create_dataframes(data):
    """Convert the JSON data to pandas dataframes for easier plotting"""
    df_standard = pd.DataFrame(data['standard_gemm'])
    df_grouped_contiguous = pd.DataFrame(data['grouped_contiguous_gemm'])
    df_grouped_masked = pd.DataFrame(data['grouped_masked_gemm'])
    df_tuning = pd.DataFrame(data['tuning_configs'])
    
    # Create a combined dataframe with a 'type' column
    df_standard['type'] = 'Standard GEMM'
    df_grouped_contiguous['type'] = 'Grouped Contiguous GEMM'
    df_grouped_masked['type'] = 'Grouped Masked GEMM'
    
    df_combined = pd.concat([
        df_standard,
        df_grouped_contiguous,
        df_grouped_masked
    ], ignore_index=True)
    
    return {
        'standard': df_standard,
        'grouped_contiguous': df_grouped_contiguous,
        'grouped_masked': df_grouped_masked,
        'tuning': df_tuning,
        'combined': df_combined
    }


def plot_tflops_comparison(dfs):
    """Interactive bar chart comparing TFLOPS across GEMM types"""
    df = dfs['combined']
    
    fig = px.bar(
        df,
        x='shape',
        y='throughput_tflops',
        color='type',
        title='DeepGEMM Performance Comparison',
        labels={'throughput_tflops': 'Throughput (TFLOPS)', 
                'shape': 'Matrix Dimensions',
                'type': 'GEMM Type'},
        barmode='group',
        hover_data=['m', 'n', 'k', 'execution_time_us', 'memory_bandwidth_gbps']
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def plot_scaling_behavior(dfs):
    """Interactive line chart showing scaling behavior"""
    df = dfs['standard']
    
    # Get unique m values
    m_values = sorted(df['m'].unique())
    
    # Create figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    for m in m_values:
        df_m = df[df['m'] == m].sort_values(by=['m', 'n', 'k'])
        
        # Add trace for throughput
        fig.add_trace(
            go.Scatter(
                x=df_m['shape'],
                y=df_m['throughput_tflops'],
                mode='lines+markers',
                name=f'm={m} (TFLOPS)',
                hovertemplate='%{y:.2f} TFLOPS<br>%{x}',
            ),
            secondary_y=False,
        )
        
        # Add trace for memory bandwidth
        fig.add_trace(
            go.Scatter(
                x=df_m['shape'],
                y=df_m['memory_bandwidth_gbps'],
                mode='lines+markers',
                line=dict(dash='dash'),
                name=f'm={m} (GB/s)',
                hovertemplate='%{y:.2f} GB/s<br>%{x}',
                opacity=0.7,
            ),
            secondary_y=True,
        )
    
    # Set titles
    fig.update_layout(
        title_text="DeepGEMM Scaling Behavior",
        xaxis_title="Matrix Dimensions (n, k)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Throughput (TFLOPS)", secondary_y=False)
    fig.update_yaxes(title_text="Memory Bandwidth (GB/s)", secondary_y=True)
    
    # Update x-axis
    fig.update_xaxes(tickangle=-45)
    
    return fig


def plot_performance_heatmap(dfs):
    """Interactive heatmap of performance across matrix dimensions"""
    df = dfs['standard']
    
    # Get unique m values
    m_values = sorted(df['m'].unique())
    
    # User selects m value
    m_value = st.selectbox('Select m dimension:', m_values)
    
    # Filter data for selected m
    df_m = df[df['m'] == m_value]
    
    # Get unique n and k values
    n_values = sorted(df_m['n'].unique())
    k_values = sorted(df_m['k'].unique())
    
    # Create empty matrix
    performance_matrix = np.zeros((len(k_values), len(n_values)))
    
    # Fill the matrix
    for idx, row in df_m.iterrows():
        i = k_values.index(row['k'])
        j = n_values.index(row['n'])
        performance_matrix[i, j] = row['throughput_tflops']
    
    # Create heatmap with Plotly
    fig = px.imshow(
        performance_matrix,
        labels=dict(x="n dimension", y="k dimension", color="TFLOPS"),
        x=[f"n={n}" for n in n_values],
        y=[f"k={k}" for k in k_values],
        title=f"DeepGEMM Performance Heatmap (m={m_value})",
        color_continuous_scale="viridis",
        text_auto=".1f"
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
    )
    
    return fig


def plot_configuration_impact(dfs):
    """Visualize impact of different configurations on performance"""
    df = dfs['tuning']
    
    # Create figure with two subplots
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=("Impact of Block Sizes on Performance", 
                                        "Impact of Pipeline Stages on Performance"))
    
    # 1. Plot block size impact
    block_configs = df.groupby(['block_m', 'block_n'])['throughput_tflops'].mean().reset_index()
    block_configs['config'] = block_configs.apply(lambda x: f"block_m={x['block_m']}, block_n={x['block_n']}", axis=1)
    
    fig.add_trace(
        go.Bar(
            x=block_configs['config'],
            y=block_configs['throughput_tflops'],
            name="Block Configurations",
            hovertemplate='%{y:.2f} TFLOPS<br>%{x}',
        ),
        row=1, col=1
    )
    
    # 2. Plot pipeline stages impact
    stages_impact = df.groupby(['num_stages'])['throughput_tflops'].mean().reset_index()
    
    fig.add_trace(
        go.Bar(
            x=stages_impact['num_stages'].astype(str),
            y=stages_impact['throughput_tflops'],
            name="Pipeline Stages",
            hovertemplate='%{y:.2f} TFLOPS<br>%{x} stages',
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        title_text="Configuration Impact Analysis",
        showlegend=False
    )
    
    # Update axes
    fig.update_xaxes(title_text="Block Configuration", tickangle=-45, row=1, col=1)
    fig.update_yaxes(title_text="Average Throughput (TFLOPS)", row=1, col=1)
    
    fig.update_xaxes(title_text="Number of Pipeline Stages", row=1, col=2)
    fig.update_yaxes(title_text="Average Throughput (TFLOPS)", row=1, col=2)
    
    return fig


def plot_memory_bandwidth(dfs):
    """Visualize memory bandwidth utilization"""
    df = dfs['standard'].sort_values(by='matmul_size_gb')
    
    # Theoretical peak bandwidth for Hopper (H100 ~3TB/s)
    theoretical_peak_gbps = 3000
    
    # Calculate utilization percentage
    df['utilization_pct'] = df['memory_bandwidth_gbps'] / theoretical_peak_gbps * 100
    
    # Create figure with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    fig.add_trace(
        go.Bar(
            x=df['shape'],
            y=df['memory_bandwidth_gbps'],
            name="Memory Bandwidth (GB/s)",
            hovertemplate='%{y:.2f} GB/s<br>%{x}',
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['shape'],
            y=df['utilization_pct'],
            mode='lines+markers',
            name="HBM Utilization (%)",
            hovertemplate='%{y:.2f}%<br>%{x}',
            line=dict(color='orange')
        ),
        secondary_y=True,
    )
    
    # Add horizontal line for theoretical peak
    fig.add_trace(
        go.Scatter(
            x=[df['shape'].iloc[0], df['shape'].iloc[-1]],
            y=[theoretical_peak_gbps, theoretical_peak_gbps],
            mode='lines',
            name="Theoretical Peak",
            line=dict(color='red', dash='dash'),
        ),
        secondary_y=False,
    )
    
    # Set titles
    fig.update_layout(
        title_text="Memory Bandwidth Utilization",
        xaxis_title="Matrix Dimensions",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Memory Bandwidth (GB/s)", secondary_y=False)
    fig.update_yaxes(title_text="HBM Utilization (%)", secondary_y=True)
    
    # Update x-axis
    fig.update_xaxes(tickangle=-45)
    
    return fig


def plot_accuracy_vs_performance(dfs):
    """Scatter plot of numerical accuracy vs performance"""
    df = dfs['combined']
    
    fig = px.scatter(
        df,
        x='throughput_tflops',
        y='numerical_diff',
        color='type',
        title='Accuracy vs. Performance',
        labels={'throughput_tflops': 'Throughput (TFLOPS)', 
                'numerical_diff': 'Numerical Difference',
                'type': 'GEMM Type'},
        hover_data=['shape', 'm', 'n', 'k']
    )
    
    # Log scale might help visualize small differences better
    use_log = st.checkbox('Use logarithmic scale for numerical difference', value=False)
    if use_log:
        fig.update_layout(yaxis_type="log")
    
    return fig


def plot_gemm_comparison_table(dfs):
    """Create a detailed comparison table of GEMM implementations"""
    df = dfs['combined']
    
    # Compute average metrics by GEMM type
    avg_metrics = df.groupby('type').agg({
        'throughput_tflops': 'mean',
        'memory_bandwidth_gbps': 'mean',
        'numerical_diff': 'mean',
        'execution_time_us': 'mean'
    }).reset_index()
    
    # Compute max metrics by GEMM type
    max_metrics = df.groupby('type').agg({
        'throughput_tflops': 'max',
        'memory_bandwidth_gbps': 'max'
    }).reset_index()
    
    max_metrics.columns = ['type', 'max_tflops', 'max_bandwidth']
    
    # Merge the dataframes
    metrics = pd.merge(avg_metrics, max_metrics, on='type')
    
    # Format the metrics
    metrics['avg_throughput_tflops'] = metrics['throughput_tflops'].round(2)
    metrics['avg_bandwidth_gbps'] = metrics['memory_bandwidth_gbps'].round(2)
    metrics['avg_numerical_diff'] = metrics['numerical_diff'].apply(lambda x: f"{x:.6f}")
    metrics['avg_execution_time_us'] = metrics['execution_time_us'].round(2)
    metrics['max_tflops'] = metrics['max_tflops'].round(2)
    metrics['max_bandwidth'] = metrics['max_bandwidth'].round(2)
    
    # Select columns for display
    display_metrics = metrics[[
        'type', 
        'avg_throughput_tflops', 
        'max_tflops',
        'avg_bandwidth_gbps', 
        'max_bandwidth',
        'avg_numerical_diff', 
        'avg_execution_time_us'
    ]]
    
    # Rename columns for better display
    display_metrics.columns = [
        'GEMM Type', 
        'Avg TFLOPS', 
        'Max TFLOPS',
        'Avg Bandwidth (GB/s)', 
        'Max Bandwidth (GB/s)',
        'Avg Numerical Diff', 
        'Avg Execution Time (μs)'
    ]
    
    return display_metrics


def plot_matrix_size_impact(dfs):
    """Analyze the impact of matrix size on performance"""
    df_standard = dfs['standard']
    
    # Calculate matrix size in MB
    df_standard['matrix_size_mb'] = df_standard['matmul_size_gb'] * 1024
    
    fig = px.scatter(
        df_standard,
        x='matrix_size_mb',
        y='throughput_tflops',
        color='m',
        size='n',
        hover_data=['shape', 'k', 'block_m', 'block_n', 'num_stages'],
        labels={
            'matrix_size_mb': 'Matrix Size (MB)',
            'throughput_tflops': 'Throughput (TFLOPS)',
            'm': 'M Dimension',
            'n': 'N Dimension'
        },
        title='Performance vs Matrix Size'
    )
    
    # Add trend line
    fig.update_layout(
        xaxis_title='Matrix Size (MB)',
        yaxis_title='Throughput (TFLOPS)'
    )
    
    return fig


def main():
    st.title("DeepGEMM Performance Dashboard")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        """
        This dashboard visualizes performance metrics for DeepGEMM, 
        a high-performance library for FP8 General Matrix Multiplications (GEMMs)
        with fine-grained scaling specifically designed for NVIDIA Hopper tensor cores.
        """
    )
    
    st.sidebar.header("Settings")
    data_file = st.sidebar.text_input(
        "Benchmark data file path",
        value="benchmark_data.json"
    )
    
    if not os.path.exists(data_file):
        st.sidebar.error(f"File not found: {data_file}")
        st.sidebar.info(
            """
            Please run the data collection script first:
            ```
            python visualize_performance.py --run-benchmark
            ```
            """
        )
        st.stop()
    
    # Load data
    data = load_benchmark_data(data_file)
    dataframes = create_dataframes(data)
    
    # Dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Performance Overview", 
        "GEMM Comparison", 
        "Configuration Analysis",
        "Hardware Utilization"
    ])
    
    with tab1:
        st.header("Performance Overview")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        # Calculate overall metrics
        max_tflops = dataframes['combined']['throughput_tflops'].max()
        avg_tflops = dataframes['combined']['throughput_tflops'].mean()
        max_bandwidth = dataframes['combined']['memory_bandwidth_gbps'].max()
        
        with col1:
            st.metric("Max Throughput", f"{max_tflops:.2f} TFLOPS")
        
        with col2:
            st.metric("Avg Throughput", f"{avg_tflops:.2f} TFLOPS")
            
        with col3:
            st.metric("Max Memory Bandwidth", f"{max_bandwidth:.2f} GB/s")
        
        # Summary table
        st.subheader("GEMM Implementation Comparison")
        comparison_table = plot_gemm_comparison_table(dataframes)
        st.dataframe(comparison_table)
        
        # Performance vs Matrix Size
        st.subheader("Performance vs Matrix Size")
        matrix_size_fig = plot_matrix_size_impact(dataframes)
        st.plotly_chart(matrix_size_fig, use_container_width=True)
        
        # Accuracy vs Performance
        st.subheader("Accuracy vs Performance")
        accuracy_fig = plot_accuracy_vs_performance(dataframes)
        st.plotly_chart(accuracy_fig, use_container_width=True)
    
    with tab2:
        st.header("GEMM Comparison")
        
        # TFLOPS comparison
        st.subheader("Throughput Comparison")
        tflops_fig = plot_tflops_comparison(dataframes)
        st.plotly_chart(tflops_fig, use_container_width=True)
        
        # Scaling behavior
        st.subheader("Scaling Behavior")
        scaling_fig = plot_scaling_behavior(dataframes)
        st.plotly_chart(scaling_fig, use_container_width=True)
        
        # Performance heatmap
        st.subheader("Performance Heatmap")
        heatmap_fig = plot_performance_heatmap(dataframes)
        st.plotly_chart(heatmap_fig, use_container_width=True)
    
    with tab3:
        st.header("Configuration Analysis")
        
        # Configuration impact
        st.subheader("Impact of Kernel Configurations")
        config_fig = plot_configuration_impact(dataframes)
        st.plotly_chart(config_fig, use_container_width=True)
        
        # Show raw tuning data
        st.subheader("Tuning Configuration Details")
        st.dataframe(dataframes['tuning'])
    
    with tab4:
        st.header("Hardware Utilization")
        
        # Memory bandwidth
        st.subheader("Memory Bandwidth Utilization")
        bandwidth_fig = plot_memory_bandwidth(dataframes)
        st.plotly_chart(bandwidth_fig, use_container_width=True)
        
        # Advanced hardware metrics explanations
        st.subheader("Hardware Utilization Insights")
        
        # Theoretical metrics
        st.markdown("### Theoretical Peak Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            # H100 specs
            st.info("""
            **NVIDIA H100 Tensor Core Theoretical Peaks:**
            - FP8 Tensor Core: ~1000 TFLOPS
            - Memory Bandwidth: ~3.0 TB/s
            """)
        
        with col2:
            # Explain utilization
            st.success(f"""
            **Current Utilization:**
            - FP8 Compute: {max_tflops/10:.1f}% of theoretical peak
            - Memory Bandwidth: {max_bandwidth/3000*100:.1f}% of theoretical peak
            """)
        
        # Roofline model explanation
        st.markdown("### Understanding the Roofline Model")
        st.markdown("""
        The performance of DeepGEMM kernels can be understood through the roofline model:
        
        1. **Compute-bound** operations benefit from optimized tensor core utilization
        2. **Memory-bound** operations benefit from optimized memory access patterns
        
        DeepGEMM achieves high performance through:
        - Optimized blocking for tensor core utilization
        - TMA (Tensor Memory Accelerator) for efficient memory access
        - Software pipelining to hide memory latency
        - SASS-level optimizations for instruction-level parallelism
        """)


if __name__ == "__main__":
    main()
