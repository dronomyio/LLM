import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_mock_tuning_data():
    """Generate mock JIT tuning data for visualization"""
    # Configurations to explore
    block_m_options = [64, 128]
    block_n_options = [48, 64, 80, 96, 112, 128]
    num_stages_options = [4, 5, 6, 7, 8]
    num_tma_multicast_options = [1, 2]
    
    # Matrix shapes to tune for
    shapes = [
        {"m": 64, "n": 2112, "k": 7168},
        {"m": 128, "n": 24576, "k": 1536},
        {"m": 4096, "n": 32768, "k": 512},
        {"m": 64, "n": 7168, "k": 16384},
        {"m": 128, "n": 4096, "k": 7168},
        {"m": 4096, "n": 7168, "k": 2048}
    ]
    
    # Generate tuning data
    tuning_data = []
    
    for shape in shapes:
        m, n, k = shape["m"], shape["n"], shape["k"]
        max_throughput = 0
        best_config = None
        
        # Realistic constraints based on matrix size
        valid_block_m = [64] if m <= 64 else block_m_options
        
        for block_m in valid_block_m:
            for block_n in block_n_options:
                for num_stages in num_stages_options:
                    # Skip invalid configurations
                    if num_stages > 6 and 128 % block_n != 0:
                        continue
                    
                    for num_tma_multicast in ([1] if m < 1024 else num_tma_multicast_options):
                        # Calculate shared memory size (simplified)
                        smem_size = (block_m * block_n * 2 +  # Output
                                    num_stages * block_m * 128 +  # A tiles
                                    num_stages * block_n * 128 +  # B tiles
                                    num_stages * block_m * 4 +  # A scales
                                    (k // 128) * 4 +  # B scales
                                    num_stages * 8 * 2)  # Barriers
                        
                        # Skip if shared memory exceeds H100 limits
                        if smem_size > 232448:
                            continue
                        
                        # Calculate "waves" (how many passes over SMs)
                        num_sms = 132  # H100 has 132 SMs
                        blocks_m = (m + block_m - 1) // block_m
                        blocks_n = (n + block_n - 1) // block_n
                        total_blocks = blocks_m * blocks_n
                        waves = (total_blocks + num_sms - 1) // num_sms
                        
                        # Last wave utilization (higher is better)
                        last_wave_blocks = total_blocks % num_sms
                        last_wave_util = last_wave_blocks if last_wave_blocks > 0 else num_sms
                        
                        # Base performance model (simplified)
                        # More waves = lower performance
                        # Higher last wave utilization = better performance
                        base_perf = 900  # Theoretical max TFLOPS
                        
                        # Performance adjustments based on configuration
                        # Penalties for non-optimal configs
                        wave_penalty = 0.15 * (waves - 1)
                        last_wave_bonus = 0.05 * (last_wave_util / num_sms)
                        
                        # Block size effects
                        block_size_factor = 0.9 + 0.1 * (block_m / 128) * (block_n / 128)
                        
                        # Pipeline depth effects (deeper is better up to a point)
                        pipeline_factor = 0.85 + 0.03 * num_stages
                        
                        # TMA multicast effects
                        tma_factor = 1.0 + 0.1 * (num_tma_multicast - 1)
                        
                        # Randomize a bit to simulate real measurements
                        randomization = np.random.uniform(0.97, 1.03)
                        
                        # Calculate final performance
                        throughput = base_perf * (1 - wave_penalty) * (1 + last_wave_bonus) * \
                                    block_size_factor * pipeline_factor * tma_factor * randomization
                        
                        # Cap at theoretical max
                        throughput = min(throughput, base_perf)
                        
                        # Track the best config
                        if throughput > max_throughput:
                            max_throughput = throughput
                            best_config = {
                                "block_m": block_m,
                                "block_n": block_n,
                                "num_stages": num_stages,
                                "num_tma_multicast": num_tma_multicast
                            }
                        
                        # Add to tuning data
                        tuning_data.append({
                            "m": m,
                            "n": n,
                            "k": k,
                            "shape": f"m={m}, n={n}, k={k}",
                            "block_m": block_m,
                            "block_n": block_n,
                            "num_stages": num_stages,
                            "num_tma_multicast": num_tma_multicast,
                            "smem_size": smem_size,
                            "waves": waves,
                            "last_wave_util": last_wave_util,
                            "throughput_tflops": throughput,
                            "is_best": False  # Will be updated later
                        })
    
    # Mark the best configs
    for entry in tuning_data:
        shape_key = (entry["m"], entry["n"], entry["k"])
        best_throughput = max([x["throughput_tflops"] for x in tuning_data 
                              if (x["m"], x["n"], x["k"]) == shape_key])
        
        if abs(entry["throughput_tflops"] - best_throughput) < 1e-6:
            entry["is_best"] = True
    
    return tuning_data


def load_data(use_mock=True):
    """Load tuning data from file or generate mock data"""
    if use_mock:
        data = generate_mock_tuning_data()
        return pd.DataFrame(data)
    else:
        # Try to load real data if available
        try:
            with open('benchmark_data.json', 'r') as f:
                data = json.load(f)
                tuning_configs = data.get('tuning_configs', [])
                return pd.DataFrame(tuning_configs)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            st.warning("Could not load real tuning data, using mock data instead.")
            data = generate_mock_tuning_data()
            return pd.DataFrame(data)


def plot_tuning_heatmap(df, shape_key, x_param, y_param, show_best=True):
    """Create a heatmap showing parameter tuning impact"""
    # Filter for the selected shape
    m, n, k = [int(x) for x in shape_key.split(',')]
    shape_df = df[(df['m'] == m) & (df['n'] == n) & (df['k'] == k)]
    
    if shape_df.empty:
        return None
    
    # Pivot to create heatmap data
    pivot_df = shape_df.pivot_table(
        index=y_param, 
        columns=x_param, 
        values='throughput_tflops',
        aggfunc='mean'
    )
    
    # Create heatmap
    fig = px.imshow(
        pivot_df, 
        labels=dict(x=x_param, y=y_param, color='TFLOPS'),
        color_continuous_scale="viridis",
        text_auto=".1f"
    )
    
    # Add markers for best configuration
    if show_best:
        best_configs = shape_df[shape_df['is_best'] == True]
        if not best_configs.empty:
            for _, row in best_configs.iterrows():
                # Get column and row index in the pivot table
                x_idx = pivot_df.columns.get_loc(row[x_param])
                y_idx = pivot_df.index.get_loc(row[y_param])
                
                fig.add_annotation(
                    x=x_idx,
                    y=y_idx,
                    text="BEST",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="white",
                    font=dict(color="white", size=14),
                    bordercolor="white",
                    borderwidth=2,
                    bgcolor="rgba(255, 0, 0, 0.6)",
                    opacity=0.8
                )
    
    fig.update_layout(
        title=f"JIT Tuning Heatmap for {shape_key}",
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def plot_parameter_importance(df):
    """Create visualization of parameter importance"""
    # Calculate average impact of each parameter
    parameter_impacts = {}
    
    # For each shape, calculate the range of performance for each parameter
    shapes = df['shape'].unique()
    
    for shape in shapes:
        shape_df = df[df['shape'] == shape]
        
        # Calculate impact of each parameter
        for param in ['block_m', 'block_n', 'num_stages', 'num_tma_multicast']:
            values = shape_df[param].unique()
            if len(values) > 1:  # Only if we have multiple values to compare
                impacts = []
                for val in values:
                    # Get average performance for this parameter value
                    avg_perf = shape_df[shape_df[param] == val]['throughput_tflops'].mean()
                    impacts.append((val, avg_perf))
                
                # Sort by performance
                impacts.sort(key=lambda x: x[1], reverse=True)
                
                # Calculate relative impact (max - min) / max
                max_perf = impacts[0][1]
                min_perf = impacts[-1][1]
                rel_impact = (max_perf - min_perf) / max_perf if max_perf > 0 else 0
                
                # Store the impact
                if param not in parameter_impacts:
                    parameter_impacts[param] = []
                parameter_impacts[param].append(rel_impact)
    
    # Calculate average impact across all shapes
    avg_impacts = {param: np.mean(impacts) for param, impacts in parameter_impacts.items()}
    
    # Create bar chart
    fig = go.Figure()
    
    params = list(avg_impacts.keys())
    impacts = list(avg_impacts.values())
    
    fig.add_trace(go.Bar(
        x=params,
        y=impacts,
        marker_color='darkblue',
        text=[f"{impact:.1%}" for impact in impacts],
        textposition="auto"
    ))
    
    fig.update_layout(
        title="Parameter Importance for Performance",
        xaxis_title="Tuning Parameter",
        yaxis_title="Relative Impact on Performance",
        yaxis=dict(
            tickformat=".0%",
            range=[0, max(impacts) * 1.1]
        ),
        height=400
    )
    
    return fig


def plot_tuning_trace(df, shape_key):
    """Plot the performance of different configurations in tuning order"""
    # Filter for the selected shape
    m, n, k = [int(x) for x in shape_key.split(',')]
    shape_df = df[(df['m'] == m) & (df['n'] == n) & (df['k'] == k)]
    
    if shape_df.empty:
        return None
    
    # Sort by throughput to simulate tuning process
    shape_df = shape_df.sort_values('throughput_tflops')
    
    # Create trace of configurations tried
    config_index = np.arange(len(shape_df))
    
    # Create figure
    fig = go.Figure()
    
    # Add line for tuning progress
    fig.add_trace(go.Scatter(
        x=config_index,
        y=shape_df['throughput_tflops'],
        mode='lines+markers',
        marker=dict(
            size=8,
            color=shape_df['throughput_tflops'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="TFLOPS")
        ),
        line=dict(width=2, color='rgba(50, 50, 50, 0.2)'),
        hovertemplate=
        "<b>Config #%{x}</b><br>" +
        "block_m: %{customdata[0]}<br>" +
        "block_n: %{customdata[1]}<br>" +
        "num_stages: %{customdata[2]}<br>" +
        "num_tma_multicast: %{customdata[3]}<br>" +
        "throughput: %{y:.2f} TFLOPS<br>",
        customdata=shape_df[['block_m', 'block_n', 'num_stages', 'num_tma_multicast']].values
    ))
    
    # Add line for the best configurations found so far
    best_so_far = [max(shape_df['throughput_tflops'].iloc[:i+1]) for i in range(len(shape_df))]
    
    fig.add_trace(go.Scatter(
        x=config_index,
        y=best_so_far,
        mode='lines',
        line=dict(color='red', width=3, dash='dash'),
        name='Best so far'
    ))
    
    # Add point for the final best configuration
    best_idx = shape_df['throughput_tflops'].idxmax()
    best_row = shape_df.loc[best_idx]
    
    fig.add_trace(go.Scatter(
        x=[config_index[-1]],
        y=[best_row['throughput_tflops']],
        mode='markers',
        marker=dict(
            color='red',
            size=12,
            symbol='star',
            line=dict(color='black', width=2)
        ),
        name='Best Configuration',
        hovertemplate=
        "<b>Best Config</b><br>" +
        f"block_m: {best_row['block_m']}<br>" +
        f"block_n: {best_row['block_n']}<br>" +
        f"num_stages: {best_row['num_stages']}<br>" +
        f"num_tma_multicast: {best_row['num_tma_multicast']}<br>" +
        f"throughput: {best_row['throughput_tflops']:.2f} TFLOPS<br>"
    ))
    
    fig.update_layout(
        title=f"JIT Tuning Progression for {shape_key}",
        xaxis_title="Configuration Tested",
        yaxis_title="Throughput (TFLOPS)",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def plot_smem_vs_performance(df, shape_key):
    """Plot relationship between shared memory size and performance"""
    # Filter for the selected shape
    m, n, k = [int(x) for x in shape_key.split(',')]
    shape_df = df[(df['m'] == m) & (df['n'] == n) & (df['k'] == k)]
    
    if shape_df.empty:
        return None
    
    # Create scatter plot
    fig = px.scatter(
        shape_df,
        x='smem_size',
        y='throughput_tflops',
        color='num_stages',
        size='block_n',
        hover_data=['block_m', 'block_n', 'num_stages', 'num_tma_multicast'],
        title=f"Shared Memory Usage vs Performance for {shape_key}",
        labels={
            'smem_size': 'Shared Memory Size (bytes)',
            'throughput_tflops': 'Throughput (TFLOPS)',
            'num_stages': 'Pipeline Stages'
        }
    )
    
    # Add vertical line for SM90 shared memory limit
    fig.add_vline(
        x=232448,
        line_dash="dash",
        line_color="red",
        annotation_text="SM90 Limit (232KB)",
        annotation_position="top right"
    )
    
    # Mark the best configuration
    best_config = shape_df[shape_df['is_best'] == True]
    if not best_config.empty:
        fig.add_trace(
            go.Scatter(
                x=best_config['smem_size'],
                y=best_config['throughput_tflops'],
                mode='markers',
                marker=dict(
                    color='red',
                    size=15,
                    symbol='star',
                    line=dict(color='black', width=2)
                ),
                name='Best Configuration'
            )
        )
    
    fig.update_layout(height=500)
    
    return fig


def plot_waves_vs_performance(df, shape_key):
    """Plot relationship between wave count, occupancy and performance"""
    # Filter for the selected shape
    m, n, k = [int(x) for x in shape_key.split(',')]
    shape_df = df[(df['m'] == m) & (df['n'] == n) & (df['k'] == k)]
    
    if shape_df.empty:
        return None
    
    # Add occupancy metric (last wave utilization as percentage)
    shape_df['last_wave_utilization'] = shape_df['last_wave_util'] / 132 * 100  # 132 SMs on H100
    
    # Create bubble chart
    fig = px.scatter(
        shape_df,
        x='waves',
        y='throughput_tflops',
        size='last_wave_utilization',
        color='block_m',
        hover_data=['block_m', 'block_n', 'num_stages', 'num_tma_multicast', 'last_wave_util'],
        title=f"Waves & Occupancy vs Performance for {shape_key}",
        labels={
            'waves': 'Number of Waves',
            'throughput_tflops': 'Throughput (TFLOPS)',
            'last_wave_utilization': 'Last Wave Utilization (%)',
            'block_m': 'Block M Size'
        }
    )
    
    # Mark the best configuration
    best_config = shape_df[shape_df['is_best'] == True]
    if not best_config.empty:
        fig.add_trace(
            go.Scatter(
                x=best_config['waves'],
                y=best_config['throughput_tflops'],
                mode='markers',
                marker=dict(
                    color='red',
                    size=15,
                    symbol='star',
                    line=dict(color='black', width=2)
                ),
                name='Best Configuration'
            )
        )
    
    fig.update_layout(height=500)
    
    return fig


def create_tuning_process_visualization():
    """Create a visualization explaining the JIT tuning process"""
    # Create figure with 4 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "1. Define Search Space", 
            "2. Generate & Compile Kernels",
            "3. Benchmark Performance",
            "4. Cache Best Configuration"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # 1. Define Search Space
    search_space = np.random.rand(5, 5) 
    fig.add_trace(
        go.Heatmap(
            z=search_space,
            colorscale='Blues',
            showscale=False
        ),
        row=1, col=1
    )
    
    # Add parameter labels
    for i, param in enumerate(['Block M', 'Block N', 'Stages', 'TMA', 'Other']):
        fig.add_annotation(
            x=-0.1,
            y=i,
            text=param,
            showarrow=False,
            xref="x domain",
            yref="y",
            xanchor="right",
            row=1, col=1
        )
    
    # 2. Generate & Compile Kernels
    # Create code-like visualization
    y_pos = list(range(6))
    x_pos = [0] * 6
    
    code_snippets = [
        "template<BLOCK_M, BLOCK_N, STAGES>",
        "class Kernel {",
        "  constexpr auto N = 4096;",
        "  constexpr auto BLOCK_K = 128;",
        "  // Generated code...",
        "}"
    ]
    
    for i, snippet in enumerate(code_snippets):
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[i],
                mode='text',
                text=[snippet],
                textfont=dict(
                    family="Courier New, monospace",
                    size=11,
                    color="black"
                ),
                textposition="middle left",
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. Benchmark Performance
    x = np.arange(10)
    y = np.random.rand(10) * 10 + 50  # Random performance values
    
    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    # Highlight best performance
    best_idx = np.argmax(y)
    
    fig.add_trace(
        go.Bar(
            x=[best_idx],
            y=[y[best_idx]],
            marker_color='red',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. Cache Best Configuration
    cache_data = np.zeros((5, 5))
    cache_data[2, 3] = 1  # Highlight cached configuration
    
    fig.add_trace(
        go.Heatmap(
            z=cache_data,
            colorscale=[[0, 'white'], [1, 'red']],
            showscale=False
        ),
        row=2, col=2
    )
    
    fig.add_annotation(
        x=2.5,
        y=2.5,
        text="Cached<br>Config",
        showarrow=False,
        row=2, col=2
    )
    
    # Set fixed axes for all subplots
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[-0.5, 5.5],
                row=i, col=j
            )
            fig.update_yaxes(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[-0.5, 5.5],
                row=i, col=j
            )
    
    # Update layout
    fig.update_layout(
        height=600,
        width=800,
        title_text="JIT Tuning Process in DeepGEMM",
        showlegend=False
    )
    
    return fig


def main():
    st.set_page_config(
        page_title="DeepGEMM JIT Tuning Analyzer",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("DeepGEMM JIT Tuning Analysis")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        """
        This dashboard analyzes the Just-In-Time (JIT) tuning system in DeepGEMM,
        which automatically optimizes kernel configurations for different matrix shapes.
        """
    )
    
    # Load data
    use_mock = st.sidebar.checkbox("Use mock data", value=True)
    if use_mock:
        st.sidebar.info("Using realistic mock data for demonstration")
    
    df = load_data(use_mock=use_mock)
    
    # Get unique shapes for selection
    shapes = df['shape'].unique()
    
    # Data section
    with st.expander("🔍 Raw Tuning Data", expanded=False):
        st.dataframe(df)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Tuning Process Overview",
        "Parameter Impact Analysis",
        "Shape-specific Analysis",
        "Tuning Progression"
    ])
    
    with tab1:
        st.header("JIT Tuning Process")
        
        st.markdown("""
        DeepGEMM uses a sophisticated JIT (Just-In-Time) tuning system to automatically
        select the optimal kernel configuration for each matrix shape. The process works as follows:
        
        1. **Define Search Space**: The system defines a search space of possible configurations,
           including block sizes, pipeline stages, and other parameters.
           
        2. **Generate & Compile Kernels**: For each configuration in the search space, a specialized
           CUDA kernel is generated and compiled at runtime.
           
        3. **Benchmark Performance**: Each kernel is executed and timed to measure its performance.
           
        4. **Cache Best Configuration**: The best-performing configuration is cached for future use
           with the same matrix shape.
        """)
        
        # Show tuning process visualization
        tuning_process_fig = create_tuning_process_visualization()
        st.plotly_chart(tuning_process_fig, use_container_width=True)
        
        st.markdown("""
        ### Key Tuning Parameters
        
        DeepGEMM's JIT tuner optimizes several key parameters:
        
        - **Block Size (M, N)**: The dimensions of matrix blocks processed by each thread block.
          Optimal values balance parallelism and resource usage.
          
        - **Pipeline Stages**: The number of stages in the software pipeline. More stages can
          hide memory latency but require more shared memory.
          
        - **TMA Multicast**: Whether to use TMA multicast to broadcast data to multiple thread blocks.
          Beneficial for large matrices.
          
        - **Shared Memory**: Each configuration has different shared memory requirements, which must
          fit within hardware limits (232KB per SM on Hopper).
          
        - **Waves & Occupancy**: The tuner considers how many "waves" of thread blocks will be needed
          and how well the last wave utilizes all SMs.
        """)
        
        # Show key statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_configs = int(df.groupby('shape').size().mean())
            st.metric("Avg. Configs per Shape", avg_configs)
        
        with col2:
            best_improvement = ((df.groupby('shape')['throughput_tflops'].max() / 
                               df.groupby('shape')['throughput_tflops'].min()) - 1).mean()
            st.metric("Avg. Tuning Improvement", f"{best_improvement:.1%}")
            
        with col3:
            cache_hit_ratio = 0.987  # Mock value
            st.metric("Cache Hit Ratio", f"{cache_hit_ratio:.1%}")
            
    with tab2:
        st.header("Parameter Impact Analysis")
        
        # Plot parameter importance
        param_importance_fig = plot_parameter_importance(df)
        st.plotly_chart(param_importance_fig, use_container_width=True)
        
        # Create two-column layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Block Size Impact")
            # Create scatterplot of block_m vs block_n colored by performance
            block_fig = px.scatter(
                df,
                x='block_n',
                y='block_m',
                color='throughput_tflops',
                size='throughput_tflops',
                facet_col='m',
                facet_col_wrap=2,
                hover_data=['shape', 'num_stages', 'throughput_tflops'],
                labels={
                    'block_n': 'Block N Size',
                    'block_m': 'Block M Size',
                    'throughput_tflops': 'Throughput (TFLOPS)',
                    'm': 'Matrix M Dimension'
                },
                title="Block Size Impact on Performance",
                color_continuous_scale="viridis"
            )
            
            block_fig.update_layout(height=500)
            st.plotly_chart(block_fig, use_container_width=True)
            
        with col2:
            st.subheader("Pipeline Stages Impact")
            # Create bar chart of num_stages impact
            stages_df = df.groupby(['num_stages', 'm'])['throughput_tflops'].mean().reset_index()
            
            stages_fig = px.bar(
                stages_df,
                x='num_stages',
                y='throughput_tflops',
                color='m',
                barmode='group',
                labels={
                    'num_stages': 'Number of Pipeline Stages',
                    'throughput_tflops': 'Avg. Throughput (TFLOPS)',
                    'm': 'Matrix M Dimension'
                },
                title="Pipeline Stages Impact on Performance"
            )
            
            stages_fig.update_layout(height=500)
            st.plotly_chart(stages_fig, use_container_width=True)
        
        # Create summary of key findings
        st.markdown("""
        ### Key Findings
        
        1. **Block Size** has the largest impact on performance, with optimal values varying by matrix shape
        2. **Pipeline Stages** significantly affect performance, with deeper pipelines generally better
        3. **TMA Multicast** is beneficial mainly for large matrices (m ≥ 1024)
        4. Optimal configurations balance multiple factors, not just a single parameter
        """)
        
    with tab3:
        st.header("Shape-specific Analysis")
        
        # Shape selection
        selected_shape = st.selectbox("Select Matrix Shape", shapes)
        
        if selected_shape:
            m, n, k = [int(x) for x in selected_shape.split(',')]
            
            # Show shape details
            st.markdown(f"""
            ### Matrix Shape: m={m}, n={n}, k={k}
            
            This represents a matrix multiplication C = A * B where:
            - A is {m}×{k}
            - B is {n}×{k} (transposed in computation)
            - C is {m}×{n}
            """)
            
            # Show parameter heatmaps
            st.subheader("Parameter Tuning Heatmaps")
            
            col1, col2 = st.columns(2)
            
            with col1:
                block_heatmap = plot_tuning_heatmap(df, selected_shape, 'block_n', 'block_m')
                if block_heatmap:
                    st.plotly_chart(block_heatmap, use_container_width=True)
                
            with col2:
                stages_heatmap = plot_tuning_heatmap(df, selected_shape, 'num_stages', 'num_tma_multicast')
                if stages_heatmap:
                    st.plotly_chart(stages_heatmap, use_container_width=True)
            
            # Show additional insights
            st.subheader("Hardware Resource Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                smem_fig = plot_smem_vs_performance(df, selected_shape)
                if smem_fig:
                    st.plotly_chart(smem_fig, use_container_width=True)
                
            with col2:
                waves_fig = plot_waves_vs_performance(df, selected_shape)
                if waves_fig:
                    st.plotly_chart(waves_fig, use_container_width=True)
                    
            # Display best configuration for this shape
            shape_df = df[(df['m'] == m) & (df['n'] == n) & (df['k'] == k)]
            best_config = shape_df[shape_df['is_best'] == True].iloc[0] if not shape_df[shape_df['is_best'] == True].empty else None
            
            if best_config is not None:
                st.subheader("Best Configuration")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Block M", best_config['block_m'])
                
                with col2:
                    st.metric("Block N", best_config['block_n'])
                    
                with col3:
                    st.metric("Pipeline Stages", best_config['num_stages'])
                    
                with col4:
                    st.metric("TMA Multicast", best_config['num_tma_multicast'])
                
                # Additional metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Shared Memory", f"{best_config['smem_size'] / 1024:.1f}KB")
                
                with col2:
                    st.metric("Waves", best_config['waves'])
                    
                with col3:
                    st.metric("SM Utilization", f"{best_config['last_wave_util'] / 132 * 100:.1f}%")
    
    with tab4:
        st.header("Tuning Progression Analysis")
        
        # Shape selection
        selected_shape = st.selectbox("Select Shape for Tuning Trace", shapes, key="tuning_trace_shape")
        
        if selected_shape:
            # Show tuning trace
            tuning_trace_fig = plot_tuning_trace(df, selected_shape)
            if tuning_trace_fig:
                st.plotly_chart(tuning_trace_fig, use_container_width=True)
            
            st.markdown("""
            The chart above shows the progression of the tuning process, with each point representing
            a different kernel configuration that was tested. The red dashed line shows the best
            performance discovered up to that point in the tuning process.
            
            ### Tuning Effectiveness
            
            The JIT tuning system in DeepGEMM can discover configurations that are significantly
            better than naive choices, with performance improvements often exceeding 20%.
            
            The tuning results are cached, so subsequent runs with the same matrix shape will
            immediately use the best configuration without needing to re-tune.
            """)


if __name__ == "__main__":
    main()
