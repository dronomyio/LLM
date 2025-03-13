import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def format_to_bits(number, bits, exp_bits, mantissa_bits, bias):
    """Format a floating point number to its binary representation"""
    if number == 0:
        # Special case for zero
        return "0" * bits
    
    # Handle sign bit
    sign = 0 if number >= 0 else 1
    number = abs(number)
    
    # Find exponent
    exponent = 0
    normalized = number
    
    if normalized >= 1.0:
        while normalized >= 2.0:
            normalized /= 2.0
            exponent += 1
    else:
        while normalized < 1.0:
            normalized *= 2.0
            exponent -= 1
    
    # Apply bias to exponent
    biased_exponent = exponent + bias
    
    # Ensure exponent is within valid range
    if biased_exponent < 0:
        # Underflow to zero
        return "0" * bits
    elif biased_exponent >= 2**exp_bits:
        # Overflow to infinity
        exponent_bits = "1" * exp_bits
        mantissa_bits_str = "0" * mantissa_bits
        return f"{sign}{exponent_bits}{mantissa_bits_str}"
    
    # Convert exponent to binary
    exponent_bits = bin(biased_exponent)[2:].zfill(exp_bits)
    
    # Calculate mantissa bits
    mantissa = normalized - 1.0  # Remove hidden bit
    mantissa_bits_str = ""
    for i in range(mantissa_bits):
        mantissa *= 2
        bit = int(mantissa)
        mantissa_bits_str += str(bit)
        mantissa -= bit
    
    return f"{sign}{exponent_bits}{mantissa_bits_str}"

def binary_to_decimal(binary_str, exp_bits, mantissa_bits, bias):
    """Convert binary representation back to floating point"""
    if binary_str == "0" * len(binary_str):
        return 0.0
    
    sign_bit = int(binary_str[0])
    exponent_bits = binary_str[1:1+exp_bits]
    mantissa_bits = binary_str[1+exp_bits:]
    
    sign = -1 if sign_bit else 1
    exponent = int(exponent_bits, 2) - bias
    
    # Calculate mantissa value (add hidden bit)
    mantissa = 1.0
    for i, bit in enumerate(mantissa_bits):
        if bit == '1':
            mantissa += 2 ** -(i + 1)
    
    return sign * mantissa * (2 ** exponent)

def plot_fp_comparison():
    """Create visualization of different floating point formats"""
    # Create figure
    fig = go.Figure()
    
    # Define formats
    formats = [
        {"name": "FP32", "bits": 32, "exp": 8, "mantissa": 23, "color": "blue"},
        {"name": "FP16", "bits": 16, "exp": 5, "mantissa": 10, "color": "green"},
        {"name": "BF16", "bits": 16, "exp": 8, "mantissa": 7, "color": "orange"},
        {"name": "FP8 (E4M3)", "bits": 8, "exp": 4, "mantissa": 3, "color": "red"}
    ]
    
    # Add traces for bit visualization
    y_position = 0
    annotations = []
    
    for fmt in formats:
        # Calculate bias
        bias = 2**(fmt["exp"]-1) - 1
        
        # Create bit representation
        sign_bar = go.Bar(
            x=[1], y=[y_position], 
            orientation='h',
            marker=dict(color='lightblue'),
            showlegend=False,
            hoverinfo='none'
        )
        
        exp_bar = go.Bar(
            x=[fmt["exp"]], y=[y_position], 
            orientation='h',
            marker=dict(color='lightgreen'),
            showlegend=False,
            hoverinfo='none'
        )
        
        mantissa_bar = go.Bar(
            x=[fmt["mantissa"]], y=[y_position], 
            orientation='h',
            marker=dict(color='salmon'),
            showlegend=False,
            hoverinfo='none'
        )
        
        fig.add_trace(sign_bar)
        fig.add_trace(exp_bar)
        fig.add_trace(mantissa_bar)
        
        # Add annotations
        annotations.append(dict(
            x=0.5, y=y_position,
            text="S",
            showarrow=False,
            font=dict(color='black')
        ))
        
        annotations.append(dict(
            x=1 + fmt["exp"]/2, y=y_position,
            text="Exp",
            showarrow=False,
            font=dict(color='black')
        ))
        
        annotations.append(dict(
            x=1 + fmt["exp"] + fmt["mantissa"]/2, y=y_position,
            text="Mantissa",
            showarrow=False,
            font=dict(color='black')
        ))
        
        annotations.append(dict(
            x=-2, y=y_position,
            text=fmt["name"],
            showarrow=False,
            font=dict(color='black', size=14)
        ))
        
        # Increment y position for next format
        y_position += 1
    
    fig.update_layout(
        barmode='stack',
        title="Floating Point Format Comparison",
        xaxis=dict(
            title="Number of Bits",
            range=[-5, 35]
        ),
        yaxis=dict(
            showticklabels=False,
            range=[-0.5, len(formats) - 0.5]
        ),
        annotations=annotations,
        height=400,
        margin=dict(l=100)
    )
    
    return fig

def plot_dynamic_range_comparison():
    """Plot the dynamic range of different floating point formats"""
    # Create a range of values
    x = np.logspace(-15, 15, 1000)
    
    # Create figure
    fig = go.Figure()
    
    # Define formats with min and max values
    formats = [
        {"name": "FP32", "min": 2**-126, "max": (2-2**-23) * 2**127, "color": "blue"},
        {"name": "FP16", "min": 2**-14, "max": (2-2**-10) * 2**15, "color": "green"},
        {"name": "BF16", "min": 2**-126, "max": (2-2**-7) * 2**127, "color": "orange"},
        {"name": "FP8 (E4M3)", "min": 2**-6, "max": (2-2**-3) * 2**7, "color": "red"}
    ]
    
    # Add horizontal lines for each format
    for i, fmt in enumerate(formats):
        fig.add_trace(go.Scatter(
            x=[fmt["min"], fmt["max"]],
            y=[i, i],
            mode='lines',
            line=dict(color=fmt["color"], width=10),
            name=fmt["name"]
        ))
        
        # Add min and max annotations
        fig.add_annotation(
            x=fmt["min"],
            y=i,
            text=f"{fmt['min']:.2e}",
            showarrow=False,
            yshift=20,
            xshift=-30,
            font=dict(size=10)
        )
        
        fig.add_annotation(
            x=fmt["max"],
            y=i,
            text=f"{fmt['max']:.2e}",
            showarrow=False,
            yshift=20,
            xshift=30,
            font=dict(size=10)
        )
    
    fig.update_layout(
        title="Dynamic Range Comparison of Floating Point Formats",
        xaxis=dict(
            type="log",
            title="Range (log scale)"
        ),
        yaxis=dict(
            tickvals=list(range(len(formats))),
            ticktext=[fmt["name"] for fmt in formats],
            title=""
        ),
        height=400
    )
    
    return fig

def plot_precision_comparison():
    """Plot precision comparison of different floating point formats"""
    # Create figure
    fig = go.Figure()
    
    # Define ranges to show precision differences
    x = np.linspace(1, 2, 1000)
    
    # Calculate representable values for different formats
    def get_representable_values(exp_bits, mantissa_bits, exponent):
        bias = 2**(exp_bits-1) - 1
        biased_exp = exponent + bias
        
        values = []
        for i in range(2**mantissa_bits):
            mantissa = 0
            for bit in range(mantissa_bits):
                if (i >> bit) & 1:
                    mantissa += 2**-(bit+1)
            value = (1 + mantissa) * (2**exponent)
            values.append(value)
        
        return sorted(values)
    
    formats = [
        {"name": "FP32", "exp": 8, "mantissa": 23, "color": "blue"},
        {"name": "FP16", "exp": 5, "mantissa": 10, "color": "green"},
        {"name": "BF16", "exp": 8, "mantissa": 7, "color": "orange"},
        {"name": "FP8 (E4M3)", "exp": 4, "mantissa": 3, "color": "red"}
    ]
    
    # Plot representable values between 1 and 2
    for fmt in formats:
        values = get_representable_values(fmt["exp"], fmt["mantissa"], 0)
        if len(values) > 100:
            # Subsample for clarity if too many points
            values = values[::len(values)//100]
        
        fig.add_trace(go.Scatter(
            x=values,
            y=[fmt["name"]] * len(values),
            mode='markers',
            marker=dict(
                color=fmt["color"],
                size=8,
                opacity=0.7
            ),
            name=fmt["name"]
        ))
    
    fig.update_layout(
        title="Precision Comparison: Representable Values Between 1 and 2",
        xaxis=dict(
            title="Value",
            range=[1, 2]
        ),
        yaxis=dict(
            title="",
            categoryorder='array',
            categoryarray=[fmt["name"] for fmt in reversed(formats)]
        ),
        height=400
    )
    
    return fig

def plot_scaling_comparison():
    """Visualize how scaling helps FP8 precision"""
    # Create a set of example values that highlight the issue
    values = [100.5, 150.75, 200.25, 0.065, 0.0125, 0.037, 1000.5, 750.25]
    values_array = np.array(values)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Original Values (FP32)", "Direct FP8 Conversion (Loss of Precision)", 
                         "Scaled Values", "FP8 with Per-Group Scaling")
    )
    
    # Calculate FP8 E4M3 values (simplified)
    def fp8_quantize(val):
        # Simplified implementation of FP8 E4M3 quantization
        # In reality, this would involve proper rounding, handling denormals, etc.
        binary = format_to_bits(val, 8, 4, 3, 7)
        return binary_to_decimal(binary, 4, 3, 7)
    
    # Calculate with and without scaling
    fp8_direct = [fp8_quantize(val) for val in values]
    
    # Calculate scaling factors (per 4-element group)
    scaling_factors = []
    for i in range(0, len(values), 4):
        group = values_array[i:i+4]
        max_abs = np.max(np.abs(group))
        # Use a target range that maximizes FP8 precision
        scale = 8.0 / max_abs if max_abs > 0 else 1.0
        scaling_factors.extend([scale] * min(4, len(values) - i))
    
    # Apply scaling
    scaled_values = values_array * np.array(scaling_factors)
    fp8_scaled = [fp8_quantize(val) for val in scaled_values]
    fp8_descaled = np.array(fp8_scaled) / np.array(scaling_factors)
    
    # Plot original values
    fig.add_trace(
        go.Bar(x=list(range(len(values))), y=values, name="FP32 Original"),
        row=1, col=1
    )
    
    # Plot direct FP8 conversion
    error_direct = [abs(fp8 - orig)/max(abs(orig), 1e-10) for fp8, orig in zip(fp8_direct, values)]
    fig.add_trace(
        go.Bar(
            x=list(range(len(values))), 
            y=fp8_direct, 
            name="FP8 Direct",
            marker=dict(
                color=[f'rgba(255, 0, 0, {min(err * 10, 1.0)})' for err in error_direct]
            )
        ),
        row=1, col=2
    )
    
    # Plot scaled values
    fig.add_trace(
        go.Bar(
            x=list(range(len(values))), 
            y=scaled_values, 
            name="Scaled Values",
            marker=dict(color='green')
        ),
        row=2, col=1
    )
    
    # Add scaling factor line
    fig.add_trace(
        go.Scatter(
            x=list(range(len(values))),
            y=scaling_factors,
            mode='lines+markers',
            line=dict(color='black', dash='dash'),
            name="Scaling Factors",
            yaxis="y2"
        ),
        row=2, col=1
    )
    
    # Plot FP8 with scaling
    error_scaled = [abs(fp8 - orig)/max(abs(orig), 1e-10) for fp8, orig in zip(fp8_descaled, values)]
    fig.add_trace(
        go.Bar(
            x=list(range(len(values))), 
            y=fp8_descaled, 
            name="FP8 with Scaling",
            marker=dict(
                color=[f'rgba(0, 128, 0, {min(err * 10, 1.0)})' for err in error_scaled]
            )
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        title_text="Fine-Grained Scaling for FP8 Precision",
        showlegend=False
    )
    
    # Update y-axis for scaling factors
    fig.update_yaxes(title_text="Scale Factor", range=[0, max(scaling_factors) * 1.2], secondary_y=True, row=2, col=1)
    
    # Update x-axes
    for row in range(1, 3):
        for col in range(1, 3):
            fig.update_xaxes(
                ticktext=['Value ' + str(i) for i in range(len(values))],
                tickvals=list(range(len(values))),
                row=row, col=col
            )
    
    return fig

def plot_channel_scaling():
    """Visualize per-128-channel scaling in DeepGEMM"""
    # Create a figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Matrix with Per-Channel Scaling", "DeepGEMM Implementation"),
        specs=[[{"type": "heatmap"}, {"type": "table"}]],
        column_widths=[0.6, 0.4]
    )
    
    # Create a sample matrix (128 x 128) with different value ranges
    np.random.seed(42)
    matrix = np.zeros((128, 384))  # 3 blocks of 128 columns
    
    # Different regions with different magnitudes
    matrix[:, :128] = np.random.normal(0, 1, (128, 128))  # Small values
    matrix[:, 128:256] = np.random.normal(0, 10, (128, 128))  # Medium values
    matrix[:, 256:] = np.random.normal(0, 100, (128, 128))  # Large values
    
    # Add some structure for visualization
    matrix[32:64, 32:64] = np.random.normal(0, 5, (32, 32))
    matrix[64:96, 160:192] = np.random.normal(0, 50, (32, 32))
    matrix[32:64, 288:320] = np.random.normal(0, 500, (32, 32))
    
    # Create scaling factors (simplified for visualization)
    scaling = np.zeros((1, 3))  # One scaling factor per 128 columns
    
    # Calculate max absolute value per region
    for i in range(3):
        scaling[0, i] = np.max(np.abs(matrix[:, i*128:(i+1)*128])) / 7.0  # Scale to optimal FP8 range
    
    # Create heatmap of matrix
    fig.add_trace(
        go.Heatmap(
            z=np.log1p(np.abs(matrix)),  # Log scale for better visualization
            colorscale='Viridis',
            colorbar=dict(title='Log(abs(Value) + 1)'),
            showscale=True
        ),
        row=1, col=1
    )
    
    # Add annotations for scaling factors
    for i in range(3):
        fig.add_annotation(
            x=i*128 + 64,  # Center of each block
            y=-10,
            text=f"Scale: {scaling[0, i]:.2f}",
            showarrow=False,
            font=dict(size=12)
        )
        
        # Add vertical lines to separate blocks
        if i > 0:
            fig.add_shape(
                type="line",
                x0=i*128, y0=0,
                x1=i*128, y1=128,
                line=dict(color="white", width=2, dash="dash")
            )
    
    # Create code example table
    code_example = [
        ["1", 'x_view = x.view(m, -1, 128)'],
        ["2", 'x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)'],
        ["3", 'scale = 448.0 / x_amax.unsqueeze(2)'],
        ["4", 'x_fp8 = (x_view * scale).to(torch.float8_e4m3fn)'],
        ["5", 'scale_factor = (x_amax / 448.0).view(m, -1)']
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Line", "Code"],
                fill_color='lightblue',
                align='center'
            ),
            cells=dict(
                values=list(zip(*code_example)),
                fill_color='white',
                align='left',
                font=dict(family="Courier New, monospace", size=12)
            )
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Per-128-Channel Scaling in DeepGEMM",
        height=600
    )
    
    # Update x and y axes for heatmap
    fig.update_xaxes(
        title_text="Columns (grouped in 128-channel blocks)",
        range=[-10, 384],
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Rows",
        range=[128, -20],  # Reverse y-axis to match matrix orientation
        row=1, col=1
    )
    
    return fig

def plot_matmul_with_scaling():
    """Visualize matrix multiplication with per-channel scaling"""
    # Create figure
    fig = go.Figure()
    
    # Define matrix sizes and colors
    matrices = [
        {"name": "A (FP8)", "rows": 6, "cols": 8, "color": "lightblue", "x": 0, "y": 0},
        {"name": "× B (FP8)", "rows": 4, "cols": 8, "color": "lightgreen", "x": 10, "y": 0, "transpose": True},
        {"name": "= C (BF16)", "rows": 6, "cols": 4, "color": "lightsalmon", "x": 20, "y": 0}
    ]
    
    # Add scaling factor boxes
    scales = [
        {"name": "A Scales", "rows": 6, "cols": 1, "color": "blue", "x": 0, "y": 9, "opacity": 0.7},
        {"name": "B Scales", "rows": 1, "cols": 1, "color": "green", "x": 10, "y": 9, "opacity": 0.7}
    ]
    
    # Add matrices to figure
    for matrix in matrices:
        # Add rectangle for matrix
        width = matrix["cols"]
        height = matrix["rows"]
        
        if matrix.get("transpose", False):
            width, height = height, width
        
        fig.add_shape(
            type="rect",
            x0=matrix["x"],
            y0=matrix["y"],
            x1=matrix["x"] + width,
            y1=matrix["y"] + height,
            fillcolor=matrix["color"],
            line=dict(color="black"),
            opacity=0.7
        )
        
        # Add label
        fig.add_annotation(
            x=matrix["x"] + width/2,
            y=matrix["y"] + height/2,
            text=matrix["name"],
            showarrow=False,
            font=dict(size=14, color="black")
        )
        
        # Add grid lines
        for i in range(1, max(matrix["rows"], matrix["cols"])):
            if i < matrix["rows"] and not matrix.get("transpose", False):
                fig.add_shape(
                    type="line",
                    x0=matrix["x"],
                    y0=matrix["y"] + i,
                    x1=matrix["x"] + matrix["cols"],
                    y1=matrix["y"] + i,
                    line=dict(color="black", width=0.5)
                )
            
            if i < matrix["cols"]:
                if matrix.get("transpose", False):
                    fig.add_shape(
                        type="line",
                        x0=matrix["x"] + i,
                        y0=matrix["y"],
                        x1=matrix["x"] + i,
                        y1=matrix["y"] + matrix["rows"],
                        line=dict(color="black", width=0.5)
                    )
                else:
                    fig.add_shape(
                        type="line",
                        x0=matrix["x"] + i,
                        y0=matrix["y"],
                        x1=matrix["x"] + i,
                        y1=matrix["y"] + matrix["rows"],
                        line=dict(color="black", width=0.5)
                    )
    
    # Add scaling factors
    for scale in scales:
        fig.add_shape(
            type="rect",
            x0=scale["x"],
            y0=scale["y"],
            x1=scale["x"] + scale["cols"],
            y1=scale["y"] + scale["rows"],
            fillcolor=scale["color"],
            line=dict(color="black"),
            opacity=scale["opacity"]
        )
        
        # Add label
        fig.add_annotation(
            x=scale["x"] + scale["cols"]/2,
            y=scale["y"] + scale["rows"]/2,
            text=scale["name"],
            showarrow=False,
            font=dict(size=14, color="white")
        )
    
    # Add arrows showing the scaling application
    fig.add_shape(
        type="path",
        path=f"M {scales[0]['x'] + scales[0]['cols']/2} {scales[0]['y']} V {matrices[0]['y'] + matrices[0]['rows']}",
        line=dict(color="blue", width=2, dash="dash")
    )
    
    fig.add_shape(
        type="path",
        path=f"M {scales[1]['x'] + scales[1]['cols']/2} {scales[1]['y']} V {matrices[1]['y'] + matrices[1]['rows']}",
        line=dict(color="green", width=2, dash="dash")
    )
    
    # Add annotation labels for 128-column blocks
    for i in range(2):
        k_block = i * 4
        x_pos = matrices[0]["x"] + k_block + 2
        y_pos = matrices[0]["y"] - 0.5
        
        fig.add_annotation(
            x=x_pos,
            y=y_pos,
            text=f"Block {i+1}",
            showarrow=False,
            font=dict(size=12)
        )
    
    # Update layout
    fig.update_layout(
        title="Matrix Multiplication with Per-128-Channel Scaling",
        showlegend=False,
        width=800,
        height=600,
        xaxis=dict(
            range=[-2, 26],
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            range=[-1, 11],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1
        )
    )
    
    # Add annotations explaining the process
    steps = [
        {"text": "1. Matrix A is divided into 128-column blocks", "x": 0, "y": -0.5},
        {"text": "2. Each block has its own scaling factor", "x": 0, "y": 11},
        {"text": "3. Matrix B is similarly blocked with scaling factors", "x": 10, "y": 11},
        {"text": "4. During computation, A and B values are scaled back to higher precision", "x": 15, "y": 5},
        {"text": "5. Result is in higher precision (BF16)", "x": 20, "y": -0.5}
    ]
    
    for i, step in enumerate(steps):
        fig.add_annotation(
            x=step["x"],
            y=step["y"],
            text=step["text"],
            showarrow=False,
            font=dict(size=12),
            align="left"
        )
    
    return fig

def main():
    st.set_page_config(
        page_title="DeepGEMM FP8 Fine-Grained Scaling Visualization",
        page_icon="🧮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("DeepGEMM FP8 Fine-Grained Scaling")
    
    st.sidebar.header("About")
    st.sidebar.info(
        """
        This dashboard visualizes how DeepGEMM uses fine-grained per-128-channel 
        scaling factors to maintain precision when using FP8 (8-bit floating point) data types.
        """
    )
    
    st.sidebar.header("Navigation")
    visualization = st.sidebar.radio(
        "Select Visualization",
        ["FP8 Overview", "Dynamic Range", "Precision Comparison", 
         "Scaling Visualization", "Per-Channel Scaling", "Matrix Multiplication"]
    )
    
    if visualization == "FP8 Overview":
        st.header("FP8 Floating Point Format")
        
        st.markdown("""
        ### What is FP8?
        
        FP8 (8-bit floating point) is a reduced-precision data format used in deep learning to:
        
        - **Reduce memory footprint**: 4x smaller than FP32
        - **Increase computational throughput**: 4x more operations per second
        - **Improve energy efficiency**: Less memory movement and simpler calculations
        
        The most common FP8 format is **E4M3**, which has:
        - 1 sign bit
        - 4 exponent bits
        - 3 mantissa bits
        
        ### Challenges of FP8
        
        While FP8 offers performance advantages, it comes with precision challenges:
        
        - **Limited dynamic range**: Can only represent a narrow range of values
        - **Low precision**: Only 3 bits for the fractional component
        - **Quantization error**: Significant rounding errors compared to FP32
        
        DeepGEMM addresses these challenges with fine-grained scaling techniques.
        """)
        
        # Create the visualization
        fp_fig = plot_fp_comparison()
        st.plotly_chart(fp_fig, use_container_width=True)
        
        st.markdown("""
        ### Key Format Characteristics
        
        | Format | Size (bits) | Exponent Bits | Mantissa Bits | Bias | Dynamic Range | Precision |
        |--------|-------------|---------------|---------------|------|---------------|-----------|
        | FP32   | 32          | 8             | 23            | 127  | ~1.4e-45 to ~3.4e38  | 2^-23 ≈ 1.2e-7 |
        | FP16   | 16          | 5             | 10            | 15   | ~6.0e-8 to ~6.5e4    | 2^-10 ≈ 9.8e-4 |
        | BF16   | 16          | 8             | 7             | 127  | ~1.4e-45 to ~3.4e38  | 2^-7 ≈ 7.8e-3  |
        | FP8 (E4M3) | 8      | 4             | 3             | 7    | ~3.0e-3 to ~240      | 2^-3 ≈ 0.125   |
        
        DeepGEMM uses **FP8 (E4M3)** for input matrices and **BF16** for output, combined with
        fine-grained scaling to maintain numerical accuracy.
        """)
        
    elif visualization == "Dynamic Range":
        st.header("Dynamic Range of Floating Point Formats")
        
        st.markdown("""
        ### Dynamic Range Comparison
        
        The **dynamic range** of a floating point format is the ratio between the largest and smallest 
        (non-zero) values it can represent. This is primarily determined by the number of exponent bits.
        
        FP8 (E4M3) has:
        - Minimum value: 2^-6 ≈ 0.015625
        - Maximum value: (2-2^-3) × 2^7 ≈ 240
        
        This is a much narrower range than FP32's approximately 10^84 dynamic range, which means 
        FP8 cannot directly represent very large or very small values without scaling.
        """)
        
        # Create the visualization
        range_fig = plot_dynamic_range_comparison()
        st.plotly_chart(range_fig, use_container_width=True)
        
        st.markdown("""
        ### Impact on Deep Learning
        
        In deep learning, dynamic range limitations affect:
        
        - **Activation functions**: Values that grow very large or small
        - **Gradients**: Can span many orders of magnitude during training
        - **Model weights**: Often have varying magnitudes across layers
        
        DeepGEMM addresses this challenge by using fine-grained scaling factors to map values 
        from different parts of the input matrices into the optimal range for FP8 representation.
        """)
        
    elif visualization == "Precision Comparison":
        st.header("Precision Comparison of Floating Point Formats")
        
        st.markdown("""
        ### Representable Values
        
        The **precision** of a floating point format refers to how closely it can represent values. 
        This is determined primarily by the number of mantissa bits.
        
        The chart below shows representable values between 1 and 2 for different floating point formats. 
        Each dot represents a value that can be exactly represented by the format:
        
        - **FP32**: 2^23 = 8,388,608 representable values between 1 and 2
        - **FP16**: 2^10 = 1,024 representable values between 1 and 2
        - **BF16**: 2^7 = 128 representable values between 1 and 2
        - **FP8 (E4M3)**: 2^3 = 8 representable values between 1 and 2
        
        With only 8 representable values between 1 and 2, FP8 has very coarse precision, which
        can lead to significant quantization error without proper scaling.
        """)
        
        # Create the visualization
        precision_fig = plot_precision_comparison()
        st.plotly_chart(precision_fig, use_container_width=True)
        
        st.markdown("""
        ### The Need for Scaling
        
        FP8's limited precision means:
        
        - Values must be carefully scaled to minimize quantization error
        - Different regions of a matrix may need different scaling factors
        - A single global scaling factor is often insufficient
        
        DeepGEMM's fine-grained scaling approach addresses this by using separate scaling
        factors for different parts of the input matrices.
        """)
        
    elif visualization == "Scaling Visualization":
        st.header("Fine-Grained Scaling for FP8 Precision")
        
        st.markdown("""
        ### How Scaling Works
        
        Scaling is essential for using FP8 effectively:
        
        1. **Without Scaling**: Direct conversion to FP8 causes significant precision loss
        2. **With Scaling**: Values are first scaled to the optimal range for FP8, then converted
        3. **During Computation**: Scaling factors are applied to recover the original value ranges
        
        The visualization below demonstrates how scaling improves precision when using FP8:
        """)
        
        # Create the visualization
        scaling_fig = plot_scaling_comparison()
        st.plotly_chart(scaling_fig, use_container_width=True)
        
        st.markdown("""
        ### Key Insights
        
        - **Group-Based Scaling**: Values are divided into groups, with each group having its own scaling factor
        - **Target Range**: Values are scaled to maximize use of FP8's representable range (near 8.0)
        - **Error Reduction**: Proper scaling can reduce quantization error by orders of magnitude
        - **Dynamic Adjustment**: Scaling factors change based on the actual values in each group
        
        DeepGEMM takes this approach further with per-128-channel scaling to handle matrices with
        widely varying magnitudes across different regions.
        """)
        
    elif visualization == "Per-Channel Scaling":
        st.header("Per-128-Channel Scaling in DeepGEMM")
        
        st.markdown("""
        ### Per-128-Channel Scaling Approach
        
        DeepGEMM uses a sophisticated per-128-channel scaling approach:
        
        1. Each matrix is divided into blocks of 128 columns
        2. Each block gets its own scaling factor based on the maximum absolute value
        3. Scaling factors are stored alongside the FP8 matrix
        4. During computation, values are dynamically scaled back to their original range
        
        This approach preserves precision across matrices with diverse value distributions:
        """)
        
        # Create the visualization
        channel_fig = plot_channel_scaling()
        st.plotly_chart(channel_fig, use_container_width=True)
        
        st.markdown("""
        ### Implementation Details
        
        From DeepGEMM's code:
        
        ```python
        # For LHS (left-hand side) matrix
        assert lhs_scales.shape == (m, (k + 127) // 128)
        
        # For RHS (right-hand side) matrix
        assert rhs_scales.shape == ((n + 127) // 128, (k + 127) // 128)
        ```
        
        The scaling tensors store one FP32 value for each 128-element channel block:
        - For an m×k matrix: m × ⌈k/128⌉ scaling factors
        - 128-element granularity offers excellent precision with minimal overhead
        - Only adds ~0.8% memory overhead (1 FP32 value per 128 FP8 values)
        """)
        
    elif visualization == "Matrix Multiplication":
        st.header("Matrix Multiplication with Fine-Grained Scaling")
        
        st.markdown("""
        ### How DeepGEMM Applies Scaling in GEMM
        
        DeepGEMM's matrix multiplication combines fine-grained scaling with efficient GPU execution:
        
        1. Input matrices A and B are stored in FP8 with per-128-channel scaling factors
        2. During computation, values are scaled back to higher precision
        3. Matrix multiplication is performed in higher precision
        4. Result is produced in BF16 to maintain accuracy
        
        This achieves both high performance and high accuracy:
        """)
        
        # Create the visualization
        matmul_fig = plot_matmul_with_scaling()
        st.plotly_chart(matmul_fig, use_container_width=True)
        
        st.markdown("""
        ### CUDA Kernel Implementation
        
        In the DeepGEMM CUDA kernel, scaling is applied during computation:
        
        ```cpp
        // From fp8_gemm.cuh:
        // Read A scales
        auto scale_a_0 = ld_shared(smem_scales_a[s] + r_0);
        auto scale_a_1 = ld_shared(smem_scales_a[s] + r_1);
        
        // Read B scales
        float scale_b_0 = ld_shared(smem_scales_b + k_iter * kNumStages + s);
        
        // Promote with scales (combine scaling factors)
        float scale_0_0 = scale_a_0 * scale_b_0;
        float scale_1_0 = scale_a_1 * scale_b_0;
        
        // Apply scaling to accumulator
        final_accum[i * 4 + 0] += scale_0_0 * accum[i * 4 + 0];
        ```
        
        This fine-grained scaling approach, combined with optimized tensor core operations,
        allows DeepGEMM to achieve excellent performance while maintaining high numerical accuracy.
        """)

if __name__ == "__main__":
    main()
