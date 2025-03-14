import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import io
from PIL import Image
import base64

st.set_page_config(
    page_title="NVIDIA Hopper Architecture for DeepGEMM",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

def draw_hopper_architecture():
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Define colors directly without dictionary to avoid key errors
    sm_color = '#4472C4'          # Streaming Multiprocessor
    l2_color = '#70AD47'          # L2 Cache
    tensor_core_color = '#ED7D31' # Tensor Core
    tma_color = '#FFC000'         # Tensor Memory Accelerator
    shared_mem_color = '#5B9BD5'  # Shared Memory
    hbm_color = '#A5A5A5'         # HBM Memory
    warp_color = '#9E480E'        # Warp Scheduler
    dispatch_color = '#C00000'    # Dispatch Units
    reg_file_color = '#8064A2'    # Register File
    bg_color = '#F2F2F2'          # Background
    border_color = '#404040'      # Borders
    
    # Set background color
    ax.set_facecolor(bg_color)
    
    # Draw the overall GPU diagram
    
    # Main GPU outline
    gpu_outline = patches.Rectangle((0, 0), 15, 11, linewidth=2, edgecolor=border_color, facecolor='white')
    ax.add_patch(gpu_outline)
    
    # Title
    ax.text(7.5, 10.5, "NVIDIA H100 Hopper GPU Architecture", fontsize=20, fontweight='bold', ha='center')
    
    # Draw GPU Components
    
    # GPC (Graphics Processing Clusters) - simplified as 7 rows of SMs
    ax.text(7.5, 9.7, "132 Streaming Multiprocessors (SMs)", fontsize=16, ha='center')
    
    # Draw multiple SMs (simplified representation)
    sm_width, sm_height = 1.8, 0.95
    sm_rows, sm_cols = 6, 6  # 6x6 grid (showing 36 of 132 SMs)
    
    for row in range(sm_rows):
        for col in range(sm_cols):
            x = 1 + col * (sm_width + 0.2)
            y = 8 - row * (sm_height + 0.15)
            
            # SM outline
            sm = patches.Rectangle((x, y), sm_width, sm_height, linewidth=1, 
                                   edgecolor=border_color, facecolor=sm_color, alpha=0.8)
            ax.add_patch(sm)
            
            # Draw internal SM components in the first SM only (top-left)
            if row == 0 and col == 0:
                # Tensor Cores (4 per SM)
                for tc_idx in range(4):
                    tc_x = x + 0.1 + (tc_idx % 2) * 0.8
                    tc_y = y + 0.1 + (tc_idx // 2) * 0.35
                    tc = patches.Rectangle((tc_x, tc_y), 0.7, 0.3, linewidth=1, 
                                          edgecolor=border_color, facecolor=tensor_core_color)
                    ax.add_patch(tc)
                    ax.text(tc_x + 0.35, tc_y + 0.15, "TC", ha='center', va='center', fontsize=8, color='white')
                
                # TMA
                tma = patches.Rectangle((x + 0.1, y + 0.55), 0.4, 0.3, linewidth=1,
                                       edgecolor=border_color, facecolor=tma_color)
                ax.add_patch(tma)
                ax.text(x + 0.3, y + 0.7, "TMA", ha='center', va='center', fontsize=8, color='black')
                
                # Shared Memory
                shm = patches.Rectangle((x + 1.1, y + 0.55), 0.6, 0.3, linewidth=1,
                                       edgecolor=border_color, facecolor=shared_mem_color)
                ax.add_patch(shm)
                ax.text(x + 1.4, y + 0.7, "L1/Shared", ha='center', va='center', fontsize=7, color='white')
                
                # SM label
                sm_label = "Detailed SM"
            else:
                # SM label with number
                sm_idx = row * sm_cols + col + 1
                sm_label = f"SM {sm_idx}"
            
            # Add SM label
            ax.text(x + sm_width/2, y + sm_height/2, sm_label, 
                    ha='center', va='center', fontsize=9, color='white')
    
    # Add ellipsis to indicate more SMs
    ax.text(14, 5, "...", fontsize=24, ha='center', va='center', color=border_color)
    
    # L2 Cache
    l2_height = 0.8
    l2_cache = patches.Rectangle((1, 2), 13, l2_height, linewidth=1,
                               edgecolor=border_color, facecolor=l2_color)
    ax.add_patch(l2_cache)
    ax.text(7.5, 2 + l2_height/2, "L2 Cache (50 MB)", ha='center', va='center', fontsize=14, color='white')
    
    # HBM3 Memory Controller
    hbm_height = 0.8
    hbm = patches.Rectangle((1, 1), 13, hbm_height, linewidth=1,
                           edgecolor=border_color, facecolor=hbm_color)
    ax.add_patch(hbm)
    ax.text(7.5, 1 + hbm_height/2, "HBM3 Memory (80GB, ~3TB/s bandwidth)", 
            ha='center', va='center', fontsize=14, color='black')
    
    # Detailed SM Component Zoomed View
    # Draw a zoom box around the first SM
    zoom_sm_x, zoom_sm_y = 1, 8
    zoom_rect = patches.Rectangle((zoom_sm_x-0.05, zoom_sm_y-0.05), sm_width+0.1, sm_height+0.1, 
                                 fill=False, linewidth=1.5, edgecolor='red', linestyle='--')
    ax.add_patch(zoom_rect)
    
    # Connect to the expanded view
    ax.annotate('', xy=(zoom_sm_x + sm_width/2, zoom_sm_y - 0.05), 
                xytext=(11, 7), arrowprops=dict(arrowstyle='->',
                                                connectionstyle="arc3,rad=-0.3",
                                                color='red', linewidth=1.5))
    
    # Draw expanded SM
    expanded_sm_width, expanded_sm_height = 3, 2.5
    expanded_sm_x, expanded_sm_y = 10, 4.3
    
    expanded_sm = patches.Rectangle((expanded_sm_x, expanded_sm_y), expanded_sm_width, expanded_sm_height,
                                   linewidth=1.5, edgecolor=border_color, facecolor=sm_color, alpha=0.9)
    ax.add_patch(expanded_sm)
    
    # SM Title
    ax.text(expanded_sm_x + expanded_sm_width/2, expanded_sm_y + expanded_sm_height - 0.2, 
            "Hopper SM Architecture", fontsize=12, fontweight='bold', ha='center', color='white')
    
    # Tensor Cores (4th Gen)
    tensor_width, tensor_height = 0.6, 0.4
    for i in range(4):
        x_pos = expanded_sm_x + 0.3 + (i % 2) * 1.5
        y_pos = expanded_sm_y + 0.5 + (i // 2) * 1.0
        tensor = patches.Rectangle((x_pos, y_pos), tensor_width, tensor_height,
                                  linewidth=1, edgecolor=border_color, facecolor=tensor_core_color)
        ax.add_patch(tensor)
        ax.text(x_pos + tensor_width/2, y_pos + tensor_height/2, "4th Gen\nTensor Core", 
                ha='center', va='center', fontsize=8, color='white')
    
    # TMA
    tma_x, tma_y = expanded_sm_x + 0.3, expanded_sm_y + 2.0
    tma = patches.Rectangle((tma_x, tma_y), 0.9, 0.3, linewidth=1,
                           edgecolor=border_color, facecolor=tma_color)
    ax.add_patch(tma)
    ax.text(tma_x + 0.45, tma_y + 0.15, "Tensor Memory\nAccelerator (TMA)", 
            ha='center', va='center', fontsize=8, color='black')
    
    # Shared Memory
    shm_x, shm_y = expanded_sm_x + 1.5, expanded_sm_y + 2.0
    shm = patches.Rectangle((shm_x, shm_y), 1.2, 0.3, linewidth=1,
                           edgecolor=border_color, facecolor=shared_mem_color)
    ax.add_patch(shm)
    ax.text(shm_x + 0.6, shm_y + 0.15, "L1/Shared Memory\n(232KB)", 
            ha='center', va='center', fontsize=8, color='white')
    
    # Register File
    reg_x, reg_y = expanded_sm_x + 0.3, expanded_sm_y + 0.2
    reg = patches.Rectangle((reg_x, reg_y), 2.4, 0.25, linewidth=1,
                           edgecolor=border_color, facecolor=reg_file_color)
    ax.add_patch(reg)
    ax.text(reg_x + 1.2, reg_y + 0.125, "Register File (256KB)", 
            ha='center', va='center', fontsize=8, color='white')
    
    # Add DeepGEMM Key Components Callout
    # Draw a box highlighting DeepGEMM key components
    highlight_components = [
        {"name": "TMA for Efficient\nMemory Transfer", "x": 12, "y": 9, "color": tma_color},
        {"name": "Tensor Cores for\nFP8 Matrix Multiply", "x": 12, "y": 8, "color": tensor_core_color},
        {"name": "Shared Memory for\nData Tiling", "x": 12, "y": 7, "color": shared_mem_color},
        {"name": "Warp Specialization", "x": 12, "y": 6, "color": warp_color}
    ]
    
    for i, comp in enumerate(highlight_components):
        # Create colored circle
        circle = plt.Circle((comp["x"] - 0.5, comp["y"]), 0.15, color=comp["color"], ec=border_color)
        ax.add_patch(circle)
        
        # Add component name
        ax.text(comp["x"], comp["y"], comp["name"], fontsize=10, va='center')
    
    # Add title for DeepGEMM components
    ax.text(12, 9.7, "DeepGEMM Key Components", fontsize=12, fontweight='bold')
    
    # Add arrows connecting components to the SM
    arrow_targets = [
        (expanded_sm_x + 0.75, expanded_sm_y + 2.0),  # TMA
        (expanded_sm_x + 1.0, expanded_sm_y + 1.2),   # Tensor Core
        (expanded_sm_x + 2.1, expanded_sm_y + 2.0),   # Shared Mem
        (expanded_sm_x + 1.5, expanded_sm_y + 0.8)    # Warp
    ]
    
    for i, comp in enumerate(highlight_components):
        # Draw arrow
        target_x, target_y = arrow_targets[i]
        ax.annotate('', xy=(target_x, target_y), 
                    xytext=(comp["x"] - 0.5, comp["y"]), 
                    arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=0.2",
                                   color=border_color, linewidth=1))
    
    # Set axis limits and remove ticks
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(0.5, 11.5)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add DeepGEMM title and description
    ax.text(7.5, 0.6, "DeepGEMM optimizes FP8 matrix multiplication on Hopper through efficient use of Tensor Cores, TMA, and specialized warp scheduling",
            fontsize=11, ha='center', style='italic')
    
    ax.text(7.5, 0.3, "© 2025 - H100 GPU with 132 SMs, 4th Gen Tensor Cores, and TMA support",
            fontsize=9, ha='center', color='gray')
    
    plt.tight_layout(pad=0.5)
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img


def draw_tensor_core_detail():
    # Create tensor core detail figure
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Define colors
    sm_color = '#4472C4'          # Streaming Multiprocessor
    tensor_core_color = '#ED7D31' # Tensor Core
    tma_color = '#FFC000'         # Tensor Memory Accelerator
    bg_color = '#F2F2F2'          # Background
    border_color = '#404040'      # Borders
    
    # Background
    ax2.set_facecolor(bg_color)
    
    # Title
    ax2.text(5, 5.5, "Hopper 4th Gen Tensor Core", fontsize=18, fontweight='bold', ha='center')
    
    # Tensor Core outline
    tc_outline = patches.Rectangle((1, 1), 8, 4, linewidth=2, 
                                  edgecolor=border_color, facecolor=tensor_core_color, alpha=0.9)
    ax2.add_patch(tc_outline)
    
    # FP8 Matrix Multiply Units
    for i in range(4):
        x = 1.5 + (i % 2) * 3.5
        y = 1.5 + (i // 2) * 2
        
        mma_unit = patches.Rectangle((x, y), 3, 1.5, linewidth=1, 
                                   edgecolor=border_color, facecolor='white', alpha=0.9)
        ax2.add_patch(mma_unit)
        
        ax2.text(x + 1.5, y + 0.75, f"FP8 MMA Unit", ha='center', va='center', fontsize=10)
        
        # Matrix dimensions
        if i == 0:
            ax2.annotate("16×16×16", xy=(x + 1.5, y - 0.2), ha='center', fontsize=9)
    
    # Tensor Core Info
    info_text = (
        "• Performs FP8 matrix multiplication with BF16 accumulation\n"
        "• Supports E4M3 and E5M2 FP8 formats\n"
        "• 4th Gen adds support for 8-bit precision\n"
        "• ~1000 TFLOPS for FP8 operations\n"
        "• DeepGEMM uses fine-grained per-128-channel scaling"
    )
    
    ax2.text(5, 0.5, info_text, ha='center', va='center', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", fc='white', ec=border_color, alpha=0.9))
    
    # Matrix multiplication visualization
    matrix_x, matrix_y = 10, 3
    matrix_size = 0.8
    
    # Matrix A (FP8)
    ax2.add_patch(patches.Rectangle((matrix_x, matrix_y), matrix_size, matrix_size, 
                                   facecolor=sm_color, alpha=0.8, ec=border_color))
    ax2.text(matrix_x + matrix_size/2, matrix_y + matrix_size/2, "A\nFP8", 
             ha='center', va='center', fontsize=10, color='white')
    
    # Matrix B (FP8)
    ax2.add_patch(patches.Rectangle((matrix_x + matrix_size*1.5, matrix_y), matrix_size, matrix_size, 
                                   facecolor=sm_color, alpha=0.8, ec=border_color))
    ax2.text(matrix_x + matrix_size*1.5 + matrix_size/2, matrix_y + matrix_size/2, "B\nFP8", 
             ha='center', va='center', fontsize=10, color='white')
    
    # Multiply symbol
    ax2.text(matrix_x + matrix_size*1.2, matrix_y + matrix_size/2, "×", 
             ha='center', va='center', fontsize=14)
    
    # Equals symbol
    ax2.text(matrix_x + matrix_size*2.2, matrix_y + matrix_size/2, "=", 
             ha='center', va='center', fontsize=14)
    
    # Result matrix (BF16)
    ax2.add_patch(patches.Rectangle((matrix_x + matrix_size*2.7, matrix_y), matrix_size, matrix_size, 
                                   facecolor=tensor_core_color, alpha=0.8, ec=border_color))
    ax2.text(matrix_x + matrix_size*2.7 + matrix_size/2, matrix_y + matrix_size/2, "C\nBF16", 
             ha='center', va='center', fontsize=10, color='white')
    
    # Scale factors
    scale_width = 0.4
    scale_height = 0.3
    
    # Scale A
    ax2.add_patch(patches.Rectangle((matrix_x, matrix_y - 0.5), scale_width, scale_height, 
                                   facecolor=tma_color, alpha=0.9, ec=border_color))
    ax2.text(matrix_x + scale_width/2, matrix_y - 0.5 + scale_height/2, "Scale A", 
             ha='center', va='center', fontsize=8)
    
    # Scale B
    ax2.add_patch(patches.Rectangle((matrix_x + matrix_size*1.5, matrix_y - 0.5), scale_width, scale_height, 
                                   facecolor=tma_color, alpha=0.9, ec=border_color))
    ax2.text(matrix_x + matrix_size*1.5 + scale_width/2, matrix_y - 0.5 + scale_height/2, "Scale B", 
             ha='center', va='center', fontsize=8)
    
    # Connect scales to matrices
    ax2.plot([matrix_x + scale_width/2, matrix_x + matrix_size/2], 
             [matrix_y - 0.5 + scale_height, matrix_y], 'k--', linewidth=0.8)
    
    ax2.plot([matrix_x + matrix_size*1.5 + scale_width/2, matrix_x + matrix_size*1.5 + matrix_size/2], 
             [matrix_y - 0.5 + scale_height, matrix_y], 'k--', linewidth=0.8)
    
    # Set axis limits and remove ticks
    ax2.set_xlim(0, 14)
    ax2.set_ylim(0, 6)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img


def draw_fp8_format():
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Background
    ax.set_facecolor('#F2F2F2')
    
    # Formats to display
    formats = [
        {"name": "FP32", "bits": 32, "sign": 1, "exp": 8, "mantissa": 23, "color": "blue"},
        {"name": "FP16", "bits": 16, "sign": 1, "exp": 5, "mantissa": 10, "color": "green"},
        {"name": "BF16", "bits": 16, "sign": 1, "exp": 8, "mantissa": 7, "color": "orange"},
        {"name": "FP8 (E4M3)", "bits": 8, "sign": 1, "exp": 4, "mantissa": 3, "color": "red"}
    ]
    
    # Draw format diagrams
    y_spacing = 1
    y_position = 4
    
    for fmt in formats:
        # Format title
        ax.text(0.5, y_position, fmt["name"], fontsize=12, fontweight='bold')
        
        # Total size indicator
        total_width = 10
        ax.text(1, y_position - 0.3, f"Total: {fmt['bits']} bits", fontsize=10)
        
        # Sign bit
        sign_width = fmt["sign"] / fmt["bits"] * total_width
        sign = patches.Rectangle((2, y_position - 0.5), sign_width, 0.8, 
                                 facecolor='lightblue', edgecolor='black')
        ax.add_patch(sign)
        ax.text(2 + sign_width/2, y_position - 0.1, "S", ha='center', va='center', fontsize=10)
        
        # Exponent bits
        exp_width = fmt["exp"] / fmt["bits"] * total_width
        exp = patches.Rectangle((2 + sign_width, y_position - 0.5), exp_width, 0.8, 
                               facecolor='lightgreen', edgecolor='black')
        ax.add_patch(exp)
        ax.text(2 + sign_width + exp_width/2, y_position - 0.1, "Exponent", ha='center', va='center', fontsize=10)
        
        # Mantissa bits
        mantissa_width = fmt["mantissa"] / fmt["bits"] * total_width
        mantissa = patches.Rectangle((2 + sign_width + exp_width, y_position - 0.5), mantissa_width, 0.8, 
                                    facecolor='salmon', edgecolor='black')
        ax.add_patch(mantissa)
        ax.text(2 + sign_width + exp_width + mantissa_width/2, y_position - 0.1, "Mantissa", 
               ha='center', va='center', fontsize=10)
        
        # Bit counts
        ax.text(2 + sign_width/2, y_position - 0.7, f"{fmt['sign']}", ha='center', va='center', fontsize=8)
        ax.text(2 + sign_width + exp_width/2, y_position - 0.7, f"{fmt['exp']}", ha='center', va='center', fontsize=8)
        ax.text(2 + sign_width + exp_width + mantissa_width/2, y_position - 0.7, 
               f"{fmt['mantissa']}", ha='center', va='center', fontsize=8)
        
        y_position -= y_spacing
    
    # Add title
    ax.text(6, 5, "Floating Point Format Comparison", fontsize=16, fontweight='bold', ha='center')
    
    # Add FP8 characteristics
    fp8_info = """
    FP8 (E4M3) Characteristics:
    • 1 sign bit
    • 4 exponent bits (bias of 7)
    • 3 mantissa bits
    • Dynamic range: ~0.003 to ~240
    • Requires per-128-channel scaling for numerical stability
    """
    
    ax.text(9, 2.5, fp8_info, fontsize=10, ha='left',
           bbox=dict(boxstyle="round,pad=0.5", fc='white', ec='red', alpha=0.9))
    
    # Set axis limits and remove ticks
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.5)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img


def fp8_scaling_diagram():
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Background
    ax.set_facecolor('#F2F2F2')
    
    # Title
    ax.text(6, 5.7, "Per-128-Channel FP8 Scaling in DeepGEMM", fontsize=16, fontweight='bold', ha='center')
    
    # Draw matrix representation
    matrix_width, matrix_height = 8, 4
    matrix_x, matrix_y = 2, 1
    
    # Matrix outline
    matrix = patches.Rectangle((matrix_x, matrix_y), matrix_width, matrix_height, 
                              linewidth=2, edgecolor='black', facecolor='white')
    ax.add_patch(matrix)
    
    # Draw grid lines to show the 128-column blocks
    block_width = 2  # Each block is 2 units wide (representing 128 columns)
    
    # Vertical grid lines for column blocks
    for i in range(1, 4):
        ax.axvline(x=matrix_x + i * block_width, color='black', linestyle='--', linewidth=1)
    
    # Label the blocks
    for i in range(4):
        ax.text(matrix_x + i * block_width + block_width/2, matrix_y + matrix_height + 0.2, 
               f"Block {i+1}\n(128 columns)", ha='center', va='center', fontsize=9)
    
    # Add text to show these are FP8 values
    ax.text(matrix_x + matrix_width/2, matrix_y + matrix_height/2, 
           "FP8 Matrix Values\n(E4M3 Format)", ha='center', va='center', fontsize=12)
    
    # Draw scaling factors
    scale_width, scale_height = 2, 0.6
    scale_x, scale_y = 2, 0.1
    
    # Draw scale boxes
    for i in range(4):
        scale = patches.Rectangle((scale_x + i * block_width, scale_y), block_width, scale_height, 
                                linewidth=1, edgecolor='black', facecolor='gold', alpha=0.8)
        ax.add_patch(scale)
        ax.text(scale_x + i * block_width + block_width/2, scale_y + scale_height/2, 
               f"Scale Factor\nBlock {i+1}", ha='center', va='center', fontsize=9)
    
    # Connect scales to matrix blocks with arrows
    for i in range(4):
        center_x = scale_x + i * block_width + block_width/2
        ax.arrow(center_x, scale_y + scale_height, 0, matrix_y - (scale_y + scale_height), 
                head_width=0.1, head_length=0.1, fc='black', ec='black', alpha=0.7)
    
    # Add explanatory text
    explanation = """
    DeepGEMM's Fine-Grained Scaling Approach:
    
    1. Each 128-column block gets its own scaling factor
    2. Values are scaled to optimal FP8 range before conversion
    3. Scaling factors are stored alongside FP8 values
    4. During computation, values are dynamically rescaled
    5. This preserves precision across diverse value ranges
    6. Only adds ~0.8% memory overhead
    """
    
    ax.text(11, 3, explanation, fontsize=10, ha='left',
           bbox=dict(boxstyle="round,pad=0.5", fc='white', ec='black', alpha=0.9))
    
    # Show matrix calculation with scales
    calc_text = """
    Matrix multiplication with scaling:
    
    [A_fp8] × [B_fp8] → [C_bf16]
        ↑        ↑
        |        |
    [Scale_A] [Scale_B]
    
    Each value is rescaled during calculation:
    C[i,j] = ∑(A[i,k] × Scale_A[i,⌊k/128⌉] × B[j,k] × Scale_B[j,⌊k/128⌉])
    """
    
    ax.text(11, 1.5, calc_text, fontsize=10, ha='left', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", fc='white', ec='black', alpha=0.9))
    
    # Set axis limits and remove ticks
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img


# Sidebar
st.sidebar.title("NVIDIA Hopper for DeepGEMM")
st.sidebar.image("https://developer-blogs.nvidia.com/wp-content/uploads/2022/03/H100-chip.jpg", width=300)

# Navigation
view_option = st.sidebar.radio(
    "Select View",
    ["Hopper Architecture Overview", "Tensor Core Details", "FP8 Format", "Per-128-Channel Scaling"]
)

# Information sidebar
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info(
    """
    This interactive visualization explains the NVIDIA Hopper GPU architecture 
    features that enable DeepGEMM's high-performance FP8 matrix multiplication.
    
    DeepGEMM leverages Hopper's 4th generation Tensor Cores and Tensor Memory 
    Accelerator (TMA) for efficient FP8 computation with per-128-channel scaling.
    """
)

# Key metrics sidebar
st.sidebar.markdown("---")
st.sidebar.header("Hopper H100 Key Specs")
st.sidebar.markdown(
    """
    - **SMs**: 132 Streaming Multiprocessors
    - **Tensor Cores**: 528 (4 per SM)
    - **FP8 Perf**: ~1000 TFLOPS
    - **Memory**: 80GB HBM3
    - **Bandwidth**: ~3 TB/s
    - **L2 Cache**: 50MB
    - **Shared Memory**: 232KB per SM
    """
)

# Main content based on selected view
if view_option == "Hopper Architecture Overview":
    st.header("NVIDIA Hopper GPU Architecture")
    
    st.markdown(
        """
        The NVIDIA Hopper architecture (H100 GPU) provides the hardware foundation for DeepGEMM's 
        high-performance FP8 matrix multiplication. Key architectural features include:
        
        - **4th Generation Tensor Cores** with native FP8 support
        - **Tensor Memory Accelerator (TMA)** for efficient data movement
        - **Large shared memory** (232KB per SM) for data tiling
        - **High-bandwidth memory** (~3TB/s) for fast data access
        """
    )
    
    # Display the architecture diagram
    architecture_img = draw_hopper_architecture()
    st.image(architecture_img, use_column_width=True)
    
    # DeepGEMM integration explanation
    st.subheader("How DeepGEMM Leverages Hopper's Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            **Tensor Cores**
            - Uses 4th Gen Tensor Cores for FP8 matrix multiplication
            - Optimizes instruction scheduling for maximum throughput
            - Implements SASS-level optimizations for better performance
            """
        )
        
        st.markdown(
            """
            **Tensor Memory Accelerator (TMA)**
            - Uses TMA for efficient global-to-shared memory transfers
            - Leverages TMA's tensor data layout transformations
            - Employs TMA multicast for large matrices
            """
        )
    
    with col2:
        st.markdown(
            """
            **Shared Memory**
            - Efficiently tiles matrices into shared memory blocks
            - Manages per-128-channel scaling factors in shared memory
            - Implements software pipelining for memory access
            """
        )
        
        st.markdown(
            """
            **Block Scheduling**
            - Uses persistent thread blocks for efficient SM utilization
            - Implements specialized warp scheduling for different tasks
            - Optimizes block dimensions for different matrix shapes
            """
        )

elif view_option == "Tensor Core Details":
    st.header("Hopper 4th Generation Tensor Cores")
    
    st.markdown(
        """
        Hopper's 4th Generation Tensor Cores are specialized processing units designed 
        for accelerating matrix multiplication operations. They are particularly 
        optimized for 8-bit computation, making them ideal for DeepGEMM's FP8 workloads.
        """
    )
    
    # Display tensor core details
    tensor_core_img = draw_tensor_core_detail()
    st.image(tensor_core_img, use_column_width=True)
    
    # Technical details in expandable sections
    with st.expander("Tensor Core Technical Details"):
        st.markdown(
            """
            **FP8 Matrix-Matrix Operation**
            ```
            D = A × B
            ```
            Where:
            - A and B are FP8 (E4M3) matrices
            - D is a BF16 accumulation matrix
            - Operation is performed in a single instruction
            
            **Matrix Dimensions**
            - Tensor Core operates on 16×16×16 matrix fragments
            - Multiple tensor cores can be combined for larger matrices
            - DeepGEMM tiles large matrices into optimal chunks
            
            **FP8 Formats Supported**
            - E4M3 (4 exponent bits, 3 mantissa bits)
            - E5M2 (5 exponent bits, 2 mantissa bits)
            """
        )
    
    with st.expander("DeepGEMM Tensor Core Utilization"):
        st.markdown(
            """
            **Optimizations**
            - Instruction scheduling to maximize tensor core utilization
            - Block size tuning to match tensor core dimensions
            - Software pipelining to overlap memory and compute operations
            - SASS-level FFMA interleaving for better performance
            
            **Precision Management**
            - Scaling matrices to optimal range for FP8 representation
            - Implementing per-128-channel scaling to maintain accuracy
            - Managing scale factors alongside matrix computation
            """
        )

elif view_option == "FP8 Format":
    st.header("FP8 (E4M3) Floating Point Format")
    
    st.markdown(
        """
        FP8 (8-bit floating point) is a reduced-precision data format that offers significant 
        performance benefits for deep learning workloads. DeepGEMM uses the E4M3 variant, 
        which provides a good balance between range and precision.
        """
    )
    
    # Display FP8 format diagram
    fp8_img = draw_fp8_format()
    st.image(fp8_img, use_column_width=True)
    
    # Comparison table
    st.subheader("Format Comparison")
    
    format_data = {
        "Format": ["FP32", "FP16", "BF16", "FP8 (E4M3)"],
        "Size (bytes)": [4, 2, 2, 1],
        "Dynamic Range": ["~1.4×10⁻⁴⁵ to ~3.4×10³⁸", "~6.0×10⁻⁸ to ~6.5×10⁴", "~1.4×10⁻⁴⁵ to ~3.4×10³⁸", "~0.003 to ~240"],
        "Precision": ["2⁻²³ ≈ 1.2×10⁻⁷", "2⁻¹⁰ ≈ 9.8×10⁻⁴", "2⁻⁷ ≈ 7.8×10⁻³", "2⁻³ ≈ 0.125"],
        "Memory Savings vs FP32": ["0%", "50%", "50%", "75%"]
    }
    
    st.table(format_data)
    
    # Expandable sections for more details
    with st.expander("FP8 (E4M3) Calculation Examples"):
        st.markdown(
            """
            **Largest Normal Value**
            - Exponent: 1110 binary (14 decimal)
            - Bias-adjusted: 14 - 7 = 7
            - Mantissa: 111 binary = 0.875 decimal
            - Value: (1 + 0.875) × 2⁷ = 1.875 × 128 = 240
            
            **Smallest Normal Value**
            - Exponent: 0001 binary (1 decimal)
            - Bias-adjusted: 1 - 7 = -6
            - Mantissa: 000 binary = 0.0 decimal
            - Value: (1 + 0.0) × 2⁻⁶ = 0.015625
            
            **Representable Values Near 1.0**
            - 1.0 = 1.000 binary × 2⁰
            - 1.125 = 1.001 binary × 2⁰
            - 1.25 = 1.010 binary × 2⁰
            - 1.375 = 1.011 binary × 2⁰
            - 1.5 = 1.100 binary × 2⁰
            - 1.625 = 1.101 binary × 2⁰
            - 1.75 = 1.110 binary × 2⁰
            - 1.875 = 1.111 binary × 2⁰
            """
        )
    
    with st.expander("Why Scaling is Necessary for FP8"):
        st.markdown(
            """
            FP8's limited precision and dynamic range create challenges:
            
            1. **Limited Dynamic Range**: FP8 can only represent values from ~0.003 to ~240, 
               while neural network values can span many more orders of magnitude.
            
            2. **Coarse Precision**: With only 3 mantissa bits, FP8 has large gaps between 
               representable values, causing significant quantization error.
            
            3. **Varying Magnitude**: Different regions of a matrix often have vastly different 
               value distributions, making a single global scale factor insufficient.
            
            DeepGEMM addresses these challenges through its per-128-channel scaling approach, 
            which adapts scaling factors to local value distributions within the matrix.
            """
        )

elif view_option == "Per-128-Channel Scaling":
    st.header("DeepGEMM's Per-128-Channel Scaling")
    
    st.markdown(
        """
        DeepGEMM uses a sophisticated per-128-channel scaling approach to maintain 
        numerical accuracy while leveraging FP8's performance benefits. This technique 
        divides matrices into 128-column blocks, each with its own scaling factor.
        """
    )
    
    # Display scaling diagram
    scaling_img = fp8_scaling_diagram()
    st.image(scaling_img, use_column_width=True)
    
    # Code implementation details
    st.subheader("Implementation in DeepGEMM")
    
    with st.expander("Python Implementation"):
        st.code("""
# From test_core.py - Per-token cast to FP8 with scaling
def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    
    # Reshape to group by 128 columns
    x_view = x.view(m, -1, 128)
    
    # Calculate max absolute value per 128-column block
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    
    # Scale to optimal range and convert to FP8
    fp8_values = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    
    # Store scaling factors for later use
    scale_factors = (x_amax / 448.0).view(m, -1)
    
    return fp8_values.view(m, n), scale_factors
""", language="python")
    
    with st.expander("CUDA Kernel Implementation"):
        st.code("""
// From fp8_gemm.cuh - Applying scaling during computation
// Read A scales and B scales
auto scale_a_0 = ld_shared(smem_scales_a[s] + r_0);
auto scale_a_1 = ld_shared(smem_scales_a[s] + r_1);
float scale_b_0 = ld_shared(smem_scales_b + k_iter * kNumStages + s);

// Combine scaling factors
float scale_0_0 = scale_a_0 * scale_b_0, scale_1_0 = scale_a_1 * scale_b_0;

// Apply scaling factors to accumulator values
final_accum[i * 4 + 0] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 0];
final_accum[i * 4 + 1] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 1];
final_accum[i * 4 + 2] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 2];
final_accum[i * 4 + 3] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 3];
""", language="cpp")
    
    # Benefits section
    st.subheader("Benefits of Per-128-Channel Scaling")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            **Precision Benefits**
            - Preserves numerical precision in regions with small values
            - Adapts to varying value distributions within matrices
            - Maintains accuracy comparable to higher-precision formats
            - Enables effective use of FP8's limited dynamic range
            """
        )
    
    with col2:
        st.markdown(
            """
            **Performance Benefits**
            - Minimal memory overhead (~0.8%)
            - Efficient implementation using tensor memory access
            - Scaling factors stored in shared memory for fast access
            - Scaling applied during computation for maximum efficiency
            """
        )

# Footer
st.markdown("---")
st.markdown("© 2025 DeepGEMM Visualization | NVIDIA Hopper Architecture | FP8 Matrix Multiplication")

if __name__ == "__main__":
    # This code will be executed when the script is run directly
    pass