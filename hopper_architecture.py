import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

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
    plt.savefig('hopper_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create tensor core detail figure
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
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
    plt.savefig('tensor_core_detail.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations created: hopper_architecture.png and tensor_core_detail.png")

if __name__ == "__main__":
    draw_hopper_architecture()
