import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import os
from pathlib import Path
import base64
from io import BytesIO


def create_directory(dir_path):
    """Create directory if it doesn't exist"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_image_as_base64(fig):
    """Convert a matplotlib figure to base64 string for HTML display"""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return img_str


def plot_block_scheduling():
    """Visualize the persistent block scheduling concept used in DeepGEMM"""
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.5])
    
    # First subplot: Block scheduling overview
    ax1 = fig.add_subplot(gs[0, :])
    
    # Create a representation of how blocks are scheduled across SMs
    num_sms = 8  # Example number of SMs
    num_blocks = 30  # Example number of blocks
    
    # Create data array (0 = inactive, 1-30 = block IDs)
    data = np.zeros((num_sms, 10))  # 10 time steps
    
    # Assign blocks to SMs over time to demonstrate scheduling
    block_id = 1
    for t in range(10):
        for sm in range(num_sms):
            # After blocks complete, schedule new ones
            if t > 0 and np.random.random() > 0.7:
                data[sm, t] = 0  # Block completed
            
            # Schedule a new block if SM is available
            if data[sm, t] == 0 and block_id <= num_blocks:
                data[sm, t] = block_id
                block_id += 1
            else:
                # Continue previous block
                data[sm, t] = data[sm, t-1] if t > 0 else 0
    
    # Plot the data
    cmap = plt.cm.viridis
    bounds = np.arange(0, num_blocks + 2) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    im = ax1.imshow(data, aspect='auto', cmap=cmap, norm=norm)
    ax1.set_yticks(np.arange(num_sms))
    ax1.set_yticklabels([f'SM {i}' for i in range(num_sms)])
    ax1.set_xticks(np.arange(10))
    ax1.set_xticklabels([f'T{i}' for i in range(10)])
    ax1.set_title('Persistent Block Scheduling Across SMs')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Streaming Multiprocessor (SM)')
    
    # Legend for active blocks
    cb = plt.colorbar(im, ax=ax1, ticks=np.arange(0, num_blocks + 1, 5))
    cb.set_label('Block ID (0 = Inactive)')
    
    # Second subplot: TMA workflow
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Create a visual representation of TMA (Tensor Memory Accelerator)
    # Define the components
    components = [
        {'name': 'Global Memory', 'color': 'lightblue', 'position': (0.1, 0.7, 0.8, 0.2)},
        {'name': 'TMA', 'color': 'orange', 'position': (0.3, 0.4, 0.4, 0.15)},
        {'name': 'Shared Memory', 'color': 'lightgreen', 'position': (0.1, 0.1, 0.8, 0.2)}
    ]
    
    # Add the components to the plot
    for comp in components:
        rect = patches.Rectangle(
            (comp['position'][0], comp['position'][1]),
            comp['position'][2], comp['position'][3],
            linewidth=1, edgecolor='black', facecolor=comp['color'], alpha=0.8
        )
        ax2.add_patch(rect)
        ax2.text(
            comp['position'][0] + comp['position'][2]/2,
            comp['position'][1] + comp['position'][2]/4,
            comp['name'],
            ha='center', va='center'
        )
    
    # Add arrows to show data flow
    ax2.arrow(0.5, 0.7, 0, -0.15, head_width=0.05, head_length=0.05, fc='black', ec='black')
    ax2.arrow(0.5, 0.4, 0, -0.1, head_width=0.05, head_length=0.05, fc='black', ec='black')
    
    # Add annotations
    ax2.text(0.6, 0.6, '1. Global to Shared\nMemory Transfer', ha='left', va='center')
    ax2.text(0.6, 0.35, '2. Tensor Layout\nTransformation', ha='left', va='center')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('Tensor Memory Accelerator (TMA) Workflow')
    ax2.axis('off')
    
    # Third subplot: Memory Layout and Access Pattern
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Create a grid representing memory layout
    memory_grid = np.zeros((8, 8))
    
    # Create a pattern to show memory access
    for i in range(8):
        memory_grid[i, :] = i
    
    im3 = ax3.imshow(memory_grid, cmap='Blues')
    
    # Add annotations for memory layout
    for i in range(8):
        for j in range(8):
            ax3.text(j, i, f'({i},{j})', ha='center', va='center', color='black' if memory_grid[i,j] < 4 else 'white')
    
    # Add arrows to show memory access pattern
    ax3.arrow(1, 8.5, 0, -1, head_width=0.2, head_length=0.2, fc='red', ec='red')
    ax3.arrow(3, 8.5, 0, -1, head_width=0.2, head_length=0.2, fc='red', ec='red')
    ax3.arrow(5, 8.5, 0, -1, head_width=0.2, head_length=0.2, fc='red', ec='red')
    ax3.arrow(7, 8.5, 0, -1, head_width=0.2, head_length=0.2, fc='red', ec='red')
    
    ax3.set_title('Memory Access Pattern')
    ax3.set_xticks(np.arange(8))
    ax3.set_yticks(np.arange(8))
    ax3.set_xticklabels([f'Col {i}' for i in range(8)])
    ax3.set_yticklabels([f'Row {i}' for i in range(8)])
    
    # Fourth subplot: Software Pipeline
    ax4 = fig.add_subplot(gs[1, 2])
    
    # Create timeline showing software pipeline stages
    num_stages = 5
    num_iterations = 4
    
    # Create a matrix to represent pipeline stages
    # 0: not started, 1: prefetch, 2: compute, 3: completed
    pipeline = np.zeros((num_stages, num_iterations))
    
    # Fill the matrix to show staggered execution
    for s in range(num_stages):
        for i in range(num_iterations):
            if i >= s:
                if i == s:
                    pipeline[s, i] = 1  # Prefetch
                elif i == s + 1:
                    pipeline[s, i] = 2  # Compute
                else:
                    pipeline[s, i] = 3  # Completed
    
    # Define a custom colormap
    colors = ['white', 'lightblue', 'orange', 'lightgreen']
    cmap_pipeline = mcolors.ListedColormap(colors)
    
    # Plot the pipeline
    im4 = ax4.imshow(pipeline, cmap=cmap_pipeline, vmin=0, vmax=3)
    
    # Add text labels
    for s in range(num_stages):
        for i in range(num_iterations):
            if pipeline[s, i] == 1:
                ax4.text(i, s, 'Prefetch', ha='center', va='center')
            elif pipeline[s, i] == 2:
                ax4.text(i, s, 'Compute', ha='center', va='center')
            elif pipeline[s, i] == 3:
                ax4.text(i, s, 'Done', ha='center', va='center')
    
    # Add labels
    ax4.set_yticks(np.arange(num_stages))
    ax4.set_yticklabels([f'Stage {s}' for s in range(num_stages)])
    ax4.set_xticks(np.arange(num_iterations))
    ax4.set_xticklabels([f'Iter {i}' for i in range(num_iterations)])
    ax4.set_title('Software Pipelining')
    
    # Legend for pipeline states
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[0], markersize=15, label='Not Started'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[1], markersize=15, label='Prefetch'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[2], markersize=15, label='Compute'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[3], markersize=15, label='Completed')
    ]
    ax4.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def plot_warp_specialization():
    """Visualize the warp specialization concept used in DeepGEMM"""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # First subplot: Traditional warp approach
    # Create a grid representing thread blocks with warps
    traditional = np.zeros((8, 4))
    for i in range(8):
        traditional[i, :] = i % 4
    
    im1 = ax1.imshow(traditional, cmap='tab10', vmin=0, vmax=9)
    
    # Add text to indicate all warps do same work
    for i in range(8):
        for j in range(4):
            ax1.text(j, i, 'Compute', ha='center', va='center', color='white')
    
    ax1.set_title('Traditional Approach\n(All Warps Perform Same Tasks)')
    ax1.set_xticks(np.arange(4))
    ax1.set_xticklabels([f'Warp {i}' for i in range(4)])
    ax1.set_yticks(np.arange(8))
    ax1.set_yticklabels([f'Time {i}' for i in range(8)])
    
    # Second subplot: Warp specialization
    specialized = np.ones((8, 4)) * np.arange(4)
    
    im2 = ax2.imshow(specialized, cmap='tab10', vmin=0, vmax=9)
    
    # Add text to show each warp's specialized role
    for i in range(8):
        ax2.text(0, i, 'TMA', ha='center', va='center', color='white')
        ax2.text(1, i, 'MMA', ha='center', va='center', color='white')
        ax2.text(2, i, 'MMA', ha='center', va='center', color='white')
        ax2.text(3, i, 'Store', ha='center', va='center', color='white')
    
    ax2.set_title('Warp Specialization\n(Each Warp Handles Specific Tasks)')
    ax2.set_xticks(np.arange(4))
    ax2.set_xticklabels([f'Warp {i}' for i in range(4)])
    ax2.set_yticks(np.arange(8))
    ax2.set_yticklabels([f'Time {i}' for i in range(8)])
    
    # Add a shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=plt.cm.tab10(0), markersize=15, label='TMA (Tensor Memory Access)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=plt.cm.tab10(1), markersize=15, label='MMA (Matrix Multiply-Accumulate)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=plt.cm.tab10(2), markersize=15, label='MMA (Matrix Multiply-Accumulate)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=plt.cm.tab10(3), markersize=15, label='Store Results')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    
    return fig


def plot_memory_hierarchy():
    """Visualize the memory hierarchy in DeepGEMM"""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define memory hierarchy components
    components = [
        {'name': 'HBM/Global Memory', 'size': '80GB', 'bandwidth': '~3TB/s', 'level': 0, 'color': 'lightgray'},
        {'name': 'L2 Cache', 'size': '50MB', 'bandwidth': '~6TB/s', 'level': 1, 'color': 'lightblue'},
        {'name': 'Shared Memory', 'size': '232KB per SM', 'bandwidth': '~20TB/s', 'level': 2, 'color': 'lightskyblue'},
        {'name': 'Register File', 'size': '232 regs/thread', 'bandwidth': '~100TB/s', 'level': 3, 'color': 'steelblue'},
        {'name': 'Tensor Cores', 'size': '4th Gen', 'compute': '~1000 TFLOPS (FP8)', 'level': 4, 'color': 'orange'}
    ]
    
    # Create nested boxes representing memory hierarchy
    previous_rect = None
    text_positions = []
    
    for comp in reversed(components):
        level = comp['level']
        width = 0.8 - level * 0.1
        height = 0.7 - level * 0.1
        left = 0.5 - width/2
        bottom = 0.1 + level * 0.05
        
        rect = patches.Rectangle(
            (left, bottom), width, height,
            linewidth=2, edgecolor='black', facecolor=comp['color'], alpha=0.7
        )
        ax.add_patch(rect)
        
        # Store position for text
        text_positions.append({
            'name': comp['name'],
            'x': left + width/2,
            'y': bottom + height - 0.03,
            'details': f"Size: {comp['size']}\nBandwidth: {comp.get('bandwidth', comp.get('compute', 'N/A'))}"
        })
        
        previous_rect = rect
    
    # Add text with names and specs
    for pos in text_positions:
        ax.text(
            pos['x'], pos['y'], pos['name'],
            ha='center', va='center', fontsize=12, fontweight='bold'
        )
        ax.text(
            pos['x'], pos['y'] - 0.07, pos['details'],
            ha='center', va='center', fontsize=10
        )
    
    # Add arrows showing data movement
    # TMA arrows
    ax.arrow(0.3, 0.2, 0, 0.15, head_width=0.02, head_length=0.02, fc='red', ec='red', width=0.005)
    ax.text(0.27, 0.27, 'TMA', ha='right', va='center', fontweight='bold')
    
    # MMA arrows
    ax.arrow(0.7, 0.45, 0, 0.15, head_width=0.02, head_length=0.02, fc='green', ec='green', width=0.005)
    ax.text(0.73, 0.52, 'MMA', ha='left', va='center', fontweight='bold')
    
    # Set limits and remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Title
    ax.set_title('DeepGEMM Memory Hierarchy', fontsize=16)
    
    # Add explanatory text at the bottom
    explanation = (
        "DeepGEMM optimizes data movement through the memory hierarchy:\n"
        "• TMA efficiently transfers data from global memory to shared memory\n"
        "• Software pipelining overlaps computation and memory operations\n"
        "• Register allocation is optimized for tensor core operations\n"
        "• Shared memory is carefully managed with block-level optimizations"
    )
    
    ax.text(0.5, 0.02, explanation, ha='center', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    return fig


def plot_pipeline_stages():
    """Visualize the pipeline stages in DeepGEMM"""
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define the pipeline stages
    stages = [
        {'name': 'Prefetch A', 'color': 'skyblue'},
        {'name': 'Prefetch B', 'color': 'lightgreen'},
        {'name': 'Prefetch Scales', 'color': 'khaki'},
        {'name': 'Compute', 'color': 'salmon'},
        {'name': 'Store Results', 'color': 'plum'}
    ]
    
    num_stages = len(stages)
    num_iterations = 8
    
    # Create a matrix to show the pipeline stages over iterations
    # Each value in the matrix corresponds to a stage ID
    pipeline = np.zeros((num_stages, num_iterations))
    
    # Fill the matrix to show the pipeline
    for i in range(num_iterations):
        for s in range(num_stages):
            stage_idx = (i + s) % num_stages
            pipeline[s, i] = stage_idx
    
    # Create custom colormap from stage colors
    colors = [stage['color'] for stage in stages]
    cmap = mcolors.ListedColormap(colors)
    
    # Plot the pipeline
    im = ax.imshow(pipeline, cmap=cmap, aspect='auto')
    
    # Add text labels
    for i in range(num_iterations):
        for s in range(num_stages):
            stage_idx = int(pipeline[s, i])
            ax.text(i, s, stages[stage_idx]['name'], ha='center', va='center')
    
    # Add labels
    ax.set_xticks(np.arange(num_iterations))
    ax.set_xticklabels([f'Iter {i}' for i in range(num_iterations)])
    ax.set_yticks(np.arange(num_stages))
    ax.set_yticklabels([f'Thread Group {i}' for i in range(num_stages)])
    
    ax.set_title('DeepGEMM Software Pipeline Stages', fontsize=14)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=stage['color'], 
               markersize=10, label=stage['name'])
        for stage in stages
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
    
    # Add explanation
    explanation = (
        "Software pipelining overlaps memory access and computation to hide latency:\n"
        "• Different thread groups handle different pipeline stages in parallel\n"
        "• A given stage processes different data each iteration\n"
        "• This ensures maximum utilization of both memory bandwidth and compute resources"
    )
    
    plt.figtext(0.5, 0.01, explanation, ha='center', va='bottom', fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    return fig


def plot_sass_optimization():
    """Visualize the SASS optimization process in DeepGEMM"""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # First subplot: Before optimization
    before_code = [
        "FFMA R8, R12, R42, R8 (reuse)",
        "FADD R9, -RZ, R8",
        "FFMA R10, R13, R43, R10 (reuse)",
        "FADD R11, -RZ, R10",
        "FFMA R8, R14, R44, R8 (reuse)",
        "FADD R9, -RZ, R8",
        "FFMA R10, R15, R45, R10 (reuse)",
        "FADD R11, -RZ, R10"
    ]
    
    y_positions = np.arange(len(before_code))
    ax1.barh(y_positions, [1] * len(before_code), color='lightcoral', height=0.7)
    
    for i, code in enumerate(before_code):
        ax1.text(0.01, i, code, va='center', fontfamily='monospace', fontsize=10)
        
        # Highlight reuse registers
        if "(reuse)" in code:
            ax1.text(0.9, i, "⚠️ Reuse stalls", va='center', fontsize=9)
    
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels([f"{i}" for i in range(1, len(before_code) + 1)])
    ax1.set_ylim(len(before_code) - 0.5, -0.5)
    ax1.set_title("Before SASS Optimization")
    ax1.set_xlabel("Instruction Execution Time")
    ax1.set_ylabel("Instruction")
    ax1.set_xlim(0, 1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add annotation explaining the issue
    ax1.annotate(
        "Register reuse causes stalls\ndue to result dependencies",
        xy=(0.5, -0.2), xycoords='axes fraction',
        boxcoords="offset points", box_alignment=(0.5, 0.5),
        pad=0.5, bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.7)
    )
    
    # Second subplot: After optimization
    after_code = [
        "FFMA R8, R12, R42, R8",
        "FFMA R10, R13, R43, R10",
        "FADD R9, -RZ, R8",
        "FADD R11, -RZ, R10",
        "FFMA R8, R14, R44, R8",
        "FFMA R10, R15, R45, R10",
        "FADD R9, -RZ, R8",
        "FADD R11, -RZ, R10"
    ]
    
    y_positions = np.arange(len(after_code))
    
    # Colors indicating dependency chains
    colors = ['lightblue', 'lightgreen', 'lightblue', 'lightgreen', 
              'lightblue', 'lightgreen', 'lightblue', 'lightgreen']
    
    ax2.barh(y_positions, [1] * len(after_code), color=colors, height=0.7)
    
    for i, code in enumerate(after_code):
        ax2.text(0.01, i, code, va='center', fontfamily='monospace', fontsize=10)
        
        # Add performance indicators
        ax2.text(0.9, i, "✓ No stalls", va='center', fontsize=9)
    
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([f"{i}" for i in range(1, len(after_code) + 1)])
    ax2.set_ylim(len(after_code) - 0.5, -0.5)
    ax2.set_title("After SASS Optimization")
    ax2.set_xlabel("Instruction Execution Time")
    ax2.set_ylabel("Instruction")
    ax2.set_xlim(0, 1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add annotation explaining the optimization
    ax2.annotate(
        "Interleaved FFMAs and FADDs\nfor better instruction-level parallelism",
        xy=(0.5, -0.2), xycoords='axes fraction',
        boxcoords="offset points", box_alignment=(0.5, 0.5),
        pad=0.5, bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.7)
    )
    
    # Add overall explanation
    explanation = (
        "DeepGEMM includes a post-compilation SASS optimization pass through interleave_ffma.py.\n"
        "This optimization rearranges instructions to reduce register reuse dependencies and improve instruction-level parallelism,\n"
        "resulting in measurable performance improvements (approximately 0.5% according to commit logs)."
    )
    
    plt.figtext(0.5, 0.01, explanation, ha='center', va='bottom', fontsize=11, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    return fig


def main():
    st.set_page_config(
        page_title="DeepGEMM Architecture Visualizer",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("DeepGEMM Architectural Insights")
    st.subheader("Visual exploration of key architectural concepts in DeepGEMM")

    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        """
        This interactive dashboard visualizes the key architectural concepts
        behind DeepGEMM, a high-performance library for FP8 General Matrix
        Multiplications (GEMMs) specifically designed for NVIDIA Hopper tensor cores.
        """
    )
    
    st.sidebar.header("Topics")
    topic = st.sidebar.radio(
        "Select a topic to explore:",
        ["Overview", "Block Scheduling", "Warp Specialization", 
         "Memory Hierarchy", "Pipeline Stages", "SASS Optimization"]
    )

    # Main content
    if topic == "Overview":
        st.write("""
        ## DeepGEMM Architecture Overview
        
        DeepGEMM achieves high performance through several key architectural innovations:
        
        1. **Persistent Block Scheduling**: Thread blocks persist on SMs and process multiple tiles
        2. **Warp Specialization**: Different warps have specialized roles (TMA, MMA, Store)
        3. **Memory Hierarchy Optimization**: Careful management of all memory levels
        4. **Software Pipelining**: Overlapping computation and memory operations
        5. **SASS Optimization**: Post-compilation assembly optimization
        
        Select a topic from the sidebar to explore each concept in detail.
        """)
        
        # Try to load the design.png image, with fallback if it doesn't exist
        import os
        if os.path.exists("figures/design.png"):
            st.image("figures/design.png", caption="DeepGEMM Design Overview", use_column_width=True)
        else:
            st.info("""
            The DeepGEMM design overview image is not available. 
            
            This would typically show a diagram of the architecture with its key components:
            - The persistent block scheduling system
            - Tensor Memory Accelerator (TMA) integration
            - Software pipeline design
            - Warp specialization
            - Memory hierarchy
            """)
            
            # Create a simple placeholder diagram with matplotlib
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create placeholder blocks for architecture components
            components = [
                {'name': 'Tensor Memory\nAccelerator (TMA)', 'color': 'orange', 'position': (0.1, 0.6, 0.35, 0.25)},
                {'name': 'Warp\nSpecialization', 'color': 'lightblue', 'position': (0.55, 0.6, 0.35, 0.25)},
                {'name': 'Software Pipeline', 'color': 'lightgreen', 'position': (0.1, 0.25, 0.35, 0.25)},
                {'name': 'SASS\nOptimization', 'color': 'lightpink', 'position': (0.55, 0.25, 0.35, 0.25)},
                {'name': 'Persistent Block Scheduling', 'color': 'lightyellow', 'position': (0.25, 0.05, 0.5, 0.1)}
            ]
            
            # Add the components to the plot
            for comp in components:
                rect = patches.Rectangle(
                    (comp['position'][0], comp['position'][1]),
                    comp['position'][2], comp['position'][3],
                    linewidth=2, edgecolor='black', facecolor=comp['color'], alpha=0.8
                )
                ax.add_patch(rect)
                ax.text(
                    comp['position'][0] + comp['position'][2]/2,
                    comp['position'][1] + comp['position'][3]/2,
                    comp['name'],
                    ha='center', va='center', fontweight='bold'
                )
            
            # Add connecting lines
            ax.plot([0.3, 0.3], [0.15, 0.25], 'k-', linewidth=1.5)
            ax.plot([0.7, 0.7], [0.15, 0.25], 'k-', linewidth=1.5)
            ax.plot([0.3, 0.3], [0.5, 0.6], 'k-', linewidth=1.5)
            ax.plot([0.7, 0.7], [0.5, 0.6], 'k-', linewidth=1.5)
            
            # Add title
            ax.set_title('DeepGEMM Architecture (Placeholder Diagram)', fontsize=14)
            
            # Set limits and remove axes
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            st.pyplot(fig)
        
    elif topic == "Block Scheduling":
        st.write("""
        ## Persistent Block Scheduling
        
        DeepGEMM uses persistent blocks that stay resident on SMs throughout the entire computation:
        
        - Each thread block processes multiple tiles of the matrix
        - Blocks are scheduled across SMs to maximize utilization
        - TMA (Tensor Memory Accelerator) efficiently loads data from global to shared memory
        - Memory access patterns are optimized for coalescing
        - Software pipelining overlaps computation and memory transfers
        """)
        
        block_scheduling_fig = plot_block_scheduling()
        st.pyplot(block_scheduling_fig)
        
    elif topic == "Warp Specialization":
        st.write("""
        ## Warp Specialization
        
        Traditional CUDA kernels often have all warps perform similar tasks. DeepGEMM instead
        assigns specialized roles to different warps:
        
        - **TMA Warps**: Handle data loading and prefetching
        - **MMA Warps**: Execute tensor core operations
        - **Store Warps**: Handle writing results back to global memory
        
        This specialization enables better pipelining and hardware utilization.
        """)
        
        warp_fig = plot_warp_specialization()
        st.pyplot(warp_fig)
        
    elif topic == "Memory Hierarchy":
        st.write("""
        ## Memory Hierarchy Optimization
        
        DeepGEMM carefully optimizes data movement through the GPU memory hierarchy:
        
        - **Global Memory (HBM)**: ~3TB/s bandwidth, used for input/output matrices
        - **L2 Cache**: Used for intermediate storage and bandwidth amplification
        - **Shared Memory**: Stores matrix tiles and scaling factors
        - **Register File**: Holds intermediate results and operands
        - **Tensor Cores**: 4th generation tensor cores execute matrix operations
        
        The library uses specialized hardware features like TMA (Tensor Memory Accelerator)
        to efficiently move data between memory levels.
        """)
        
        memory_fig = plot_memory_hierarchy()
        st.pyplot(memory_fig)
        
    elif topic == "Pipeline Stages":
        st.write("""
        ## Software Pipelining
        
        DeepGEMM implements sophisticated software pipelining to overlap computation and memory operations:
        
        - Multiple pipeline stages operate concurrently
        - Different thread groups handle different stages in parallel
        - Memory operations are overlapped with computation to hide latency
        - Pipeline depth is auto-tuned based on shared memory capacity
        
        This pipelining is critical for achieving peak performance, as it ensures both
        memory and compute resources are fully utilized.
        """)
        
        pipeline_fig = plot_pipeline_stages()
        st.pyplot(pipeline_fig)
        
    elif topic == "SASS Optimization":
        st.write("""
        ## SASS-level Optimization
        
        DeepGEMM includes a post-compilation SASS optimization pass:
        
        - After CUDA compilation, the interleave_ffma.py script analyzes and modifies the SASS assembly
        - The optimization targets register reuse dependencies
        - Instructions are reordered to interleave operations and reduce stalls
        - This yields approximately 0.5% performance improvement
        
        This optimization is unique to DeepGEMM and demonstrates the attention to detail
        in maximizing performance.
        """)
        
        sass_fig = plot_sass_optimization()
        st.pyplot(sass_fig)


if __name__ == "__main__":
    main()
