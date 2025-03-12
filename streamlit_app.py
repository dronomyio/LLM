import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from PIL import Image
import base64
import time
from io import BytesIO

# Set page configuration
st.set_page_config(
  page_title="Distributed ML Learning System",
  page_icon="🧠",
  layout="wide",
  initial_sidebar_state="expanded"
)

# Define CSS styling
st.markdown("""
<style>
  .main-header {
      font-size: 2.5rem;
      color: #4257b2;
      margin-bottom: 1rem;
  }
  .section-header {
      font-size: 1.8rem;
      color: #3c9caf;
      margin-top: 2rem;
      margin-bottom: 1rem;
  }
  .concept-title {
      font-size: 1.4rem;
      color: #5c4caf;
      margin-top: 1.5rem;
      margin-bottom: 0.5rem;
  }
  .concept-text {
      font-size: 1.1rem;
      line-height: 1.6;
  }
  .highlight {
      background-color: #f0f7fb;
      border-left: 5px solid #3498db;
      padding: 1rem;
      margin: 1rem 0;
  }
  .code-box {
      background-color: #f7f7f7;
      border-left: 5px solid #2c3e50;
      padding: 1rem;
      margin: 1rem 0;
      font-family: monospace;
  }
  .success-box {
      background-color: #d4edda;
      border-left: 5px solid #28a745;
      padding: 1rem;
      margin: 1rem 0;
  }
  .quiz-box {
      background-color: #fff3cd;
      border-left: 5px solid #ffc107;
      padding: 1rem;
      margin: 1rem 0;
  }
</style>
""", unsafe_allow_html=True)

# Helper functions for visualizations
def create_gpu_visualization(num_gpus=4, experts_per_gpu=2, tokens_per_gpu=8, links="both"):
  """Create a visualization of GPUs with experts and communication links"""
  fig = go.Figure()

  # Layout parameters
  gpus_per_row = min(4, num_gpus)
  rows = (num_gpus + gpus_per_row - 1) // gpus_per_row

  # Store positions
  gpu_positions = {}
  expert_positions = {}

  # Colors
  gpu_color = "rgba(66, 135, 245, 0.8)"
  expert_color = "rgba(106, 90, 205, 0.7)"
  nvlink_color = "rgba(255, 165, 0, 0.7)"
  rdma_color = "rgba(220, 20, 60, 0.7)"

  # Create GPUs and experts
  for i in range(num_gpus):
      row = i // gpus_per_row
      col = i % gpus_per_row

      # GPU position
      x = col * 4
      y = row * 6
      gpu_positions[i] = (x, y)

      # Add GPU node
      fig.add_trace(go.Scatter(
          x=[x], y=[y],
          mode='markers+text',
          marker=dict(size=30, color=gpu_color),
          text=[f'GPU {i}'],
          textposition="bottom center",
          name=f'GPU {i}',
          hoverinfo='name'
      ))

      # Add experts for this GPU
      for j in range(experts_per_gpu):
          expert_x = x + (j - (experts_per_gpu-1)/2) * 0.8
          expert_y = y - 1.5
          expert_id = i * experts_per_gpu + j
          expert_positions[expert_id] = (expert_x, expert_y)

          fig.add_trace(go.Scatter(
              x=[expert_x], y=[expert_y],
              mode='markers+text',
              marker=dict(size=20, color=expert_color),
              text=[f'E{expert_id}'],
              textposition="middle center",
              name=f'Expert {expert_id}',
              hoverinfo='name'
          ))

          # Connect expert to its GPU
          fig.add_trace(go.Scatter(
              x=[expert_x, x], y=[expert_y, y],
              mode='lines',
              line=dict(width=2, color='rgba(150, 150, 150, 0.5)'),
              hoverinfo='none',
              showlegend=False
          ))

  # Add communication links
  if links in ["nvlink", "both"]:
      # Add NVLink connections within each row
      for row in range(rows):
          gpus_in_row = min(gpus_per_row, num_gpus - row * gpus_per_row)
          for i in range(gpus_in_row):
              for j in range(i+1, gpus_in_row):
                  gpu1 = row * gpus_per_row + i
                  gpu2 = row * gpus_per_row + j
                  x1, y1 = gpu_positions[gpu1]
                  x2, y2 = gpu_positions[gpu2]

                  fig.add_trace(go.Scatter(
                      x=[x1, x2], y=[y1, y2],
                      mode='lines',
                      line=dict(width=3, color=nvlink_color),
                      name='NVLink',
                      hoverinfo='name',
                      showlegend=(gpu1 == 0 and gpu2 == 1)
                  ))

  if links in ["rdma", "both"]:
      # Add RDMA connections between rows
      for i in range(min(gpus_per_row, num_gpus)):
          for row in range(1, rows):
              if row * gpus_per_row + i < num_gpus:
                  gpu1 = i
                  gpu2 = row * gpus_per_row + i
                  x1, y1 = gpu_positions[gpu1]
                  x2, y2 = gpu_positions[gpu2]

                  fig.add_trace(go.Scatter(
                      x=[x1, x2], y=[y1, y2],
                      mode='lines',
                      line=dict(width=3, color=rdma_color, dash='dash'),
                      name='RDMA',
                      hoverinfo='name',
                      showlegend=(i == 0 and row == 1)
                  ))

  # Layout settings
  fig.update_layout(
      showlegend=True,
      hovermode='closest',
      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
      width=800, height=600
  )

  return fig, gpu_positions, expert_positions

def create_token_routing_animation(num_steps=5, num_gpus=4, experts_per_gpu=2, tokens_per_gpu=8, topk=2):
  """Create a series of figures showing token routing animation"""

  # Get system layout
  base_fig, gpu_positions, expert_positions = create_gpu_visualization(
      num_gpus, experts_per_gpu, tokens_per_gpu
  )

  # Generate token data
  num_tokens = num_gpus * tokens_per_gpu
  num_experts = num_gpus * experts_per_gpu
  token_data = []

  for i in range(num_tokens):
      source_gpu = i // tokens_per_gpu
      # Assign experts (ensure they're on different GPUs for visual clarity)
      expert_candidates = list(range(num_experts))
      np.random.shuffle(expert_candidates)
      assigned_experts = expert_candidates[:topk]

      token_data.append({
          'token_id': i,
          'source_gpu': source_gpu,
          'assigned_experts': assigned_experts,
          'weights': np.random.dirichlet(np.ones(topk)).tolist()
      })

  # Create animation frames
  frames = []

  # Initial positions (tokens at source GPUs)
  initial_fig = go.Figure(base_fig)
  token_positions = {}

  for i, token in enumerate(token_data):
      gpu_id = token['source_gpu']
      x, y = gpu_positions[gpu_id]

      # Distribute tokens around their GPU
      angle = 2 * np.pi * (i % tokens_per_gpu) / tokens_per_gpu
      radius = 1.0
      token_x = x + radius * np.cos(angle)
      token_y = y + radius * np.sin(angle)
      token_positions[i] = (token_x, token_y)

      # Format hover text
      experts_text = ", ".join([f'E{e}' for e in token['assigned_experts']])
      weights_text = ", ".join([f'{w:.2f}' for w in token['weights']])
      hover_text = f"Token {i}<br>Experts: {experts_text}<br>Weights: {weights_text}"

      initial_fig.add_trace(go.Scatter(
          x=[token_x], y=[token_y],
          mode='markers',
          marker=dict(size=10, color="rgba(100, 149, 237, 0.7)"),
          name=f'Token {i}',
          hovertext=hover_text,
          hoverinfo='text'
      ))

  frames.append(initial_fig)

  # Dispatch animation (tokens moving to experts)
  for step in range(1, num_steps):
      frame_fig = go.Figure(base_fig)
      progress = step / (num_steps - 1)

      for i, token in enumerate(token_data):
          start_x, start_y = token_positions[i]

          # For each assigned expert
          for e_idx, expert_id in enumerate(token['assigned_experts']):
              # Target is the expert position
              target_x, target_y = expert_positions[expert_id]

              # Interpolate position
              current_x = start_x + (target_x - start_x) * progress
              current_y = start_y + (target_y - start_y) * progress

              # Add trace for this token
              hover_text = f"Token {i} to Expert {expert_id}<br>Weight: {token['weights'][e_idx]:.2f}"

              frame_fig.add_trace(go.Scatter(
                  x=[current_x], y=[current_y],
                  mode='markers',
                  marker=dict(
                      size=8,
                      color="rgba(100, 149, 237, 0.7)",
                      opacity=0.7 + 0.3 * token['weights'][e_idx]  # Higher weight = more visible
                  ),
                  name=f'Token {i} to E{expert_id}',
                  hovertext=hover_text,
                  hoverinfo='text'
              ))

              # Add line showing path
              frame_fig.add_trace(go.Scatter(
                  x=[start_x, current_x], y=[start_y, current_y],
                  mode='lines',
                  line=dict(width=1, color="rgba(100, 149, 237, 0.3)"),
                  showlegend=False,
                  hoverinfo='none'
              ))

      frames.append(frame_fig)

  return frames

def create_bandwidth_comparison_chart():
  """Create a chart comparing bandwidth of different communication methods"""
  methods = ['PCIe 4.0 x16', 'InfiniBand HDR', 'NVLink 4.0', 'DeepEP Intranode', 'DeepEP Internode']
  bandwidths = [32, 50, 160, 150, 45]

  fig = px.bar(
      x=methods, y=bandwidths,
      labels={'x': 'Communication Method', 'y': 'Bandwidth (GB/s)'},
      color=bandwidths,
      color_continuous_scale='Viridis'
  )

  fig.update_layout(
      title="Communication Bandwidth Comparison",
      coloraxis_showscale=False,
      width=800,
      height=500
  )

  return fig

def create_moe_concept_diagram():
  """Create a conceptual diagram of MoE architecture"""
  # Create figure
  fig = go.Figure()

  # Define positions
  x_input = 1
  x_router = 3
  x_experts = 6
  x_output = 9

  # Input
  fig.add_trace(go.Scatter(
      x=[x_input], y=[5],
      mode='markers+text',
      marker=dict(size=40, color='rgba(65, 105, 225, 0.7)'),
      text=['Input'],
      textposition="middle center",
      name='Input'
  ))

  # Router
  fig.add_trace(go.Scatter(
      x=[x_router], y=[5],
      mode='markers+text',
      marker=dict(size=40, color='rgba(255, 165, 0, 0.7)'),
      text=['Router'],
      textposition="middle center",
      name='Router'
  ))

  # Experts
  expert_ys = [2, 4, 6, 8]
  expert_labels = ['Expert 0', 'Expert 1', 'Expert 2', 'Expert 3']

  for i, y in enumerate(expert_ys):
      fig.add_trace(go.Scatter(
          x=[x_experts], y=[y],
          mode='markers+text',
          marker=dict(size=40, color='rgba(50, 205, 50, 0.7)'),
          text=[expert_labels[i]],
          textposition="middle center",
          name=expert_labels[i]
      ))

  # Output
  fig.add_trace(go.Scatter(
      x=[x_output], y=[5],
      mode='markers+text',
      marker=dict(size=40, color='rgba(65, 105, 225, 0.7)'),
      text=['Output'],
      textposition="middle center",
      name='Output'
  ))

  # Connections
  # Input to Router
  fig.add_trace(go.Scatter(
      x=[x_input, x_router], y=[5, 5],
      mode='lines+text',
      line=dict(width=3, color='rgba(100, 100, 100, 0.7)'),
      text=[''],
      showlegend=False
  ))

  # Router to Experts (highlight top 2)
  for i, y in enumerate(expert_ys):
      width = 4 if i < 2 else 1
      color = 'rgba(255, 0, 0, 0.7)' if i < 2 else 'rgba(200, 200, 200, 0.3)'
      label = f"weight={0.7 if i==0 else 0.3}" if i < 2 else ""

      fig.add_trace(go.Scatter(
          x=[x_router, x_experts], y=[5, y],
          mode='lines+text',
          line=dict(width=width, color=color),
          text=[label] if i < 2 else [''],
          textposition="middle center",
          showlegend=False
      ))

  # Experts to Output (highlight top 2)
  for i, y in enumerate(expert_ys):
      width = 4 if i < 2 else 1
      color = 'rgba(255, 0, 0, 0.7)' if i < 2 else 'rgba(200, 200, 200, 0.3)'

      fig.add_trace(go.Scatter(
          x=[x_experts, x_output], y=[y, 5],
          mode='lines',
          line=dict(width=width, color=color),
          showlegend=False
      ))

  # Layout settings
  fig.update_layout(
      showlegend=False,
      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
      width=800, height=500,
      margin=dict(l=20, r=20, t=20, b=20)
  )

  return fig

# Tutorial content
def intro_page():
  st.markdown('<h1 class="main-header">Distributed Machine Learning: From Basics to Expert Parallelism</h1>',
unsafe_allow_html=True)

  col1, col2 = st.columns([2, 1])

  with col1:
      st.markdown("""
      <p class="concept-text">
      Welcome to our interactive tutorial on distributed machine learning! This system will guide you through
the concepts,
      challenges, and solutions for training and running large neural networks across multiple devices.
      </p>
      
      <p class="concept-text">
      Throughout this tutorial, you'll learn about:
      </p>
      """, unsafe_allow_html=True)

      st.markdown("""
      - Data parallelism, tensor parallelism, and expert parallelism
      - Communication patterns and bottlenecks
      - Specialized hardware interconnects (NVLink, InfiniBand, etc.)
      - Mixture-of-Experts (MoE) architectures
      - Communication libraries like DeepEP
      """)

      st.markdown("""
      <p class="concept-text">
      Each module includes interactive visualizations, code examples, and quizzes to help you understand the 
concepts.
      </p>
      """, unsafe_allow_html=True)

  with col2:
      st.image("https://miro.medium.com/max/1400/1*9GoMhViF4cGgJ9Zebna8zQ.png", width=300)

  st.markdown('<h2 class="section-header">Learning Path</h2>', unsafe_allow_html=True)

  col1, col2, col3, col4 = st.columns(4)

  with col1:
      st.markdown('<div class="highlight">', unsafe_allow_html=True)
      st.markdown("### 1. Foundations")
      st.markdown("- Distributed Computing Basics")
      st.markdown("- GPU Architecture")
      st.markdown("- Communication Patterns")
      st.markdown("</div>", unsafe_allow_html=True)

  with col2:
      st.markdown('<div class="highlight">', unsafe_allow_html=True)
      st.markdown("### 2. Parallelism Strategies")
      st.markdown("- Data Parallelism")
      st.markdown("- Tensor Parallelism")
      st.markdown("- Pipeline Parallelism")
      st.markdown("</div>", unsafe_allow_html=True)

  with col3:
      st.markdown('<div class="highlight">', unsafe_allow_html=True)
      st.markdown("### 3. Expert Parallelism")
      st.markdown("- Mixture-of-Experts")
      st.markdown("- Routing Algorithms")
      st.markdown("- Load Balancing")
      st.markdown("</div>", unsafe_allow_html=True)

  with col4:
      st.markdown('<div class="highlight">', unsafe_allow_html=True)
      st.markdown("### 4. DeepEP in Practice")
      st.markdown("- System Architecture")
      st.markdown("- Performance Tuning")
      st.markdown("- Integration with Models")
      st.markdown("</div>", unsafe_allow_html=True)

  st.markdown("")
  ready = st.button("Begin Tutorial")
  if ready:
      st.session_state.page = "foundations"
      st.rerun()

def foundations_page():
  st.markdown('<h1 class="main-header">Foundations of Distributed ML</h1>', unsafe_allow_html=True)

  st.markdown('<h2 class="section-header">GPU Communication Architecture</h2>', unsafe_allow_html=True)

  col1, col2 = st.columns([3, 2])

  with col1:
      st.markdown("""
      <p class="concept-text">
      Modern ML training and inference relies on GPUs, but as models grow larger, 
      a single GPU isn't enough. This introduces the need to communicate between GPUs,
      which happens through different hardware interfaces:
      </p>
      """, unsafe_allow_html=True)

      st.markdown("""
      - **PCIe**: Standard interface between GPU and CPU, ~32 GB/s bandwidth
      - **NVLink**: NVIDIA's direct GPU-to-GPU connection, ~150-160 GB/s
      - **InfiniBand/RDMA**: Network protocol for GPU-to-GPU communication across nodes, ~50-100 GB/s
      """)

      st.markdown("""
      <p class="concept-text">
      The communication bottleneck is often the limiting factor in distributed training performance!
      </p>
      """, unsafe_allow_html=True)

  with col2:
      bandwidth_fig = create_bandwidth_comparison_chart()
      st.plotly_chart(bandwidth_fig, use_container_width=True)

  st.markdown('<div class="highlight">', unsafe_allow_html=True)
  st.markdown("""
  **Key Insight**: Different communication patterns have drastically different performance characteristics.
  - Intra-node communication via NVLink is ~3-5x faster than inter-node communication
  - Optimizing your algorithm to minimize inter-node communication is crucial
  """)
  st.markdown('</div>', unsafe_allow_html=True)

  st.markdown('<h2 class="section-header">Common Communication Patterns</h2>', unsafe_allow_html=True)

  col1, col2 = st.columns(2)

  with col1:
      st.markdown('<h3 class="concept-title">Point-to-Point</h3>', unsafe_allow_html=True)
      st.markdown("""
      - Direct communication between two GPUs
      - Example: Pipeline parallelism passing activations
      - Implemented using: `send`/`recv` operations
      """)

      st.markdown('<h3 class="concept-title">All-Reduce</h3>', unsafe_allow_html=True)
      st.markdown("""
      - Aggregates values from all GPUs and distributes result
      - Example: Gradient synchronization in data parallel training
      - Implemented using: `all_reduce` operation
      """)

  with col2:
      st.markdown('<h3 class="concept-title">All-to-All</h3>', unsafe_allow_html=True)
      st.markdown("""
      - Each GPU sends distinct data to every other GPU
      - Example: Expert parallelism in Mixture-of-Experts
      - Implemented using: `all_to_all` operation
      - **Most challenging** communication pattern to optimize!
      """)

      st.markdown('<h3 class="concept-title">Broadcast</h3>', unsafe_allow_html=True)
      st.markdown("""
      - One GPU sends same data to all other GPUs
      - Example: Parameter initialization
      - Implemented using: `broadcast` operation
      """)

  st.markdown('<div class="code-box">', unsafe_allow_html=True)
  st.code("""
# Example using PyTorch Distributed for different communication patterns

import torch.distributed as dist

# Initialize process group
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Point-to-Point (send/recv)
if rank == 0:
  dist.send(tensor_to_send, dst=1)
elif rank == 1:
  dist.recv(tensor_to_recv, src=0)

# All-Reduce (sum gradients across all GPUs)
dist.all_reduce(gradients, op=dist.ReduceOp.SUM)

# All-to-All (each GPU sends different data to every other GPU)
outputs = torch.empty_like(inputs)
dist.all_to_all(outputs, inputs)

# Broadcast (share model parameters)
if rank == 0:
  # Only fill the tensor on rank 0
  tensor = torch.randn(20, 10)
else:
  # Create empty tensor on other ranks
  tensor = torch.empty(20, 10)
dist.broadcast(tensor, src=0)
  """)
  st.markdown('</div>', unsafe_allow_html=True)

  # Quiz
  st.markdown('<div class="quiz-box">', unsafe_allow_html=True)
  st.markdown("### Quick Quiz")

  q1 = st.radio(
      "Which communication pattern is the most challenging to optimize in distributed ML?",
      ["Point-to-Point", "All-Reduce", "All-to-All", "Broadcast"]
  )

  if q1:
      if q1 == "All-to-All":
          st.success("✓ Correct! All-to-All is the most challenging because each GPU must send different data \
to every other GPU, creating complex traffic patterns.")
      else:
          st.error("✗ Not quite. All-to-All is the most challenging pattern as it creates complex \
many-to-many traffic patterns.")

  q2 = st.radio(
      "Which hardware interconnect provides the highest bandwidth for GPU-to-GPU communication within a \
node?",
      ["PCIe", "InfiniBand", "NVLink", "Ethernet"]
  )

  if q2:
      if q2 == "NVLink":
          st.success("✓ Correct! NVLink provides the highest bandwidth (~150-160 GB/s) for intra-node GPU \
communication.")
      else:
          st.error("✗ Not quite. NVLink provides the highest bandwidth at around 150-160 GB/s for GPUs within \
the same node.")

  st.markdown('</div>', unsafe_allow_html=True)

  col1, col2 = st.columns([1, 1])
  with col1:
      if st.button("← Back to Intro"):
          st.session_state.page = "intro"
          st.rerun()
  with col2:
      if st.button("Continue to Parallelism Strategies →"):
          st.session_state.page = "parallelism"
          st.rerun()

def parallelism_page():
  st.markdown('<h1 class="main-header">Parallelism Strategies for Large Models</h1>', unsafe_allow_html=True)

  st.markdown("""
  <p class="concept-text">
  As neural networks grow larger, we need different strategies to distribute them across multiple GPUs.
  Each strategy has different communication patterns and trade-offs.
  </p>
  """, unsafe_allow_html=True)

  # Data Parallelism
  st.markdown('<h2 class="section-header">Data Parallelism</h2>', unsafe_allow_html=True)

  col1, col2 = st.columns([3, 2])

  with col1:
      st.markdown("""
      <p class="concept-text">
      Data Parallelism is the simplest form of distributed training, where:
      </p>
      """, unsafe_allow_html=True)

      st.markdown("""
      - **Each GPU** has a complete copy of the model
      - **Different batches** of data are processed on different GPUs
      - **Gradients** are synchronized using all-reduce operations
      - **Parameters** are updated synchronously
      """)

      st.markdown("""
      <p class="concept-text">
      This approach is simple to implement but doesn't solve the problem of models that are too large
      to fit on a single GPU.
      </p>
      """, unsafe_allow_html=True)

  with col2:
      # Simple data parallelism diagram
      st.image("https://miro.medium.com/max/1400/1*5Z-L0RjVC4ZM8PIhFxhvzw.png", width=300)

  st.markdown('<div class="code-box">', unsafe_allow_html=True)
  st.code("""
# Data Parallelism example with PyTorch DDP
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f"cuda:{local_rank}")

# Create model and move it to GPU
model = MyLargeModel().to(device)
ddp_model = DDP(model, device_ids=[local_rank])

# Training loop
optimizer = torch.optim.AdamW(ddp_model.parameters())
for data, target in train_loader:
  data, target = data.to(device), target.to(device)
  output = ddp_model(data)
  loss = loss_fn(output, target)
  
  # Backward pass (gradients automatically synchronized)
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()
  """)
  st.markdown('</div>', unsafe_allow_html=True)

  # Tensor Parallelism
  st.markdown('<h2 class="section-header">Tensor Parallelism</h2>', unsafe_allow_html=True)

  col1, col2 = st.columns([3, 2])

  with col1:
      st.markdown("""
      <p class="concept-text">
      Tensor Parallelism splits individual layers across multiple GPUs:
      </p>
      """, unsafe_allow_html=True)

      st.markdown("""
      - **Individual tensors** (e.g., weight matrices) are partitioned
      - **Each operation** (e.g., matrix multiply) is distributed
      - Requires **all-gather** operations during the forward pass
      - Often used for **attention layers** which have large matrices
      """)

      st.markdown("""
      <p class="concept-text">
      This approach allows models that are larger than a single GPU's memory.
      </p>
      """, unsafe_allow_html=True)

  with col2:
      # Simple tensor parallelism diagram
      st.image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tensor.png", width=300)

  # Pipeline Parallelism
  st.markdown('<h2 class="section-header">Pipeline Parallelism</h2>', unsafe_allow_html=True)

  col1, col2 = st.columns([3, 2])

  with col1:
      st.markdown("""
      <p class="concept-text">
      Pipeline Parallelism splits the model by layers across multiple GPUs:
      </p>
      """, unsafe_allow_html=True)

      st.markdown("""
      - **Different layers** are assigned to different GPUs
      - **Activations** are passed between GPUs in sequence
      - Uses **micro-batches** to keep all GPUs busy
      - Increases **throughput** but may introduce **pipeline bubbles**
      """)

      st.markdown("""
      <p class="concept-text">
      Widely used in very large models as it minimizes communication overhead.
      </p>
      """, unsafe_allow_html=True)

  with col2:
      # Simple pipeline parallelism diagram
      st.image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-pipeline.png", width=300)

  # Combination
  st.markdown('<h2 class="section-header">3D Parallelism: Combining Strategies</h2>', unsafe_allow_html=True)

  st.markdown("""
  <p class="concept-text">
  For the largest models, combinations of different parallelism strategies are used:
  </p>
  """, unsafe_allow_html=True)

  st.markdown("""
  - **Data Parallelism**: Split batches across GPU clusters
  - **Pipeline Parallelism**: Split model layers within each cluster
  - **Tensor Parallelism**: Split individual layers across GPUs within each pipeline stage
  """)

  st.markdown('<div class="highlight">', unsafe_allow_html=True)
  st.markdown("""
  **Communication Bottlenecks by Parallelism Type:**
  
  | Strategy | Communication Pattern | Frequency | Bottleneck |
  |----------|----------------------|-----------|------------|
  | Data Parallel | All-Reduce | Each batch | Gradient synchronization |
  | Tensor Parallel | All-Gather, Reduce-Scatter | Each layer | Attention head synchronization |
  | Pipeline Parallel | Point-to-Point | Each micro-batch | Activation passing between stages |
  | Expert Parallel | All-to-All | Each expert layer | Token routing |
  """)
  st.markdown('</div>', unsafe_allow_html=True)

  # Quiz
  st.markdown('<div class="quiz-box">', unsafe_allow_html=True)
  st.markdown("### Quick Quiz")

  q1 = st.radio(
      "Which parallelism strategy requires the least amount of communication between GPUs?",
      ["Data Parallelism", "Tensor Parallelism", "Pipeline Parallelism", "Expert Parallelism"]
  )

  if q1:
      if q1 == "Pipeline Parallelism":
          st.success("✓ Correct! Pipeline parallelism only requires communication at the boundaries between \
pipeline stages, making it efficient for large models.")
      else:
          st.error("✗ Not quite. Pipeline parallelism has the least communication as it only needs to send \
activations between pipeline stages.")

  q2 = st.radio(
      "In data parallelism, what is synchronized between GPUs?",
      ["The input data", "The entire model", "The gradients", "The batch normalization statistics"]
  )

  if q2:
      if q2 == "The gradients":
          st.success("✓ Correct! In data parallelism, each GPU computes gradients on its portion of data, and \
these gradients are synchronized (typically via all-reduce) before updating the model parameters.")
      else:
          st.error("✗ Not quite. In data parallelism, we synchronize gradients computed from different data \
batches.")

  st.markdown('</div>', unsafe_allow_html=True)

  col1, col2 = st.columns([1, 1])
  with col1:
      if st.button("← Back to Foundations"):
          st.session_state.page = "foundations"
          st.rerun()
  with col2:
      if st.button("Continue to Expert Parallelism →"):
          st.session_state.page = "expert_parallelism"
          st.rerun()

def expert_parallelism_page():
  st.markdown('<h1 class="main-header">Expert Parallelism & Mixture-of-Experts</h1>', unsafe_allow_html=True)

  st.markdown("""
  <p class="concept-text">
  Expert Parallelism is a specialized form of model parallelism used in Mixture-of-Experts (MoE) 
architectures.
  It allows scaling models to enormous sizes while keeping computation efficient.
  </p>
  """, unsafe_allow_html=True)

  # MoE Concept
  st.markdown('<h2 class="section-header">Mixture-of-Experts Architecture</h2>', unsafe_allow_html=True)

  col1, col2 = st.columns([2, 3])

  with col1:
      st.markdown("""
      <p class="concept-text">
      A Mixture-of-Experts layer consists of:
      </p>
      """, unsafe_allow_html=True)

      st.markdown("""
      - A **router network** that decides which experts process each token
      - Multiple **expert networks** (typically MLPs), each specializing in different token patterns
      - A **gating mechanism** that determines how to weight expert outputs
      """)

      st.markdown("""
      <p class="concept-text">
      Key advantages:
      </p>
      """, unsafe_allow_html=True)

      st.markdown("""
      - **Conditional computation**: Only activate a subset of parameters for each input
      - **Specialization**: Experts naturally specialize in different input patterns
      - **Scaling efficiency**: Can scale parameter count without increasing compute per token
      """)

  with col2:
      moe_fig = create_moe_concept_diagram()
      st.plotly_chart(moe_fig, use_container_width=True)

  st.markdown('<div class="highlight">', unsafe_allow_html=True)
  st.markdown("""
  **How MoE Works in Practice:**
  
  1. Each token's representation is processed by the router network
  2. Router produces scores for each expert
  3. Top-k experts (usually k=1 or k=2) are selected for each token
  4. Tokens are sent only to their selected experts
  5. Expert outputs are weighted and combined
  6. The final output replaces what would be a standard FFN layer
  """)
  st.markdown('</div>', unsafe_allow_html=True)

  # MoE Code Example
  st.markdown('<h2 class="section-header">MoE Implementation Example</h2>', unsafe_allow_html=True)

  st.markdown('<div class="code-box">', unsafe_allow_html=True)
  st.code("""
# Simplified Mixture-of-Experts implementation with PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
  def __init__(self, input_size, hidden_size, num_experts, k=2):
      super().__init__()
      self.input_size = input_size
      self.hidden_size = hidden_size
      self.num_experts = num_experts
      self.k = k
      
      # Router network
      self.router = nn.Linear(input_size, num_experts)
      
      # Expert networks
      self.experts = nn.ModuleList([
          nn.Sequential(
              nn.Linear(input_size, hidden_size),
              nn.GELU(),
              nn.Linear(hidden_size, input_size)
          ) for _ in range(num_experts)
      ])
  
  def forward(self, x):
      # x shape: [batch_size, seq_len, input_size]
      batch_size, seq_len, _ = x.shape
      
      # Flatten sequence dimension for routing
      x_flat = x.reshape(-1, self.input_size)  # [batch_size*seq_len, input_size]
      
      # Get router scores
      router_logits = self.router(x_flat)  # [batch_size*seq_len, num_experts]
      
      # Get top-k experts and their scores
      router_probs = F.softmax(router_logits, dim=-1)
      topk_probs, topk_indices = torch.topk(router_probs, self.k, dim=-1)
      
      # Normalize probabilities
      topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
      
      # Initialize output
      final_output = torch.zeros_like(x_flat)
      
      # Dispatch tokens to their corresponding experts and combine outputs
      for i in range(self.k):
          # Get expert indices for this k
          expert_idx = topk_indices[:, i]  # [batch_size*seq_len]
          
          # Get corresponding probabilities
          prob = topk_probs[:, i].unsqueeze(-1)  # [batch_size*seq_len, 1]
          
          # For each expert, process its assigned tokens
          for expert_id in range(self.num_experts):
              # Find tokens assigned to this expert
              token_idx = torch.where(expert_idx == expert_id)[0]
              if len(token_idx) == 0:
                  continue
                  
              # Get tokens
              expert_inputs = x_flat[token_idx]
              
              # Process tokens with this expert
              expert_outputs = self.experts[expert_id](expert_inputs)
              
              # Combine weighted outputs
              final_output[token_idx] += prob[token_idx] * expert_outputs
      
      # Reshape back to [batch_size, seq_len, input_size]
      output = final_output.reshape(batch_size, seq_len, self.input_size)
      return output
  """)
  st.markdown('</div>', unsafe_allow_html=True)

  st.markdown("""
  <p class="concept-text">
  The above implementation is simplified for clarity. In real distributed MoE, we need efficient
  communication to route tokens to experts across GPUs.
  </p>
  """, unsafe_allow_html=True)

  # Expert Parallelism
  st.markdown('<h2 class="section-header">Expert Parallelism: The Communication Challenge</h2>',
unsafe_allow_html=True)

  st.markdown("""
  <p class="concept-text">
  When scaling MoE models across multiple GPUs, we face a unique communication challenge:
  </p>
  """, unsafe_allow_html=True)

  col1, col2 = st.columns(2)

  with col1:
      st.markdown("#### The Problem")
      st.markdown("""
      - Experts are distributed across multiple GPUs
      - Tokens need to be routed to their assigned experts
      - Each token goes to a different subset of experts
      - This creates an all-to-all communication pattern
      - Naive implementation leads to communication bottlenecks
      """)

  with col2:
      st.markdown("#### The Solution (DeepEP)")
      st.markdown("""
      - Specialized all-to-all communication kernels
      - Optimized for both NVLink and RDMA
      - Different modes for training vs. inference
      - Communication-computation overlapping
      - Load balancing strategies
      """)

  # Interactive visualization
  st.markdown('<h2 class="section-header">Interactive Visualization: Token Routing</h2>',
unsafe_allow_html=True)

  # Controls
  col1, col2, col3, col4 = st.columns(4)
  with col1:
      num_gpus = st.slider("Number of GPUs", 2, 8, 4)
  with col2:
      experts_per_gpu = st.slider("Experts per GPU", 1, 4, 2)
  with col3:
      topk = st.slider("Top-k Experts", 1, 4, 2)
  with col4:
      tokens_per_gpu = st.slider("Tokens per GPU", 4, 16, 8)

  # Generate visualization frames
  token_frames = create_token_routing_animation(
      num_steps=5,
      num_gpus=num_gpus,
      experts_per_gpu=experts_per_gpu,
      tokens_per_gpu=tokens_per_gpu,
      topk=topk
  )

  # Display with animation control
  frame_index = st.slider("Animation Frame", 0, len(token_frames)-1, 0)
  st.plotly_chart(token_frames[frame_index], use_container_width=True)

  st.markdown('<div class="highlight">', unsafe_allow_html=True)
  st.markdown("""
  **Key Insights from the Visualization:**
  
  1. Tokens from each GPU need to be sent to multiple destination GPUs
  2. The communication pattern is sparse (not all tokens go to all GPUs)
  3. The pattern changes for every batch of tokens
  4. Optimizing this communication is the primary goal of DeepEP
  """)
  st.markdown('</div>', unsafe_allow_html=True)

  # Quiz
  st.markdown('<div class="quiz-box">', unsafe_allow_html=True)
  st.markdown("### Quick Quiz")

  q1 = st.radio(
      "In a Mixture-of-Experts model, what decides which experts process each token?",
      ["A random assignment", "The router network", "A fixed pattern", "The attention mechanism"]
  )

  if q1:
      if q1 == "The router network":
          st.success("✓ Correct! The router network learns to assign tokens to the most appropriate \
experts.")
      else:
          st.error("✗ Not quite. The router network is responsible for determining which experts should \
process each token.")

  q2 = st.radio(
      "What is the primary communication pattern in Expert Parallelism?",
      ["All-Reduce", "Point-to-Point", "Broadcast", "All-to-All"]
  )

  if q2:
      if q2 == "All-to-All":
          st.success("✓ Correct! Expert Parallelism requires all-to-all communication as tokens from any GPU \
might need to be processed by experts on any other GPU.")
      else:
          st.error("✗ Not quite. Expert Parallelism primarily uses all-to-all communication for routing \
tokens between experts on different GPUs.")

  st.markdown('</div>', unsafe_allow_html=True)

  col1, col2 = st.columns([1, 1])
  with col1:
      if st.button("← Back to Parallelism Strategies"):
          st.session_state.page = "parallelism"
          st.rerun()
  with col2:
      if st.button("Continue to DeepEP in Practice →"):
          st.session_state.page = "deepep"
          st.rerun()

def deepep_page():
  st.markdown('<h1 class="main-header">DeepEP in Practice: Communication for Expert Parallelism</h1>',
unsafe_allow_html=True)

  st.markdown("""
  <p class="concept-text">
  DeepEP is a specialized communication library designed for Mixture-of-Experts models. It optimizes 
  the all-to-all communication pattern required for expert parallelism, enabling efficient scaling 
  of MoE models to hundreds or thousands of experts across multiple GPUs.
  </p>
  """, unsafe_allow_html=True)

  # DeepEP Architecture
  st.markdown('<h2 class="section-header">DeepEP System Architecture</h2>', unsafe_allow_html=True)

  col1, col2 = st.columns([2, 3])

  with col1:
      st.markdown("""
      <p class="concept-text">
      DeepEP's architecture consists of:
      </p>
      """, unsafe_allow_html=True)

      st.markdown("""
      - **Core Buffer Class**: Manages communication resources
      - **Dispatch Operation**: Sends tokens to experts
      - **Combine Operation**: Aggregates results back
      - **Intranode Kernels**: Using NVLink for GPUs in the same node
      - **Internode Kernels**: Using RDMA between nodes
      - **Low-Latency Mode**: Optimized for inference
      """)

      st.markdown("""
      <p class="concept-text">
      The library is designed to maximize bandwidth utilization and minimize latency.
      </p>
      """, unsafe_allow_html=True)

  with col2:
      # Generate system architecture visualization
      sys_fig, _, _ = create_gpu_visualization(num_gpus=8, experts_per_gpu=2, links="both")
      st.plotly_chart(sys_fig, use_container_width=True)

  # DeepEP Key Operations
  st.markdown('<h2 class="section-header">Key Operations: Dispatch & Combine</h2>', unsafe_allow_html=True)

  col1, col2 = st.columns(2)

  with col1:
      st.markdown('<h3 class="concept-title">Dispatch Operation</h3>', unsafe_allow_html=True)
      st.markdown("""
      - Routes tokens to their assigned experts across GPUs
      - Takes token hidden states, expert assignments, and weights
      - Returns tokens re-organized by expert
      - Preserves routing information for the combine operation
      """)

      st.markdown('<div class="code-box">', unsafe_allow_html=True)
      st.code("""
# Simplified dispatch operation
recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = \
  buffer.dispatch(
      x,                  # Token hidden states
      topk_idx,           # Expert assignments
      topk_weights,       # Expert weights
      num_tokens_per_rank,
      num_tokens_per_rdma_rank,
      is_token_in_rank,
      num_tokens_per_expert
  )
      """)
      st.markdown('</div>', unsafe_allow_html=True)

  with col2:
      st.markdown('<h3 class="concept-title">Combine Operation</h3>', unsafe_allow_html=True)
      st.markdown("""
      - Gathers processed tokens back to their source GPUs
      - Applies expert weights to the results
      - Efficiently routes data using the handle from dispatch
      - Returns the final weighted combination of expert outputs
      """)

      st.markdown('<div class="code-box">', unsafe_allow_html=True)
      st.code("""
# Simplified combine operation
combined_x, combined_topk_weights, event = \
  buffer.combine(
      expert_outputs,    # Outputs from local experts
      handle,            # Communication handle from dispatch
      topk_weights       # Expert weights
  )
      """)
      st.markdown('</div>', unsafe_allow_html=True)

  # DeepEP Performance Features
  st.markdown('<h2 class="section-header">Advanced Performance Features</h2>', unsafe_allow_html=True)

  col1, col2, col3 = st.columns(3)

  with col1:
      st.markdown('<h3 class="concept-title">Low-Latency Mode</h3>', unsafe_allow_html=True)
      st.markdown("""
      - Specialized for inference decoding
      - Pure RDMA implementation for low latency
      - Sub-200μs latency for dispatch/combine
      - Hook-based communication-computation overlap
      """)

  with col2:
      st.markdown('<h3 class="concept-title">FP8 Support</h3>', unsafe_allow_html=True)
      st.markdown("""
      - Reduces bandwidth requirements
      - Automatic precision conversion
      - 8-bit dispatch with 16-bit combine
      - Maintains accuracy for MoE operations
      """)

  with col3:
      st.markdown('<h3 class="concept-title">SM Control</h3>', unsafe_allow_html=True)
      st.markdown("""
      - Fine-grained control of GPU resources
      - Balance between communication and computation
      - Auto-tuned configurations for different sizes
      - Optimized kernel launch parameters
      """)

  st.markdown('<div class="highlight">', unsafe_allow_html=True)
  st.markdown("""
  **DeepEP Performance Results:**
  
  | Setting | Operation | #EP | Bottleneck Bandwidth | Theoretical Maximum |
  |---------|-----------|-----|---------------------|---------------------|
  | Intranode | Dispatch | 8 | 153 GB/s (NVLink) | ~160 GB/s |
  | Intranode | Combine | 8 | 158 GB/s (NVLink) | ~160 GB/s |
  | Internode | Dispatch | 64 | 46 GB/s (RDMA) | ~50 GB/s |
  | Internode | Combine | 64 | 45 GB/s (RDMA) | ~50 GB/s |
  
  DeepEP achieves near-theoretical bandwidth limits for both intranode and internode communication!
  """)
  st.markdown('</div>', unsafe_allow_html=True)

  # DeepEP Integration Example
  st.markdown('<h2 class="section-header">Integrating DeepEP in Your MoE Model</h2>', unsafe_allow_html=True)

  st.markdown('<div class="code-box">', unsafe_allow_html=True)
  st.code("""
import torch
import torch.distributed as dist
from deep_ep import Buffer, EventOverlap

class MoELayerWithDeepEP(nn.Module):
  def __init__(self, hidden_size, ffn_size, num_experts, k=2):
      super().__init__()
      self.hidden_size = hidden_size
      self.ffn_size = ffn_size
      self.num_experts = num_experts
      self.k = k
      
      # Router network
      self.router = nn.Linear(hidden_size, num_experts)
      
      # Create expert networks (one group of local experts per GPU)
      local_experts_per_rank = num_experts // dist.get_world_size()
      self.local_experts = nn.ModuleList([
          nn.Sequential(
              nn.Linear(hidden_size, ffn_size),
              nn.GELU(),
              nn.Linear(ffn_size, hidden_size)
          ) for _ in range(local_experts_per_rank)
      ])
      
      # Get DeepEP buffer
      self.buffer = self._get_buffer(hidden_size)
  
  def _get_buffer(self, hidden_size):
      # Get config for buffer size hints
      group = dist.group.WORLD
      config_dispatch = Buffer.get_dispatch_config(group.size())
      config_combine = Buffer.get_combine_config(group.size())
      
      # Calculate buffer sizes
      nvl_bytes = max(
          config_dispatch.get_nvl_buffer_size_hint(hidden_size * 2, group.size()),
          config_combine.get_nvl_buffer_size_hint(hidden_size * 2, group.size())
      )
      rdma_bytes = max(
          config_dispatch.get_rdma_buffer_size_hint(hidden_size * 2, group.size()),
          config_combine.get_rdma_buffer_size_hint(hidden_size * 2, group.size())
      )
      
      # Create buffer
      return Buffer(group, nvl_bytes, rdma_bytes)
  
  def forward(self, hidden_states):
      # hidden_states: [batch_size, seq_len, hidden_size]
      batch_size, seq_len, _ = hidden_states.shape
      
      # Reshape for routing
      hidden_flat = hidden_states.reshape(-1, self.hidden_size)
      num_tokens = hidden_flat.size(0)
      
      # Get router scores and top-k experts
      router_logits = self.router(hidden_flat)
      router_probs = F.softmax(router_logits, dim=-1)
      topk_probs, topk_indices = torch.topk(router_probs, self.k, dim=-1)
      
      # Normalize probabilities
      topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
      
      # Calculate dispatch layout
      num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = \
          self.buffer.get_dispatch_layout(topk_indices, self.num_experts)
      
      # Dispatch tokens to experts
      recv_hidden_states, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, _ = \
          self.buffer.dispatch(
              hidden_flat, 
              topk_idx=topk_indices,
              topk_weights=topk_probs,
              num_tokens_per_rank=num_tokens_per_rank,
              num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
              is_token_in_rank=is_token_in_rank,
              num_tokens_per_expert=num_tokens_per_expert
          )
      
      # Process tokens with local experts
      expert_outputs = []
      offset = 0
      for i, expert in enumerate(self.local_experts):
          # Get number of tokens for this expert
          num_tokens_for_expert = num_recv_tokens_per_expert_list[i]
          if num_tokens_for_expert == 0:
              continue
              
          # Get tokens for this expert
          expert_input = recv_hidden_states[offset:offset + num_tokens_for_expert]
          offset += num_tokens_for_expert
          
          # Process through expert
          expert_output = expert(expert_input)
          expert_outputs.append(expert_output)
      
      # Combine all expert outputs
      all_expert_outputs = torch.cat(expert_outputs, dim=0)
      
      # Combine results back to original shape
      combined_hidden_states, _, _ = self.buffer.combine(
          all_expert_outputs, 
          handle, 
          topk_weights=recv_topk_weights
      )
      
      # Reshape back to [batch_size, seq_len, hidden_size]
      output = combined_hidden_states.reshape(batch_size, seq_len, self.hidden_size)
      return output
  """)
  st.markdown('</div>', unsafe_allow_html=True)

  # Performance Tuning
  st.markdown('<h2 class="section-header">Performance Tuning Tips</h2>', unsafe_allow_html=True)

  st.markdown('<div class="highlight">', unsafe_allow_html=True)
  st.markdown("""
  **Optimizing DeepEP Performance:**
  
  1. **SM Allocation**: Tune `Buffer.set_num_sms(N)` to balance communication and computation resources
  2. **Buffer Sizing**: Properly size NVLink and RDMA buffers based on your model dimensions
  3. **Low-Latency Mode**: Use for inference decoding with `low_latency_mode=True`
  4. **Communication Overlap**: Utilize `previous_event` and `async_finish` parameters for overlapping
  5. **Kernel Configuration**: Select appropriate configurations with auto-tuning for your hardware
  6. **Network Settings**: Configure proper RDMA settings (QPs, Virtual Lanes, etc.)
  7. **Precision Control**: Use FP8 for dispatch to reduce bandwidth requirements
  """)
  st.markdown('</div>', unsafe_allow_html=True)

  # Quiz
  st.markdown('<div class="quiz-box">', unsafe_allow_html=True)
  st.markdown("### Final Quiz")

  q1 = st.radio(
      "Which DeepEP mode is specifically optimized for inference decoding?",
      ["High-throughput mode", "Low-latency mode", "Intranode mode", "Internode mode"]
  )

  if q1:
      if q1 == "Low-latency mode":
          st.success("✓ Correct! DeepEP's low-latency mode is specifically designed for inference decoding \
scenarios where minimizing latency is critical.")
      else:
          st.error("✗ Not quite. DeepEP provides a low-latency mode specifically optimized for inference \
decoding scenarios.")

  q2 = st.radio(
      "What is the primary advantage of using FP8 precision in DeepEP dispatch operations?",
      ["Faster computation on GPUs", "Reduced bandwidth requirements", "Improved numerical stability", "Lower \
power consumption"]
  )

  if q2:
      if q2 == "Reduced bandwidth requirements":
          st.success("✓ Correct! Using FP8 precision reduces the amount of data that needs to be transferred \
between GPUs, effectively reducing bandwidth requirements.")
      else:
          st.error("✗ Not quite. FP8 precision primarily helps by reducing the bandwidth required for \
communication between GPUs.")

  q3 = st.radio(
      "What unique feature does DeepEP offer for communication-computation overlap in inference?",
      ["CUDA Graphs integration", "Hook-based background communication", "Tensor parallelism", "Pipeline \
bubbles"]
  )

  if q3:
      if q3 == "Hook-based background communication":
          st.success("Correct! DeepEP's hook-based background communication allows RDMA operations to \
happen in the background without consuming SM resources.")
      else:
          st.error("✗ Not quite. DeepEP uses a hook-based approach for background communication that doesn't \
consume SM resources.")

  st.markdown('</div>', unsafe_allow_html=True)

  # Conclusion
  st.markdown('<h2 class="section-header">Conclusion</h2>', unsafe_allow_html=True)

  st.markdown("""
  <p class="concept-text">
  DeepEP solves the critical communication bottleneck in Mixture-of-Experts models by providing highly 
optimized
  all-to-all communication for expert parallelism. Its specialized kernels for different scenarios allow MoE 
models
  to scale efficiently to hundreds or thousands of experts across multiple GPUs and nodes.
  </p>
  
  <p class="concept-text">
  Key takeaways:
  </p>
  """, unsafe_allow_html=True)

  st.markdown("""
  - Communication is the primary bottleneck in large-scale MoE models
  - Different hardware interconnects (NVLink, RDMA) require specialized optimization
  - DeepEP provides near-theoretical-maximum bandwidth utilization
  - Low-latency and high-throughput modes address different use cases
  - Proper integration and tuning are essential for optimal performance
  """)

  st.markdown('<div class="success-box">', unsafe_allow_html=True)
  st.markdown("""
  **Congratulations on completing the tutorial!**
  
  You now understand the fundamentals of distributed machine learning, parallelism strategies,
  and the specific challenges and solutions for expert parallelism using DeepEP.
  
  To learn more, explore the DeepEP GitHub repository and the accompanying papers.
  """)
  st.markdown('</div>', unsafe_allow_html=True)

  col1, col2 = st.columns([1, 1])
  with col1:
      if st.button("← Back to Expert Parallelism"):
          st.session_state.page = "expert_parallelism"
          st.rerun()
  with col2:
      if st.button("Return to Tutorial Home"):
          st.session_state.page = "intro"
          st.rerun()

# Main app flow
def main():
  # Initialize session state
  if 'page' not in st.session_state:
      st.session_state.page = "intro"

  # Display the current page
  if st.session_state.page == "intro":
      intro_page()
  elif st.session_state.page == "foundations":
      foundations_page()
  elif st.session_state.page == "parallelism":
      parallelism_page()
  elif st.session_state.page == "expert_parallelism":
      expert_parallelism_page()
  elif st.session_state.page == "deepep":
      deepep_page()

if __name__ == "__main__":
  main()

