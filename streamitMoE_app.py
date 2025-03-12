import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import time

# Page setup
st.set_page_config(page_title="MoE Communication Visualizer", layout="wide")

# Define colors for different components
COLORS = {
  "dispatch": "rgba(65, 105, 225, 0.7)",  # Royal Blue
  "combine": "rgba(50, 205, 50, 0.7)",    # Lime Green
  "nvlink": "rgba(255, 165, 0, 0.7)",     # Orange
  "rdma": "rgba(220, 20, 60, 0.7)",       # Crimson
  "expert": "rgba(106, 90, 205, 0.7)",    # Slate Blue
  "token": "rgba(100, 149, 237, 0.7)"     # Cornflower Blue
}

# Title and description
st.title("DeepEP: MoE Communication Pattern Visualizer")
st.markdown("""
This visualizer demonstrates how tokens are routed between GPUs and experts in Mixture-of-Experts models
using DeepEP's optimized communication patterns.
""")

# Sidebar controls
with st.sidebar:
  st.header("System Configuration")

  num_gpus = st.slider("Number of GPUs", 2, 16, 4, 2)
  experts_per_gpu = st.slider("Experts per GPU", 1, 8, 2)
  tokens_per_gpu = st.slider("Tokens per GPU", 4, 32, 16, 4)
  topk = st.slider("Top-k Experts", 1, 4, 2)

  comm_type = st.radio(
      "Communication Type",
      ["Intranode (NVLink)", "Internode (RDMA)", "Mixed"],
      index=2
  )

  operation = st.radio(
      "Operation",
      ["Dispatch", "Combine", "Complete Flow"],
      index=0
  )

  animation_speed = st.slider("Animation Speed", 0.1, 2.0, 1.0, 0.1)

  st.markdown("---")
  st.markdown("""
  **About:**
  
  This app visualizes how DeepEP optimizes token routing in 
  Mixture-of-Experts models for efficient expert parallelism.
  """)

# Helper functions
def create_system_layout(num_gpus, experts_per_gpu):
  """Calculate GPU and expert positions based on system size"""
  # Define node positions
  nodes_per_row = min(4, num_gpus)
  rows = (num_gpus + nodes_per_row - 1) // nodes_per_row

  # GPU and expert positions
  gpu_positions = {}
  expert_positions = {}

  # Calculate positions
  for i in range(num_gpus):
      row = i // nodes_per_row
      col = i % nodes_per_row

      # GPU position
      x = col * 4
      y = row * 6
      gpu_positions[i] = (x, y)

      # Expert positions for this GPU
      for j in range(experts_per_gpu):
          expert_x = x + (j - (experts_per_gpu-1)/2) * 0.8
          expert_y = y - 1.5
          expert_id = i * experts_per_gpu + j
          expert_positions[expert_id] = (expert_x, expert_y)

  return gpu_positions, expert_positions, nodes_per_row, rows

def create_system_visualization(gpu_positions, expert_positions, experts_per_gpu, comm_type, nodes_per_row, 
rows):
  """Create a fresh visualization of the system architecture"""
  fig = go.Figure()

  # Add GPU nodes
  for i, (x, y) in gpu_positions.items():
      fig.add_trace(go.Scatter(
          x=[x], y=[y],
          mode='markers+text',
          marker=dict(size=30, color='rgba(66, 135, 245, 0.8)'),
          text=[f'GPU {i}'],
          textposition="bottom center",
          name=f'GPU {i}',
          hoverinfo='name'
      ))

  # Add expert nodes
  for expert_id, (expert_x, expert_y) in expert_positions.items():
      gpu_id = expert_id // experts_per_gpu
      x, y = gpu_positions[gpu_id]

      # Add expert
      fig.add_trace(go.Scatter(
          x=[expert_x], y=[expert_y],
          mode='markers+text',
          marker=dict(size=20, color=COLORS["expert"]),
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
  if comm_type in ["Intranode (NVLink)", "Mixed"]:
      # Add NVLink connections within rows
      for row in range(rows):
          gpus_in_row = min(nodes_per_row, len(gpu_positions) - row * nodes_per_row)
          for i in range(gpus_in_row):
              for j in range(i+1, gpus_in_row):
                  gpu1 = row * nodes_per_row + i
                  gpu2 = row * nodes_per_row + j
                  x1, y1 = gpu_positions[gpu1]
                  x2, y2 = gpu_positions[gpu2]

                  fig.add_trace(go.Scatter(
                      x=[x1, x2], y=[y1, y2],
                      mode='lines',
                      line=dict(width=3, color=COLORS["nvlink"]),
                      name='NVLink',
                      hoverinfo='name',
                      showlegend=(gpu1 == 0 and gpu2 == 1)
                  ))

  if comm_type in ["Internode (RDMA)", "Mixed"]:
      # Add RDMA connections between rows
      for i in range(min(nodes_per_row, len(gpu_positions))):
          for row in range(1, rows):
              if row * nodes_per_row + i < len(gpu_positions):
                  gpu1 = i
                  gpu2 = row * nodes_per_row + i
                  x1, y1 = gpu_positions[gpu1]
                  x2, y2 = gpu_positions[gpu2]

                  fig.add_trace(go.Scatter(
                      x=[x1, x2], y=[y1, y2],
                      mode='lines',
                      line=dict(width=3, color=COLORS["rdma"], dash='dash'),
                      name='RDMA',
                      hoverinfo='name',
                      showlegend=(i == 0 and row == 1)
                  ))

  # Set layout
  fig.update_layout(
      showlegend=True,
      hovermode='closest',
      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2, (nodes_per_row-1)*4+2]),
      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2, (rows-1)*6+2]),
      height=600,
      margin=dict(l=20, r=20, t=20, b=20)
  )

  return fig

def generate_token_data(num_gpus, tokens_per_gpu, experts_per_gpu, topk):
  """Generate token data with expert assignments"""
  token_data = []
  num_tokens = num_gpus * tokens_per_gpu
  num_experts = num_gpus * experts_per_gpu

  for i in range(num_tokens):
      source_gpu = i // tokens_per_gpu
      # Try to assign experts on different GPUs for visualization clarity
      expert_candidates = list(range(num_experts))
      np.random.shuffle(expert_candidates)
      assigned_experts = expert_candidates[:topk]

      token_data.append({
          'token_id': i,
          'source_gpu': source_gpu,
          'assigned_experts': assigned_experts,
          'weights': np.random.dirichlet(np.ones(topk)).tolist()
      })

  return token_data

def get_token_positions(token_data, gpu_positions, tokens_per_gpu):
  """Calculate initial token positions around their source GPUs"""
  token_positions = {}

  for i, token in enumerate(token_data):
      source_gpu = token['source_gpu']
      source_x, source_y = gpu_positions[source_gpu]

      # Distribute tokens around the GPU
      angle = 2 * np.pi * (i % tokens_per_gpu) / tokens_per_gpu
      radius = 1.0
      token_x = source_x + radius * np.cos(angle)
      token_y = source_y + radius * np.sin(angle)

      token_positions[i] = (token_x, token_y)

  return token_positions

def create_token_traces(token_data, token_positions, expert_positions, progress, operation_type):
  """Create visualization traces for token routing"""
  traces = []

  # For each token in the data
  for i, token in enumerate(token_data):
      token_x, token_y = token_positions[i]

      # For each assigned expert
      for e_idx, expert_id in enumerate(token['assigned_experts']):
          expert_x, expert_y = expert_positions[expert_id]
          weight = token['weights'][e_idx]

          # Calculate current position based on animation progress
          if operation_type == "Dispatch":
              # Token moves from source GPU to expert
              current_x = token_x + (expert_x - token_x) * progress
              current_y = token_y + (expert_y - token_y) * progress
              start_x, start_y = token_x, token_y
              end_x, end_y = expert_x, expert_y
              hover_text = f"Token {i} → Expert {expert_id}<br>Weight: {weight:.2f}"
          else:  # Combine
              # Token moves from expert back to source GPU
              current_x = expert_x + (token_x - expert_x) * progress
              current_y = expert_y + (token_y - expert_y) * progress
              start_x, start_y = expert_x, expert_y
              end_x, end_y = token_x, token_y
              hover_text = f"Expert {expert_id} → Token {i}<br>Weight: {weight:.2f}"

          # Add token trace
          traces.append(go.Scatter(
              x=[current_x], y=[current_y],
              mode='markers',
              marker=dict(
                  size=8,
                  color=COLORS["token"],
                  opacity=0.7 + 0.3 * weight  # Higher weight = more visible
              ),
              name=f'Token {i}',
              hovertext=hover_text,
              hoverinfo='text'
          ))

          # Add path line
          traces.append(go.Scatter(
              x=[start_x, current_x], y=[start_y, current_y],
              mode='lines',
              line=dict(width=1, color=COLORS["token"], dash='dot'),
              showlegend=False,
              hoverinfo='none'
          ))

  return traces

def create_performance_metrics(num_gpus, experts_per_gpu, tokens_per_gpu, topk, comm_type):
  """Create performance metrics visualization"""
  # Theoretical bandwidth values
  nvlink_bw = 150  # GB/s
  rdma_bw = 50     # GB/s

  # Calculate estimated performance
  total_tokens = num_gpus * tokens_per_gpu
  hidden_size = 4096  # Typical for large models
  token_size_bytes = hidden_size * 2  # BF16 precision

  # Simplified performance model
  if comm_type == "Intranode (NVLink)":
      dispatch_bw = min(nvlink_bw, 0.9 * nvlink_bw * num_gpus / (num_gpus - 1))
      combine_bw = min(nvlink_bw, 0.9 * nvlink_bw * num_gpus / (num_gpus - 1))
  elif comm_type == "Internode (RDMA)":
      dispatch_bw = min(rdma_bw, 0.9 * rdma_bw * num_gpus / (num_gpus - 1))
      combine_bw = min(rdma_bw, 0.9 * rdma_bw * num_gpus / (num_gpus - 1))
  else:  # Mixed
      dispatch_bw = (nvlink_bw + rdma_bw) / 2
      combine_bw = (nvlink_bw + rdma_bw) / 2

  # Calculate latency (microseconds)
  dispatch_latency = 1000 * total_tokens * topk * token_size_bytes / (dispatch_bw * 1e9)
  combine_latency = 1000 * total_tokens * topk * token_size_bytes / (combine_bw * 1e9)

  # Create metrics dataframe
  metrics_df = pd.DataFrame({
      'Operation': ['Dispatch', 'Combine'],
      'Bandwidth (GB/s)': [dispatch_bw, combine_bw],
      'Latency (μs)': [dispatch_latency, combine_latency]
  })

  return metrics_df

# Calculate system layout
gpu_positions, expert_positions, nodes_per_row, rows = create_system_layout(num_gpus, experts_per_gpu)

# Generate token data
token_data = generate_token_data(num_gpus, tokens_per_gpu, experts_per_gpu, topk)
token_positions = get_token_positions(token_data, gpu_positions, tokens_per_gpu)

# Create tabs for different views
tabs = st.tabs(["Visualization", "Token Details", "Performance Metrics"])

# Visualization tab
with tabs[0]:
  if operation == "Complete Flow":
      st.header("Complete MoE Flow Animation")

      # Create containers for animation
      progress_bar = st.progress(0)
      animation_placeholder = st.empty()

      # Control animation
      start_stop = st.button("Start/Stop Animation")

      # Animation state
      if 'animating' not in st.session_state:
          st.session_state.animating = False

      if start_stop:
          st.session_state.animating = not st.session_state.animating

      if st.session_state.animating:
          # Run animation
          for step in range(101):
              # Update progress
              progress = step / 100
              progress_bar.progress(progress)

              # Create fresh figure
              anim_fig = create_system_visualization(
                  gpu_positions, expert_positions, experts_per_gpu,
                  comm_type, nodes_per_row, rows
              )

              # Determine operation phase
              if progress < 0.5:
                  # Dispatch phase (0-50%)
                  dispatch_progress = progress * 2
                  token_traces = create_token_traces(
                      token_data, token_positions, expert_positions,
                      dispatch_progress, "Dispatch"
                  )
                  phase_label = "Dispatch Phase"
              else:
                  # Combine phase (50-100%)
                  combine_progress = (progress - 0.5) * 2
                  token_traces = create_token_traces(
                      token_data, token_positions, expert_positions,
                      combine_progress, "Combine"
                  )
                  phase_label = "Combine Phase"

              # Add token traces to figure
              for trace in token_traces:
                  anim_fig.add_trace(trace)

              # Update title
              anim_fig.update_layout(title=f"MoE Communication - {phase_label} ({int(progress*100)}%)")

              # Display frame
              animation_placeholder.plotly_chart(anim_fig, use_container_width=True)

              # Control speed
              time.sleep(0.05 / animation_speed)

              # Check if animation was stopped
              if not st.session_state.animating:
                  break

          # Reset animation state when finished
          if progress >= 0.99:
              st.session_state.animating = False
      else:
          # Static view (midway through flow)
          static_fig = create_system_visualization(
              gpu_positions, expert_positions, experts_per_gpu,
              comm_type, nodes_per_row, rows
          )

          # Add both dispatch and combine traces
          dispatch_traces = create_token_traces(
              token_data, token_positions, expert_positions,
              0.5, "Dispatch"
          )

          combine_traces = create_token_traces(
              token_data, token_positions, expert_positions,
              0.2, "Combine"
          )

          # Add traces to figure
          for trace in dispatch_traces + combine_traces:
              static_fig.add_trace(trace)

          # Update title
          static_fig.update_layout(title="MoE Communication - Full Flow View")

          # Display static view
          animation_placeholder.plotly_chart(static_fig, use_container_width=True)
  else:
      # Single operation view (Dispatch or Combine)
      st.header(f"{operation} Operation")

      # Container for visualization
      viz_placeholder = st.empty()

      # Animation progress
      progress = st.slider("Animation Progress", 0.0, 1.0, 0.5, 0.01, key="op_progress")

      # Create operation visualization
      op_fig = create_system_visualization(
          gpu_positions, expert_positions, experts_per_gpu,
          comm_type, nodes_per_row, rows
      )

      # Add token traces
      token_traces = create_token_traces(
          token_data, token_positions, expert_positions,
          progress, operation
      )

      # Add traces to figure
      for trace in token_traces:
          op_fig.add_trace(trace)

      # Update title
      op_fig.update_layout(title=f"MoE Communication - {operation} Operation")

      # Display visualization
      viz_placeholder.plotly_chart(op_fig, use_container_width=True)

# Token Details tab
with tabs[1]:
  st.header("Token Routing Information")

  # Select tokens to display
  display_count = min(10, len(token_data))
  selected_tokens = st.multiselect(
      "Select tokens to view details",
      options=list(range(len(token_data))),
      default=list(range(min(5, len(token_data))))
  )

  if not selected_tokens:
      st.info("Select one or more tokens to view their routing details")
  else:
      # Display token details
      for token_idx in selected_tokens:
          token = token_data[token_idx]

          st.subheader(f"Token {token_idx}")

          col1, col2 = st.columns([1, 3])

          with col1:
              st.write(f"Source: GPU {token['source_gpu']}")

          with col2:
              # Create token assignments table
              assignments = []
              for e_idx, expert_id in enumerate(token['assigned_experts']):
                  assignments.append({
                      'Expert': f"E{expert_id}",
                      'GPU': f"GPU {expert_id // experts_per_gpu}",
                      'Weight': f"{token['weights'][e_idx]:.2f}"
                  })

              assignments_df = pd.DataFrame(assignments)
              st.dataframe(assignments_df, hide_index=True)

# Performance Metrics tab
with tabs[2]:
  st.header("Communication Performance Analysis")

  # Calculate performance metrics
  metrics_df = create_performance_metrics(
      num_gpus, experts_per_gpu, tokens_per_gpu, topk, comm_type
  )

  col1, col2 = st.columns([3, 2])

  with col1:
      # Create bandwidth chart
      bw_fig = px.bar(
          metrics_df,
          x='Operation',
          y='Bandwidth (GB/s)',
          color='Operation',
          color_discrete_map={'Dispatch': COLORS['dispatch'], 'Combine': COLORS['combine']},
          text_auto='.1f'
      )

      bw_fig.update_layout(
          title="Estimated Bandwidth Utilization",
          showlegend=False
      )

      st.plotly_chart(bw_fig, use_container_width=True)

      # Create latency chart
      latency_fig = px.bar(
          metrics_df,
          x='Operation',
          y='Latency (μs)',
          color='Operation',
          color_discrete_map={'Dispatch': COLORS['dispatch'], 'Combine': COLORS['combine']},
          text_auto='.1f'
      )

      latency_fig.update_layout(
          title="Estimated Latency",
          showlegend=False
      )

      st.plotly_chart(latency_fig, use_container_width=True)

  with col2:
      # System statistics
      st.subheader("System Statistics")

      total_tokens = num_gpus * tokens_per_gpu
      total_experts = num_gpus * experts_per_gpu
      tokens_per_expert = total_tokens * topk / total_experts

      st.metric("Total GPUs", f"{num_gpus}")
      st.metric("Total Experts", f"{total_experts}")
      st.metric("Total Tokens", f"{total_tokens}")
      st.metric("Average Tokens per Expert", f"{tokens_per_expert:.1f}")

      # Communication type info
      st.subheader("Communication Info")

      if comm_type == "Intranode (NVLink)":
          st.info("Using high-bandwidth NVLink (150 GB/s) for intra-node communication.")
      elif comm_type == "Internode (RDMA)":
          st.info("Using RDMA (50 GB/s) for inter-node communication.")
      else:
          st.info("Using mixed communication: NVLink within nodes, RDMA between nodes.")

# Footer with explanation
st.markdown("---")
st.markdown("""
### About MoE Communication Patterns

This visualizer demonstrates how DeepEP optimizes communication for Mixture-of-Experts (MoE) models:

- **Dispatch Operation**: Routes tokens from their source GPUs to assigned experts across the system
- **Combine Operation**: Returns processed results back to source GPUs with appropriate weighting
- **Communication Challenges**: The all-to-all pattern is particularly challenging to optimize
- **DeepEP Solution**: Specialized kernels for both NVLink and RDMA maximize bandwidth utilization

DeepEP achieves near-theoretical bandwidth limits, enabling efficient scaling of MoE models across
hundreds of experts on multiple GPUs.
""")

st.markdown("---")
st.caption("DeepEP Communication Pattern Visualizer • Made with Streamlit and Plotly")

