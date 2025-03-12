import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

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

st.title("DeepEP: MoE Communication Pattern Visualizer")

# Create sidebar controls
st.sidebar.header("System Configuration")

num_gpus = st.sidebar.slider("Number of GPUs", 2, 16, 4, 2)
experts_per_gpu = st.sidebar.slider("Experts per GPU", 1, 8, 2)
tokens_per_gpu = st.sidebar.slider("Tokens per GPU", 4, 32, 16, 4)
topk = st.sidebar.slider("Top-k Experts", 1, 4, 2)

comm_type = st.sidebar.radio(
  "Communication Type",
  ["Intranode (NVLink)", "Internode (RDMA)", "Mixed"],
  index=2
)

operation = st.sidebar.radio(
  "Operation",
  ["Dispatch", "Combine", "Complete Flow"],
  index=0
)

animation_speed = st.sidebar.slider("Animation Speed", 0.1, 2.0, 1.0, 0.1)

# Function to create GPU system visualization
def create_system_visualization(num_gpus, experts_per_gpu, comm_type):
  fig = go.Figure()

  # Define node positions
  nodes_per_row = min(4, num_gpus)
  rows = (num_gpus + nodes_per_row - 1) // nodes_per_row

  # GPU and expert positions
  gpu_positions = {}
  expert_positions = {}

  # Create GPUs and experts
  for i in range(num_gpus):
      row = i // nodes_per_row
      col = i % nodes_per_row

      # GPU position
      x = col * 4
      y = row * 6
      gpu_positions[i] = (x, y)

      # Add GPU node
      fig.add_trace(go.Scatter(
          x=[x], y=[y],
          mode='markers+text',
          marker=dict(size=30, color='rgba(66, 135, 245, 0.8)'),
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
          gpus_in_row = min(nodes_per_row, num_gpus - row * nodes_per_row)
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
      for i in range(min(nodes_per_row, num_gpus)):
          for row in range(1, rows):
              if row * nodes_per_row + i < num_gpus:
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

  # Layout settings
  fig.update_layout(
      title="MoE System Architecture",
      showlegend=True,
      hovermode='closest',
      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2, (nodes_per_row-1)*4+2]),
      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2, (rows-1)*6+2]),
      height=600
  )

  return fig, gpu_positions, expert_positions

# Generate token data
def generate_token_data(num_gpus, tokens_per_gpu, experts_per_gpu, topk):
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

# Visualize token routing for dispatch or combine
def visualize_token_routing(token_data, gpu_positions, expert_positions, progress, operation_type):
  traces = []

  # For each token in the data
  for i, token in enumerate(token_data):
      source_gpu = token['source_gpu']
      source_x, source_y = gpu_positions[source_gpu]

      # Calculate initial position (distributed around the GPU)
      angle = 2 * np.pi * (i % tokens_per_gpu) / tokens_per_gpu
      radius = 1.0
      token_x = source_x + radius * np.cos(angle)
      token_y = source_y + radius * np.sin(angle)

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

# Create performance metrics visualization
def create_performance_metrics(num_gpus, experts_per_gpu, tokens_per_gpu, topk, comm_type):
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

  # Create figure
  fig = go.Figure()

  # Add bandwidth bars
  fig.add_trace(go.Bar(
      x=['Dispatch', 'Combine'],
      y=[dispatch_bw, combine_bw],
      text=[f"{dispatch_bw:.1f} GB/s", f"{combine_bw:.1f} GB/s"],
      textposition='auto',
      name='Bandwidth',
      marker_color=[COLORS["dispatch"], COLORS["combine"]]
  ))

  # Add latency bars on secondary axis
  fig.add_trace(go.Bar(
      x=['Dispatch Latency', 'Combine Latency'],
      y=[dispatch_latency, combine_latency],
      text=[f"{dispatch_latency:.1f} μs", f"{combine_latency:.1f} μs"],
      textposition='auto',
      name='Latency',
      marker_color='rgba(255, 99, 71, 0.7)',
      yaxis='y2'
  ))

  # Set layout
  fig.update_layout(
      title="Estimated Performance Metrics",
      yaxis=dict(title="Bandwidth (GB/s)"),
      yaxis2=dict(title="Latency (μs)", overlaying='y', side='right'),
      barmode='group',
      height=300
  )

  return fig

# Main visualization
col1, col2 = st.columns([2, 1])

with col1:
  # Create system visualization
  system_fig, gpu_positions, expert_positions = create_system_visualization(
      num_gpus, experts_per_gpu, comm_type
  )

  # Generate token data
  token_data = generate_token_data(num_gpus, tokens_per_gpu, experts_per_gpu, topk)

  # Create animation
  if operation != "Complete Flow":
      # Single operation (dispatch or combine)
      progress = st.slider("Animation Progress", 0.0, 1.0, 0.5, 0.01)

      # Add token traces
      token_traces = visualize_token_routing(
          token_data, gpu_positions, expert_positions,
          progress, "Dispatch" if operation == "Dispatch" else "Combine"
      )

      for trace in token_traces:
          system_fig.add_trace(trace)

  else:
      # Complete flow (auto-animate)
      st.write("Complete flow animation:")
      progress_placeholder = st.empty()
      animation_container = st.empty()

      # Use session state to track animation
      if 'animating' not in st.session_state:
          st.session_state.animating = False

      start_button = st.button("Start Animation")

      if start_button:
          st.session_state.animating = True

      if st.session_state.animating:
          # Create complete flow animation
          for i in range(101):
              progress = i / 100
              progress_placeholder.progress(progress)

              # Create new figure for each frame
              frame_fig, _, _ = create_system_visualization(
                  num_gpus, experts_per_gpu, comm_type
              )

              # Determine which operation to show
              if progress < 0.5:
                  # Dispatch phase (0-50%)
                  dispatch_progress = progress * 2
                  token_traces = visualize_token_routing(
                      token_data, gpu_positions, expert_positions,
                      dispatch_progress, "Dispatch"
                  )
              else:
                  # Combine phase (50-100%)
                  combine_progress = (progress - 0.5) * 2
                  token_traces = visualize_token_routing(
                      token_data, gpu_positions, expert_positions,
                      combine_progress, "Combine"
                  )

              for trace in token_traces:
                  frame_fig.add_trace(trace)

              animation_container.plotly_chart(frame_fig, use_container_width=True)

              # Control animation speed
              time.sleep(0.05 / animation_speed)

          st.session_state.animating = False
      else:
          # Static visualization with both operations
          # Dispatch traces (halfway)
          dispatch_traces = visualize_token_routing(
              token_data, gpu_positions, expert_positions,
              0.5, "Dispatch"
          )

          # Combine traces (starting)
          combine_traces = visualize_token_routing(
              token_data, gpu_positions, expert_positions,
              0.1, "Combine"
          )

          for trace in dispatch_traces + combine_traces:
              system_fig.add_trace(trace)

          st.plotly_chart(system_fig, use_container_width=True)

  if operation != "Complete Flow" or not st.session_state.animating:
      st.plotly_chart(system_fig, use_container_width=True)

with col2:
  # Display token routing information
  st.subheader("Token Routing Information")

  # Create a sample of tokens to display
  display_tokens = min(5, len(token_data))

  for i in range(display_tokens):
      token = token_data[i]
      st.markdown(f"**Token {i}** (from GPU {token['source_gpu']})")

      # Create table of assignments
      df = pd.DataFrame({
          'Expert': [f"E{e}" for e in token['assigned_experts']],
          'Weight': [f"{w:.2f}" for w in token['weights']],
          'GPU': [f"GPU {e // experts_per_gpu}" for e in token['assigned_experts']]
      })
      st.dataframe(df, hide_index=True)

  # Performance metrics
  metrics_fig = create_performance_metrics(
      num_gpus, experts_per_gpu, tokens_per_gpu, topk, comm_type
  )
  st.plotly_chart(metrics_fig, use_container_width=True)

  # Statistics
  st.subheader("Communication Statistics")
  total_tokens = num_gpus * tokens_per_gpu
  total_experts = num_gpus * experts_per_gpu
  tokens_per_expert = total_tokens * topk / total_experts

  st.metric("Total Tokens", f"{total_tokens}")
  st.metric("Total Experts", f"{total_experts}")
  st.metric("Avg. Tokens per Expert", f"{tokens_per_expert:.1f}")

  # Explain what's happening
  st.subheader("What's Happening")
  if operation == "Dispatch":
      st.markdown("""
      **Dispatch Operation**: Tokens are being routed from their source GPUs to their assigned expert 
networks.
      Each token is sent to its top-k experts, weighted by router scores.
      """)
  elif operation == "Combine":
      st.markdown("""
      **Combine Operation**: After expert processing, results are being sent back to the original GPUs.
      The results are weighted and combined to produce the final output for each token.
      """)
  else:
      st.markdown("""
      **Complete Flow**: The full MoE operation cycle is shown, with tokens first dispatched to experts,
      then processed, and finally combined back at their source GPUs.
      """)

# Add explanation section
st.markdown("---")
st.subheader("About MoE Communication Patterns")
st.markdown("""
This visualizer demonstrates the communication patterns in Mixture-of-Experts (MoE) models using DeepEP.

- **Dispatch**: Tokens are routed from source GPUs to their assigned experts
- **Combine**: Expert outputs are sent back to source GPUs and weighted
- **NVLink**: High-bandwidth connection within a node (~150 GB/s)
- **RDMA**: Network connection between nodes (~50 GB/s)

DeepEP optimizes these communication patterns to achieve near-theoretical bandwidth limits,
enabling efficient scaling of MoE models across many GPUs.
""")

# Footer
st.markdown("---")
st.markdown("DeepEP Communication Pattern Visualizer • Built with Streamlit and Plotly")

