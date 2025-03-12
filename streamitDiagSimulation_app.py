import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime
import re
import base64
import io
from PIL import Image

# Page configuration
st.set_page_config(
  page_title="ExpertFlowDiagnostics",
  page_icon="🔍",
  layout="wide",
  initial_sidebar_state="expanded"
)

# Custom CSS
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
      margin-top: 1.5rem;
      margin-bottom: 1rem;
  }
  .diagnostic-header {
      font-size: 1.4rem;
      color: #5c4caf;
      margin-top: 1rem;
      margin-bottom: 0.5rem;
  }
  .info-box {
      background-color: #e8f4f8;
      border-left: 5px solid #4285f4;
      padding: 1rem;
      margin: 1rem 0;
  }
  .warning-box {
      background-color: #fff8e1;
      border-left: 5px solid #ffc107;
      padding: 1rem;
      margin: 1rem 0;
  }
  .success-box {
      background-color: #e8f5e9;
      border-left: 5px solid #4caf50;
      padding: 1rem;
      margin: 1rem 0;
  }
  .error-box {
      background-color: #ffebee;
      border-left: 5px solid #f44336;
      padding: 1rem;
      margin: 1rem 0;
  }
  .metric-card {
      background-color: #ffffff;
      border-radius: 0.5rem;
      box-shadow: 0 0.15rem 1.75rem rgba(0, 0, 0, 0.1);
      padding: 1rem;
      margin: 0.5rem 0;
  }
  .stPlotlyChart {
      border: 1px solid #e0e0e0;
      border-radius: 0.5rem;
      padding: 1rem;
      background-color: white;
  }
</style>
""", unsafe_allow_html=True)

# App state management
if 'active_tab' not in st.session_state:
  st.session_state.active_tab = "log_analysis"
if 'analysis_complete' not in st.session_state:
  st.session_state.analysis_complete = False
if 'parsed_logs' not in st.session_state:
  st.session_state.parsed_logs = None
if 'benchmark_results' not in st.session_state:
  st.session_state.benchmark_results = None
if 'system_info' not in st.session_state:
  st.session_state.system_info = None

# Sidebar
with st.sidebar:
  st.image("https://img.icons8.com/fluency/96/000000/networking-manager.png", width=80)
  st.markdown("# ExpertFlowDiagnostics")
  st.markdown("### Bottleneck Analysis for MoE Systems")

  st.markdown("---")

  # Navigation
  selected_tab = st.radio(
      "Navigation",
      ["Log Analysis", "Communication Benchmark", "Topology Analysis", "Optimization Recommendations"],
      key="nav_tabs"
  )

  if selected_tab == "Log Analysis":
      st.session_state.active_tab = "log_analysis"
  elif selected_tab == "Communication Benchmark":
      st.session_state.active_tab = "benchmark"
  elif selected_tab == "Topology Analysis":
      st.session_state.active_tab = "topology"
  elif selected_tab == "Optimization Recommendations":
      st.session_state.active_tab = "recommendations"

  st.markdown("---")

  # System information form
  with st.expander("System Configuration", expanded=False):
      st.markdown("### Hardware Configuration")

      num_gpus = st.number_input("Number of GPUs", min_value=1, max_value=128, value=8)
      gpus_per_node = st.number_input("GPUs per Node", min_value=1, max_value=16, value=4)
      gpu_type = st.selectbox("GPU Type", ["A100", "H100", "H800", "V100", "T4", "Other"])

      if gpu_type == "Other":
          gpu_type = st.text_input("Specify GPU Model")

      st.markdown("### Network Configuration")

      intranode_conn = st.selectbox("Intranode Connection", ["NVLink", "PCIe Gen4", "PCIe Gen3", "Other"])

      nvlink_gen = None
      if intranode_conn == "NVLink":
          nvlink_gen = st.selectbox("NVLink Generation", ["NVLink 3.0", "NVLink 4.0", "NVLink 5.0"])

      internode_conn = st.selectbox("Internode Connection",
                                   ["InfiniBand HDR", "InfiniBand NDR", "InfiniBand EDR", "RoCE", "Other"])

      if internode_conn == "Other":
          internode_conn = st.text_input("Specify Connection Type")

      rdma_bandwidth = st.number_input("RDMA Bandwidth (GB/s)", min_value=1, max_value=400, value=50)

      # Save system information
      if st.button("Save Configuration"):
          st.session_state.system_info = {
              "num_gpus": num_gpus,
              "gpus_per_node": gpus_per_node,
              "num_nodes": (num_gpus + gpus_per_node - 1) // gpus_per_node,
              "gpu_type": gpu_type,
              "intranode_conn": intranode_conn,
              "nvlink_gen": nvlink_gen,
              "internode_conn": internode_conn,
              "rdma_bandwidth": rdma_bandwidth
          }
          st.success("Configuration saved!")

# Helper functions
def parse_profiling_logs(logs):
  """Parse DeepEP profiling logs to extract timings and metrics"""
  parsed_data = {
      "dispatch": [],
      "combine": [],
      "timestamp": [],
      "operation": [],
      "duration_ms": [],
      "bandwidth_GBs": [],
      "tokens": [],
      "hidden_size": [],
      "gpus_involved": []
  }

  # Regular expressions for log pattern matching
  dispatch_pattern =
r"DISPATCH.*tokens=(\d+).*hidden=(\d+).*gpus=(\d+).*time=(\d+\.\d+)ms.*BW=(\d+\.\d+)GB/s"
  combine_pattern = r"COMBINE.*tokens=(\d+).*hidden=(\d+).*gpus=(\d+).*time=(\d+\.\d+)ms.*BW=(\d+\.\d+)GB/s"

  lines = logs.strip().split('\n')
  for line in lines:
      timestamp_match = re.search(r"\[(.*?)\]", line)
      timestamp = timestamp_match.group(1) if timestamp_match else "Unknown"

      if "DISPATCH" in line:
          match = re.search(dispatch_pattern, line)
          if match:
              tokens, hidden, gpus, time, bw = match.groups()
              parsed_data["dispatch"].append(True)
              parsed_data["combine"].append(False)
              parsed_data["operation"].append("Dispatch")
              parsed_data["tokens"].append(int(tokens))
              parsed_data["hidden_size"].append(int(hidden))
              parsed_data["gpus_involved"].append(int(gpus))
              parsed_data["duration_ms"].append(float(time))
              parsed_data["bandwidth_GBs"].append(float(bw))
              parsed_data["timestamp"].append(timestamp)

      elif "COMBINE" in line:
          match = re.search(combine_pattern, line)
          if match:
              tokens, hidden, gpus, time, bw = match.groups()
              parsed_data["dispatch"].append(False)
              parsed_data["combine"].append(True)
              parsed_data["operation"].append("Combine")
              parsed_data["tokens"].append(int(tokens))
              parsed_data["hidden_size"].append(int(hidden))
              parsed_data["gpus_involved"].append(int(gpus))
              parsed_data["duration_ms"].append(float(time))
              parsed_data["bandwidth_GBs"].append(float(bw))
              parsed_data["timestamp"].append(timestamp)

  return pd.DataFrame(parsed_data)

def analyze_bandwidth_utilization(df, system_info):
  """Analyze bandwidth utilization based on theoretical limits"""
  if system_info is None:
      return None

  # Add theoretical bandwidth column based on operation type and GPUs involved
  df["theoretical_bw"] = df.apply(
      lambda row: get_theoretical_bandwidth(row, system_info),
      axis=1
  )

  # Calculate utilization
  df["utilization"] = (df["bandwidth_GBs"] / df["theoretical_bw"]) * 100

  return df

def get_theoretical_bandwidth(row, system_info):
  """Calculate theoretical bandwidth for the operation"""
  # Determine if operation is intranode or internode
  gpus_involved = row["gpus_involved"]
  gpus_per_node = system_info["gpus_per_node"]

  if gpus_involved <= gpus_per_node:
      # Intranode operation
      if system_info["intranode_conn"] == "NVLink":
          if system_info["nvlink_gen"] == "NVLink 3.0":
              return 150  # GB/s
          elif system_info["nvlink_gen"] == "NVLink 4.0":
              return 200  # GB/s
          elif system_info["nvlink_gen"] == "NVLink 5.0":
              return 250  # GB/s
      elif system_info["intranode_conn"] == "PCIe Gen4":
          return 32  # GB/s
      elif system_info["intranode_conn"] == "PCIe Gen3":
          return 16  # GB/s
      else:
          return 50  # Default fallback
  else:
      # Internode operation
      return system_info["rdma_bandwidth"]

def identify_bottlenecks(df):
  """Identify performance bottlenecks from analyzed data"""
  bottlenecks = []

  # Check overall utilization
  avg_util = df["utilization"].mean()
  if avg_util < 50:
      bottlenecks.append({
          "type": "low_utilization",
          "severity": "high",
          "description": "Low overall bandwidth utilization ({:.1f}%)".format(avg_util),
          "details": "Bandwidth utilization is significantly below potential, indicating communication inefficiency"
      })

  # Check for dispatch vs combine imbalance
  dispatch_df = df[df["operation"] == "Dispatch"]
  combine_df = df[df["operation"] == "Combine"]

  if len(dispatch_df) > 0 and len(combine_df) > 0:
      dispatch_avg = dispatch_df["duration_ms"].mean()
      combine_avg = combine_df["duration_ms"].mean()

      ratio = max(dispatch_avg, combine_avg) / min(dispatch_avg, combine_avg)
      if ratio > 1.5:
          slower_op = "Dispatch" if dispatch_avg > combine_avg else "Combine"
          bottlenecks.append({
              "type": "operation_imbalance",
              "severity": "medium",
              "description": "{} operations are {:.1f}x slower than counterpart".format(slower_op, ratio),
              "details": "Consider tuning {} configuration parameters".format(slower_op.lower())
          })

  # Check for internode performance issues
  internode_ops = df[df["gpus_involved"] > 8]
  if len(internode_ops) > 0:
      internode_util = internode_ops["utilization"].mean()
      if internode_util < 40:
          bottlenecks.append({
              "type": "internode_performance",
              "severity": "high",
              "description": "Poor internode communication efficiency ({:.1f}%)".format(internode_util),
              "details": "RDMA communication is significantly underperforming, check network configuration"
          })

  # Check for potential token distribution issues
  token_std = df["tokens"].std() / df["tokens"].mean()
  if token_std > 0.3:
      bottlenecks.append({
          "type": "load_imbalance",
          "severity": "medium",
          "description": "High variance in token counts between operations",
          "details": "Consider improving load balancing in expert assignment"
      })

  return bottlenecks

def generate_optimization_recommendations(bottlenecks, df, system_info):
  """Generate optimization recommendations based on identified bottlenecks"""
  recommendations = []

  for bottleneck in bottlenecks:
      if bottleneck["type"] == "low_utilization":
          recommendations.append({
              "title": "Increase Communication Efficiency",
              "steps": [
                  "Ensure DeepEP buffer sizes are properly tuned for your model dimensions",
                  "Set appropriate SM allocation with Buffer.set_num_sms(N)",
                  "Experiment with different expert configurations for better parallelism",
                  "Verify that your network stack is properly configured for RDMA"
              ],
              "priority": "High"
          })

      elif bottleneck["type"] == "operation_imbalance":
          slower_op = "Dispatch" if "Dispatch" in bottleneck["description"] else "Combine"
          recommendations.append({
              "title": "Balance {} Performance".format(slower_op),
              "steps": [
                  "Use Buffer.get_{}_config() with auto-tuned values".format(slower_op.lower()),
                  "Adjust token chunking parameters for more efficient transfers",
                  "Ensure appropriate batch sizes for your operation",
                  "Consider reducing hidden dimension for dispatch if using FP8"
              ],
              "priority": "Medium"
          })

      elif bottleneck["type"] == "internode_performance":
          recommendations.append({
              "title": "Optimize Internode Communication",
              "steps": [
                  "Enable IBGDA for direct GPU-initiated RDMA operations",
                  "Configure appropriate QP depth and count for your workload",
                  "Use traffic isolation via InfiniBand Virtual Lanes (set NVSHMEM_IB_SL)",
                  "Consider enabling adaptive routing on your InfiniBand switches",
                  "Verify that your RDMA stack is using GPU Direct RDMA correctly"
              ],
              "priority": "High"
          })

      elif bottleneck["type"] == "load_imbalance":
          recommendations.append({
              "title": "Improve Expert Load Balancing",
              "steps": [
                  "Consider using capacity-based expert routing algorithms",
                  "Implement expert load balancing with token dropping or auxiliary loss",
                  "Try different expert group configurations to distribute load",
                  "Monitor load distribution with get_dispatch_layout() statistics"
              ],
              "priority": "Medium"
          })

  # Add general recommendations if system_info is available
  if system_info:
      if system_info["internode_conn"].startswith("InfiniBand"):
          recommendations.append({
              "title": "InfiniBand Optimizations",
              "steps": [
                  "Ensure NVSHMEM is properly configured with GPU Direct RDMA",
                  "Set NVSHMEM_SYMMETRIC_SIZE to appropriate value for your model",
                  "Configure appropriate GID index for optimal performance",
                  "Consider setting NVSHMEM_IBGDA_NIC_HANDLER=gpu for DeepEP"
              ],
              "priority": "Medium"
          })

      if system_info["intranode_conn"] == "NVLink":
          recommendations.append({
              "title": "NVLink Optimizations",
              "steps": [
                  "Ensure GPUs are directly connected via NVLink (check nvidia-smi topo)",
                  "Use Buffer.set_num_sms() to control resource allocation",
                  "Consider GPU affinity to optimize NUMA access patterns",
                  "Use async_finish=True with appropriate event management for overlapping"
              ],
              "priority": "Medium"
          })

  return recommendations

def simulate_performance_data(system_info, duration=30, interval=1):
  """Generate simulated performance data for demonstration"""
  # Create timestamp range
  timestamps = pd.date_range(
      start=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      periods=duration,
      freq="{}s".format(interval)
  )

  # Base parameters
  num_gpus = system_info["num_gpus"]
  hidden_sizes = [4096, 5120, 7168]
  token_counts = [4096, 8192, 16384]

  # Calculate theoretical bandwidths
  nvlink_bw = 150 if system_info["intranode_conn"] == "NVLink" else 32
  rdma_bw = system_info["rdma_bandwidth"]

  # Create simulated data
  data = []

  for ts in timestamps:
      # Randomly select parameters
      hidden = np.random.choice(hidden_sizes)
      tokens = np.random.choice(token_counts)

      # Dispatch operation
      gpus_involved = np.random.randint(2, num_gpus + 1)
      is_intranode = gpus_involved <= system_info["gpus_per_node"]
      theoretical_bw = nvlink_bw if is_intranode else rdma_bw

      # Add some realistic variation and inefficiency
      efficiency = np.random.uniform(0.7, 0.95) if is_intranode else np.random.uniform(0.5, 0.85)
      actual_bw = theoretical_bw * efficiency

      # Calculate duration based on data size and bandwidth
      data_size_gb = (tokens * hidden * 2) / (1024 * 1024 * 1024)  # 2 bytes per element (BF16)
      duration_ms = (data_size_gb / actual_bw) * 1000

      # Add some random variation
      duration_ms *= np.random.uniform(0.9, 1.1)

      # Add dispatch operation
      data.append({
          "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
          "operation": "Dispatch",
          "dispatch": True,
          "combine": False,
          "tokens": tokens,
          "hidden_size": hidden,
          "gpus_involved": gpus_involved,
          "duration_ms": duration_ms,
          "bandwidth_GBs": actual_bw,
          "theoretical_bw": theoretical_bw,
          "utilization": (actual_bw / theoretical_bw) * 100
      })

      # Combine operation (slightly different parameters)
      gpus_involved = np.random.randint(2, num_gpus + 1)
      is_intranode = gpus_involved <= system_info["gpus_per_node"]
      theoretical_bw = nvlink_bw if is_intranode else rdma_bw

      # Combine often has different efficiency characteristics
      efficiency = np.random.uniform(0.75, 0.98) if is_intranode else np.random.uniform(0.55, 0.88)
      actual_bw = theoretical_bw * efficiency

      # Calculate duration
      duration_ms = (data_size_gb / actual_bw) * 1000
      duration_ms *= np.random.uniform(0.9, 1.1)

      # Add combine operation
      data.append({
          "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
          "operation": "Combine",
          "dispatch": False,
          "combine": True,
          "tokens": tokens,
          "hidden_size": hidden,
          "gpus_involved": gpus_involved,
          "duration_ms": duration_ms,
          "bandwidth_GBs": actual_bw,
          "theoretical_bw": theoretical_bw,
          "utilization": (actual_bw / theoretical_bw) * 100
      })

  return pd.DataFrame(data)

def generate_topology_visualization(system_info):
  """Generate a visualization of the system topology"""
  if not system_info:
      return None

  num_gpus = system_info["num_gpus"]
  gpus_per_node = system_info["gpus_per_node"]
  num_nodes = (num_gpus + gpus_per_node - 1) // gpus_per_node

  # Create figure
  fig = go.Figure()

  # Calculate layout parameters
  node_width = 300
  node_height = 200
  gpu_radius = 30
  node_spacing = 50

  # Calculate nodes arrangement
  nodes_per_row = min(3, num_nodes)
  rows = (num_nodes + nodes_per_row - 1) // nodes_per_row

  # Create nodes
  for node_idx in range(num_nodes):
      row = node_idx // nodes_per_row
      col = node_idx % nodes_per_row

      # Node box position
      node_x = col * (node_width + node_spacing)
      node_y = row * (node_height + node_spacing)

      # Add node box
      fig.add_shape(
          type="rect",
          x0=node_x,
          y0=node_y,
          x1=node_x + node_width,
          y1=node_y + node_height,
          line=dict(color="blue", width=2),
          fillcolor="rgba(135, 206, 250, 0.2)"
      )

      # Add node label
      fig.add_annotation(
          x=node_x + node_width/2,
          y=node_y + node_height - 15,
          text="Node {}".format(node_idx),
          showarrow=False,
          font=dict(size=14, color="blue")
      )

      # Calculate GPUs per node for this specific node
      actual_gpus = min(gpus_per_node, num_gpus - node_idx * gpus_per_node)

      # Determine GPU layout
      gpus_per_row = min(2, actual_gpus)
      gpu_rows = (actual_gpus + gpus_per_row - 1) // gpus_per_row

      # Add GPUs
      for gpu_idx in range(actual_gpus):
          gpu_row = gpu_idx // gpus_per_row
          gpu_col = gpu_idx % gpus_per_row

          # GPU position
          gpu_x = node_x + node_width/(gpus_per_row+1) * (gpu_col + 1)
          gpu_y = node_y + node_height/(gpu_rows+1) * (gpu_row + 1)

          # Add GPU marker
          fig.add_trace(go.Scatter(
              x=[gpu_x],
              y=[gpu_y],
              mode="markers",
              marker=dict(
                  size=gpu_radius,
                  color="green",
                  line=dict(color="darkgreen", width=2)
              ),
              text="GPU {}".format(node_idx * gpus_per_node + gpu_idx),
              hoverinfo="text",
              showlegend=False
          ))

          # Add GPU label
          fig.add_annotation(
              x=gpu_x,
              y=gpu_y,
              text="{}".format(node_idx * gpus_per_node + gpu_idx),
              showarrow=False,
              font=dict(size=12, color="white")
          )

      # Add NVLink connections between GPUs within node
      if system_info["intranode_conn"] == "NVLink" and actual_gpus > 1:
          for i in range(actual_gpus):
              for j in range(i+1, actual_gpus):
                  i_row = i // gpus_per_row
                  i_col = i % gpus_per_row
                  j_row = j // gpus_per_row
                  j_col = j % gpus_per_row

                  i_x = node_x + node_width/(gpus_per_row+1) * (i_col + 1)
                  i_y = node_y + node_height/(gpu_rows+1) * (i_row + 1)
                  j_x = node_x + node_width/(gpus_per_row+1) * (j_col + 1)
                  j_y = node_y + node_height/(gpu_rows+1) * (j_row + 1)

                  fig.add_trace(go.Scatter(
                      x=[i_x, j_x],
                      y=[i_y, j_y],
                      mode="lines",
                      line=dict(color="orange", width=3),
                      hoverinfo="text",
                      text="NVLink",
                      showlegend=(i == 0 and j == 1 and node_idx == 0)
                  ))

  # Add RDMA connections between nodes
  if num_nodes > 1:
      for i in range(num_nodes):
          for j in range(i+1, num_nodes):
              i_row = i // nodes_per_row
              i_col = i % nodes_per_row
              j_row = j // nodes_per_row
              j_col = j % nodes_per_row

              i_x = i_col * (node_width + node_spacing) + node_width/2
              i_y = i_row * (node_height + node_spacing) + node_height/2
              j_x = j_col * (node_width + node_spacing) + node_width/2
              j_y = j_row * (node_height + node_spacing) + node_height/2

              fig.add_trace(go.Scatter(
                  x=[i_x, j_x],
                  y=[i_y, j_y],
                  mode="lines",
                  line=dict(color="red", width=2, dash="dash"),
                  hoverinfo="text",
                  text="{}".format(system_info['internode_conn']),
                  showlegend=(i == 0 and j == 1)
              ))

  # Layout settings
  fig.update_layout(
      title="System Topology: {} GPUs across {} Nodes".format(num_gpus, num_nodes),
      showlegend=True,
      height=max(400, 200 * rows + 100),
      width=max(600, 350 * nodes_per_row),
      margin=dict(l=20, r=20, t=50, b=20),
      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
  )

  return fig

def run_communication_benchmark(system_info, benchmark_params):
  """Simulate running a communication benchmark"""
  if not system_info:
      return None

  results = []

  # Extract benchmark parameters
  hidden_size = benchmark_params["hidden_size"]
  tokens_list = benchmark_params["tokens_list"]
  num_gpus = benchmark_params["num_gpus"]
  operation = benchmark_params["operation"]

  # Theoretical limits
  nvlink_bw = 150 if system_info["intranode_conn"] == "NVLink" else 32
  rdma_bw = system_info["rdma_bandwidth"]

  # Run benchmark for different configurations
  for tokens in tokens_list:
      for num_involved_gpus in range(2, num_gpus + 1, 2):  # Step by 2 for clarity
          is_intranode = num_involved_gpus <= system_info["gpus_per_node"]
          theoretical_bw = nvlink_bw if is_intranode else rdma_bw

          # Simulate realistic benchmark with some variation
          if operation == "Dispatch":
              efficiency = np.random.uniform(0.7, 0.9) if is_intranode else np.random.uniform(0.6, 0.85)
          else:  # Combine
              efficiency = np.random.uniform(0.75, 0.92) if is_intranode else np.random.uniform(0.65, 0.88)

          # Add some consistent patterns
          efficiency *= (1 - 0.01 * (num_involved_gpus - 2))  # Slight decrease with more GPUs
          efficiency *= (1 - 0.02 * (tokens / 16384))  # Slight decrease with more tokens

          actual_bw = theoretical_bw * efficiency

          # Calculate duration
          data_size_gb = (tokens * hidden_size * 2) / (1024 * 1024 * 1024)  # 2 bytes per element (BF16)
          duration_ms = (data_size_gb / actual_bw) * 1000

          # Add some random variation
          duration_ms *= np.random.uniform(0.95, 1.05)

          # Record result
          results.append({
              "operation": operation,
              "tokens": tokens,
              "hidden_size": hidden_size,
              "gpus_involved": num_involved_gpus,
              "intranode": is_intranode,
              "duration_ms": duration_ms,
              "bandwidth_GBs": actual_bw,
              "theoretical_bw": theoretical_bw,
              "utilization": (actual_bw / theoretical_bw) * 100
          })

  # Simulate benchmark progress
  total_benchmarks = len(tokens_list) * (num_gpus // 2)
  progress_bar = st.progress(0)

  for i in range(total_benchmarks):
      progress_bar.progress((i + 1) / total_benchmarks)
      time.sleep(0.2)

  st.success("Benchmark completed!")

  return pd.DataFrame(results)

# Log Analysis Tab
if st.session_state.active_tab == "log_analysis":
  st.markdown('<h1 class="main-header">DeepEP Performance Log Analysis</h1>', unsafe_allow_html=True)

  st.markdown("""
  Upload your DeepEP performance logs to analyze communication patterns and identify bottlenecks.
  The analyzer will extract operation metrics, calculate bandwidth utilization, and provide recommendations.
  """)

  # Log input methods
  input_method = st.radio("Input Method", ["Upload Log File", "Paste Log Content", "Generate Sample Data"])

  if input_method == "Upload Log File":
      uploaded_file = st.file_uploader("Upload DeepEP log file", type=["log", "txt"])

      if uploaded_file is not None:
          log_content = uploaded_file.getvalue().decode("utf-8")
          st.session_state.parsed_logs = parse_profiling_logs(log_content)
          st.session_state.analysis_complete = True

  elif input_method == "Paste Log Content":
      log_content = st.text_area("Paste log content here", height=200)

      if st.button("Analyze Logs") and log_content:
          st.session_state.parsed_logs = parse_profiling_logs(log_content)
          st.session_state.analysis_complete = True

  elif input_method == "Generate Sample Data":
      if st.session_state.system_info is None:
          st.warning("Please configure system information in the sidebar first")
      else:
          if st.button("Generate Sample Data"):
              with st.spinner("Generating sample performance data..."):
                  st.session_state.parsed_logs = simulate_performance_data(st.session_state.system_info)
                  st.session_state.analysis_complete = True

  # Display analysis if available
  if st.session_state.analysis_complete and st.session_state.parsed_logs is not None:
      df = st.session_state.parsed_logs

      if st.session_state.system_info is not None:
          df = analyze_bandwidth_utilization(df, st.session_state.system_info)

      st.markdown('<h2 class="section-header">Performance Analysis Results</h2>', unsafe_allow_html=True)

      # Summary metrics
      col1, col2, col3, col4 = st.columns(4)

      with col1:
          st.metric("Total Operations", len(df))

      with col2:
          avg_bw = df["bandwidth_GBs"].mean()
          st.metric("Avg Bandwidth", "{:.1f} GB/s".format(avg_bw))

      with col3:
          avg_duration = df["duration_ms"].mean()
          st.metric("Avg Latency", "{:.2f} ms".format(avg_duration))

      with col4:
          if "utilization" in df.columns:
              avg_util = df["utilization"].mean()
              st.metric("Avg Utilization", "{:.1f}%".format(avg_util))

      # Detailed metrics
      st.markdown('<h3 class="diagnostic-header">Operation Breakdown</h3>', unsafe_allow_html=True)

      col1, col2 = st.columns([3, 2])

      with col1:
          # Bandwidth chart
          try:
              bandwidth_fig = px.line(
                  df,
                  x=list(range(len(df))),
                  y="bandwidth_GBs",
                  color="operation",
                  labels={"x": "Operation Index", "bandwidth_GBs": "Bandwidth (GB/s)"},
                  title="Bandwidth by Operation"
              )

              if "theoretical_bw" in df.columns:
                  # Add theoretical bandwidth line
                  bandwidth_fig.add_trace(
                      go.Scatter(
                          x=list(range(len(df))),
                          y=df["theoretical_bw"],
                          mode="lines",
                          line=dict(dash="dash", color="gray"),
                          name="Theoretical"
                      )
                  )

              st.plotly_chart(bandwidth_fig, use_container_width=True)
          except Exception as e:
              st.error("Error creating bandwidth chart: {}".format(str(e)))

      with col2:
          # Operation type breakdown
          try:
              op_counts = df["operation"].value_counts()

              op_fig = px.pie(
                  values=op_counts.values,
                  names=op_counts.index,
                  title="Operation Types"
              )

              st.plotly_chart(op_fig, use_container_width=True)
          except Exception as e:
              st.error("Error creating operation chart: {}".format(str(e)))

      # Latency analysis
      st.markdown('<h3 class="diagnostic-header">Latency Analysis</h3>', unsafe_allow_html=True)

      col1, col2 = st.columns(2)

      with col1:
          # Latency by GPU count
          try:
              latency_by_gpu = df.groupby(["operation", "gpus_involved"]).agg({
                  "duration_ms": "mean"
              }).reset_index()

              gpu_latency_fig = px.bar(
                  latency_by_gpu,
                  x="gpus_involved",
                  y="duration_ms",
                  color="operation",
                  labels={"gpus_involved": "GPUs Involved", "duration_ms": "Latency (ms)"},
                  title="Average Latency by GPU Count"
              )

              st.plotly_chart(gpu_latency_fig, use_container_width=True)
          except Exception as e:
              st.error("Error creating GPU latency chart: {}".format(str(e)))

      with col2:
          # Latency by token count - using manual binning to avoid serialization issues
          try:
              # Create token bins manually
              token_ranges = [
                  df["tokens"].min(),
                  df["tokens"].min() + (df["tokens"].max() - df["tokens"].min())/4,
                  df["tokens"].min() + 2*(df["tokens"].max() - df["tokens"].min())/4,
                  df["tokens"].min() + 3*(df["tokens"].max() - df["tokens"].min())/4,
                  df["tokens"].max()
              ]

              token_bins = []
              latency_values = []
              operation_types = []

              for op in df["operation"].unique():
                  op_df = df[df["operation"] == op]

                  for i in range(len(token_ranges)-1):
                      bin_min = token_ranges[i]
                      bin_max = token_ranges[i+1]
                      bin_name = "{}-{}".format(int(bin_min), int(bin_max))

                      # Filter data for this bin
                      bin_data = op_df[(op_df["tokens"] >= bin_min) & (op_df["tokens"] < bin_max)]

                      if len(bin_data) > 0:
                          avg_latency = bin_data["duration_ms"].mean()
                          token_bins.append(bin_name)
                          latency_values.append(avg_latency)
                          operation_types.append(op)

              # Create dataframe from manual binning
              latency_by_tokens = pd.DataFrame({
                  "tokens_bin": token_bins,
                  "duration_ms": latency_values,
                  "operation": operation_types
              })

              token_latency_fig = px.bar(
                  latency_by_tokens,
                  x="tokens_bin",
                  y="duration_ms",
                  color="operation",
                  labels={"tokens_bin": "Token Count", "duration_ms": "Latency (ms)"},
                  title="Average Latency by Token Count"
              )

              st.plotly_chart(token_latency_fig, use_container_width=True)
          except Exception as e:
              st.error("Error creating token latency chart: {}".format(str(e)))

      # Utilization analysis
      if "utilization" in df.columns:
          st.markdown('<h3 class="diagnostic-header">Bandwidth Utilization</h3>', unsafe_allow_html=True)

          col1, col2 = st.columns(2)

          with col1:
              # Utilization histogram
              try:
                  util_fig = px.histogram(
                      df,
                      x="utilization",
                      color="operation",
                      nbins=20,
                      labels={"utilization": "Bandwidth Utilization (%)"},
                      title="Bandwidth Utilization Distribution"
                  )

                  st.plotly_chart(util_fig, use_container_width=True)
              except Exception as e:
                  st.error("Error creating utilization histogram: {}".format(str(e)))

          with col2:
              # Utilization by GPU count
              try:
                  util_by_gpu = df.groupby(["operation", "gpus_involved"]).agg({
                      "utilization": "mean"
                  }).reset_index()

                  util_gpu_fig = px.line(
                      util_by_gpu,
                      x="gpus_involved",
                      y="utilization",
                      color="operation",
                      markers=True,
                      labels={"gpus_involved": "GPUs Involved", "utilization": "Utilization (%)"},
                      title="Bandwidth Utilization by GPU Count"
                  )

                  st.plotly_chart(util_gpu_fig, use_container_width=True)
              except Exception as e:
                  st.error("Error creating utilization by GPU chart: {}".format(str(e)))

      # Raw data
      with st.expander("View Raw Performance Data"):
          st.dataframe(df)

      # Bottleneck identification
      if "utilization" in df.columns:
          st.markdown('<h2 class="section-header">Bottleneck Analysis</h2>', unsafe_allow_html=True)

          bottlenecks = identify_bottlenecks(df)

          if bottlenecks:
              for bottleneck in bottlenecks:
                  severity_color = {
                      "high": "error-box",
                      "medium": "warning-box",
                      "low": "info-box"
                  }.get(bottleneck["severity"], "info-box")

                  st.markdown('<div class="{}">'.format(severity_color), unsafe_allow_html=True)
                  st.markdown("**{}**".format(bottleneck['description']))
                  st.markdown(bottleneck["details"])
                  st.markdown('</div>', unsafe_allow_html=True)
          else:
              st.markdown('<div class="success-box">', unsafe_allow_html=True)
              st.markdown("**No significant bottlenecks detected**")
              st.markdown("The communication performance appears to be efficient based on the analyzed metriics.")
              st.markdown('</div>', unsafe_allow_html=True)

# Communication Benchmark Tab
elif st.session_state.active_tab == "benchmark":
  st.markdown('<h1 class="main-header">DeepEP Communication Benchmark</h1>', unsafe_allow_html=True)

  st.markdown("""
  Benchmark DeepEP communication performance across different configurations to identify optimal settings
  and potential bottlenecks.
  """)

  if st.session_state.system_info is None:
      st.warning("Please configure system information in the sidebar first")
  else:
      # Benchmark configuration
      st.markdown('<h2 class="section-header">Benchmark Configuration</h2>', unsafe_allow_html=True)

      col1, col2 = st.columns(2)

      with col1:
          operation = st.selectbox("Operation", ["Dispatch", "Combine"])
          hidden_size = st.selectbox("Hidden Size", [4096, 5120, 7168, 8192], index=0)

      with col2:
          tokens = st.multiselect(
              "Token Counts",
              options=[1024, 2048, 4096, 8192, 16384, 32768],
              default=[4096, 8192, 16384]
          )

          max_gpus = st.slider("Maximum GPUs", 2, st.session_state.system_info["num_gpus"],
st.session_state.system_info["num_gpus"])

      benchmark_params = {
          "operation": operation,
          "hidden_size": hidden_size,
          "tokens_list": tokens,
          "num_gpus": max_gpus
      }

      if st.button("Run Benchmark"):
          with st.spinner("Running benchmark..."):
              st.session_state.benchmark_results = run_communication_benchmark(
                  st.session_state.system_info,
                  benchmark_params
              )

      # Display benchmark results
      if st.session_state.benchmark_results is not None:
          df = st.session_state.benchmark_results

          st.markdown('<h2 class="section-header">Benchmark Results</h2>', unsafe_allow_html=True)

          # Summary metrics
          col1, col2, col3 = st.columns(3)

          with col1:
              max_bw = df["bandwidth_GBs"].max()
              st.metric("Peak Bandwidth", "{:.1f} GB/s".format(max_bw))

          with col2:
              min_latency = df["duration_ms"].min()
              st.metric("Minimum Latency", "{:.2f} ms".format(min_latency))

          with col3:
              avg_util = df["utilization"].mean()
              st.metric("Avg Utilization", "{:.1f}%".format(avg_util))

          # Bandwidth by configuration
          st.markdown('<h3 class="diagnostic-header">Bandwidth by Configuration</h3>',
unsafe_allow_html=True)

          # Group by token count and GPU count
          try:
              token_tabs = st.tabs(["{} Tokens".format(t) for t in sorted(df["tokens"].unique())])

              for i, token_count in enumerate(sorted(df["tokens"].unique())):
                  with token_tabs[i]:
                      token_df = df[df["tokens"] == token_count]

                      # Bandwidth plot
                      bw_fig = px.bar(
                          token_df,
                          x="gpus_involved",
                          y="bandwidth_GBs",
                          color="intranode",
                          labels={"gpus_involved": "GPUs Involved", "bandwidth_GBs": "Bandwidth (GB/s)"},
                          title="Bandwidth with {} Tokens".format(token_count),
                          color_discrete_map={True: "green", False: "orange"}
                      )

                      # Add theoretical bandwidth line
                      bw_fig.add_trace(
                          go.Scatter(
                              x=token_df["gpus_involved"],
                              y=token_df["theoretical_bw"],
                              mode="lines",
                              line=dict(dash="dash", color="gray"),
                              name="Theoretical"
                          )
                      )

                      st.plotly_chart(bw_fig, use_container_width=True)

                      # Latency plot
                      latency_fig = px.line(
                          token_df,
                          x="gpus_involved",
                          y="duration_ms",
                          markers=True,
                          labels={"gpus_involved": "GPUs Involved", "duration_ms": "Latency (ms)"},
                          title="Latency with {} Tokens".format(token_count)
                      )

                      st.plotly_chart(latency_fig, use_container_width=True)
          except Exception as e:
              st.error("Error creating token tabs: {}".format(str(e)))

          # Utilization heatmap
          st.markdown('<h3 class="diagnostic-header">Bandwidth Utilization</h3>', unsafe_allow_html=True)

          try:
              # Create manual data for heatmap to avoid pivot_table serialization issues
              heatmap_data = []
              for token in sorted(df["tokens"].unique()):
                  for gpu in sorted(df["gpus_involved"].unique()):
                      subset = df[(df["tokens"] == token) & (df["gpus_involved"] == gpu)]
                      if len(subset) > 0:
                          heatmap_data.append({
                              "tokens": str(token),
                              "gpus": str(gpu),
                              "utilization": subset["utilization"].mean()
                          })

              heatmap_df = pd.DataFrame(heatmap_data)

              # Create heatmap
              util_fig = px.density_heatmap(
                  heatmap_df,
                  x="gpus",
                  y="tokens",
                  z="utilization",
                  labels=dict(x="GPUs Involved", y="Tokens", color="Utilization (%)"),
                  color_continuous_scale="RdYlGn",
                  title="Bandwidth Utilization Heatmap"
              )

              st.plotly_chart(util_fig, use_container_width=True)
          except Exception as e:
              st.error("Error creating utilization heatmap: {}".format(str(e)))

          # Raw data
          with st.expander("View Raw Benchmark Data"):
              st.dataframe(df)

          # Optimal configuration
          st.markdown('<h3 class="diagnostic-header">Optimal Configuration</h3>', unsafe_allow_html=True)

          # Find configuration with best throughput
          best_throughput = df.loc[df["bandwidth_GBs"].idxmax()]

          st.markdown('<div class="success-box">', unsafe_allow_html=True)
          st.markdown("""
          **Best Throughput Configuration:**
          - **Tokens:** {}
          - **GPUs:** {}
          - **Bandwidth:** {:.1f} GB/s ({:.1f}% of theoretical)
          - **Latency:** {:.2f} ms
          """.format(
              best_throughput['tokens'],
              best_throughput['gpus_involved'],
              best_throughput['bandwidth_GBs'],
              best_throughput['utilization'],
              best_throughput['duration_ms']
          ))
          st.markdown('</div>', unsafe_allow_html=True)

          # Best latency configuration
          best_latency = df.loc[df["duration_ms"].idxmin()]

          st.markdown('<div class="info-box">', unsafe_allow_html=True)
          st.markdown("""
          **Best Latency Configuration:**
          - **Tokens:** {}
          - **GPUs:** {}
          - **Bandwidth:** {:.1f} GB/s ({:.1f}% of theoretical)
          - **Latency:** {:.2f} ms
          """.format(
              best_latency['tokens'],
              best_latency['gpus_involved'],
              best_latency['bandwidth_GBs'],
              best_latency['utilization'],
              best_latency['duration_ms']
          ))
          st.markdown('</div>', unsafe_allow_html=True)

# Topology Analysis Tab
elif st.session_state.active_tab == "topology":
  st.markdown('<h1 class="main-header">System Topology Analysis</h1>', unsafe_allow_html=True)

  st.markdown("""
  Visualize your system's topology to better understand the communication pathways and potential bottlenecks
  in your expert parallelism setup.
  """)

  if st.session_state.system_info is None:
      st.warning("Please configure system information in the sidebar first")
  else:
      # Generate topology visualization
      st.markdown('<h2 class="section-header">System Topology Visualization</h2>', unsafe_allow_html=True)

      try:
          topo_fig = generate_topology_visualization(st.session_state.system_info)

          if topo_fig:
              st.plotly_chart(topo_fig, use_container_width=True)
      except Exception as e:
          st.error("Error generating topology visualization: {}".format(str(e)))

      # System characteristics
      st.markdown('<h2 class="section-header">Communication Characteristics</h2>', unsafe_allow_html=True)

      col1, col2 = st.columns(2)

      with col1:
          st.markdown('<div class="metric-card">', unsafe_allow_html=True)
          st.markdown("### Intranode Communication")
          st.markdown("**Connection Type:** {}".format(st.session_state.system_info['intranode_conn']))

          if st.session_state.system_info['intranode_conn'] == "NVLink":
              st.markdown("**NVLink Generation:** {}".format(st.session_state.system_info['nvlink_gen']))

              if st.session_state.system_info['nvlink_gen'] == "NVLink 3.0":
                  st.markdown("**Theoretical Bandwidth:** ~150 GB/s")
              elif st.session_state.system_info['nvlink_gen'] == "NVLink 4.0":
                  st.markdown("**Theoretical Bandwidth:** ~200 GB/s")
              elif st.session_state.system_info['nvlink_gen'] == "NVLink 5.0":
                  st.markdown("**Theoretical Bandwidth:** ~250 GB/s")
          elif st.session_state.system_info['intranode_conn'] == "PCIe Gen4":
              st.markdown("**Theoretical Bandwidth:** ~32 GB/s")
          elif st.session_state.system_info['intranode_conn'] == "PCIe Gen3":
              st.markdown("**Theoretical Bandwidth:** ~16 GB/s")

          st.markdown("</div>", unsafe_allow_html=True)

      with col2:
          st.markdown('<div class="metric-card">', unsafe_allow_html=True)
          st.markdown("### Internode Communication")
          st.markdown("**Connection Type:** {}".format(st.session_state.system_info['internode_conn']))
          st.markdown("**Theoretical Bandwidth:** ~{} 
GB/s".format(st.session_state.system_info['rdma_bandwidth']))

          if "InfiniBand" in st.session_state.system_info['internode_conn']:
              st.markdown("**RDMA Technology:** InfiniBand (optimal for DeepEP)")
          elif "RoCE" in st.session_state.system_info['internode_conn']:
              st.markdown("**RDMA Technology:** RoCE (RDMA over Converged Ethernet)")

          st.markdown("</div>", unsafe_allow_html=True)

      # Topology analysis
      st.markdown('<h2 class="section-header">Topology Analysis</h2>', unsafe_allow_html=True)

      num_gpus = st.session_state.system_info["num_gpus"]
      gpus_per_node = st.session_state.system_info["gpus_per_node"]
      num_nodes = (num_gpus + gpus_per_node - 1) // gpus_per_node

      # Full mesh connections analysis
      total_connections = (num_gpus * (num_gpus - 1)) // 2
      intranode_connections = num_nodes * ((gpus_per_node * (gpus_per_node - 1)) // 2)
      internode_connections = total_connections - intranode_connections

      conn_data = pd.DataFrame({
          "Connection Type": ["Intranode (NVLink)", "Internode (RDMA)", "Total"],
          "Count": [intranode_connections, internode_connections, total_connections],
          "Percentage": [
              100 * intranode_connections / total_connections if total_connections > 0 else 0,
              100 * internode_connections / total_connections if total_connections > 0 else 0,
              100
          ]
      })

      try:
          conn_fig = px.pie(
              conn_data,
              values="Count",
              names="Connection Type",
              title="Communication Path Distribution"
          )

          st.plotly_chart(conn_fig, use_container_width=True)
      except Exception as e:
          st.error("Error creating connection distribution chart: {}".format(str(e)))

      # Potential bottlenecks
      st.markdown('<h3 class="diagnostic-header">Potential Topology Bottlenecks</h3>',
unsafe_allow_html=True)

      if internode_connections > intranode_connections:
          st.markdown('<div class="warning-box">', unsafe_allow_html=True)
          st.markdown("""
          **High Internode Communication Ratio**
          
          Your system topology has more internode connections than intranode connections, which could lead to
          communication bottlenecks due to lower RDMA bandwidth compared to NVLink. Consider:
          
          1. Adjusting expert assignments to favor intranode communication
          2. Using capacity-based routing to minimize cross-node traffic
          3. Implementing topology-aware expert sharding
          """)
          st.markdown('</div>', unsafe_allow_html=True)

      if num_nodes > 4:
          st.markdown('<div class="info-box">', unsafe_allow_html=True)
          st.markdown("""
          **Large Node Count**
          
          Your system has a relatively large number of nodes, which increases the complexity of all-to-all
          communication patterns. Consider:
          
          1. Using hierarchical expert grouping to minimize cross-node communication
          2. Implementing node-aware expert routing algorithms
          3. Fine-tuning RDMA parameters for your specific network topology
          """)
          st.markdown('</div>', unsafe_allow_html=True)

      if gpus_per_node > 4 and st.session_state.system_info['intranode_conn'] != "NVLink":
          st.markdown('<div class="error-box">', unsafe_allow_html=True)
          st.markdown("""
          **High GPU Density without NVLink**
          
          Your nodes have multiple GPUs but are not using NVLink for intranode communication.
          This will significantly limit intranode bandwidth and cause bottlenecks. Consider:
          
          1. Using NVLink-enabled hardware for better intranode communication
          2. Reducing experts per GPU to minimize intranode traffic
          """)
          st.markdown('</div>', unsafe_allow_html=True)

# Optimization Recommendations Tab
elif st.session_state.active_tab == "recommendations":
  st.markdown('<h1 class="main-header">DeepEP Optimization Recommendations</h1>', unsafe_allow_html=True)

  st.markdown("""
  Based on the analysis of your performance logs and system configuration, here are tailored recommendations
  to optimize your DeepEP implementation and maximize expert parallelism efficiency.
  """)

  if st.session_state.parsed_logs is not None and "utilization" in st.session_state.parsed_logs.columns:
      df = st.session_state.parsed_logs
      bottlenecks = identify_bottlenecks(df)
      recommendations = generate_optimization_recommendations(bottlenecks, df, st.session_state.system_info)

      if recommendations:
          st.markdown('<h2 class="section-header">Optimization Recommendations</h2>', unsafe_allow_html=True)

          # Group recommendations by priority
          high_priority = [r for r in recommendations if r["priority"] == "High"]
          medium_priority = [r for r in recommendations if r["priority"] == "Medium"]
          low_priority = [r for r in recommendations if r["priority"] == "Low"]

          # Display high priority recommendations
          if high_priority:
              st.markdown("### High Priority Optimizations")

              for rec in high_priority:
                  st.markdown('<div class="error-box">', unsafe_allow_html=True)
                  st.markdown("**{}**".format(rec['title']))
                  st.markdown("Steps:")
                  for step in rec["steps"]:
                      st.markdown("- {}".format(step))
                  st.markdown('</div>', unsafe_allow_html=True)

          # Display medium priority recommendations
          if medium_priority:
              st.markdown("### Medium Priority Optimizations")

              for rec in medium_priority:
                  st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                  st.markdown("**{}**".format(rec['title']))
                  st.markdown("Steps:")
                  for step in rec["steps"]:
                      st.markdown("- {}".format(step))
                  st.markdown('</div>', unsafe_allow_html=True)

          # Display low priority recommendations
          if low_priority:
              st.markdown("### Additional Optimizations")

              for rec in low_priority:
                  st.markdown('<div class="info-box">', unsafe_allow_html=True)
                  st.markdown("**{}**".format(rec['title']))
                  st.markdown("Steps:")
                  for step in rec["steps"]:
                      st.markdown("- {}".format(step))
                  st.markdown('</div>', unsafe_allow_html=True)
      else:
          st.markdown('<div class="success-box">', unsafe_allow_html=True)
          st.markdown("**Your system appears to be well-optimized!**")
          st.markdown("Based on the analyzed data, no significant optimization opportunities were 
identified.")
          st.markdown('</div>', unsafe_allow_html=True)

      # Code snippet examples
      st.markdown('<h2 class="section-header">Implementation Examples</h2>', unsafe_allow_html=True)

      with st.expander("Buffer Configuration Code"):
          st.code("""
# Optimized Buffer Configuration
import torch.distributed as dist
from deep_ep import Buffer

# Set SM allocation for high-throughput kernels
Buffer.set_num_sms(24)  # Adjust based on GPU model and workload

# Get the distributed process group
group = dist.group.WORLD

# Calculate optimal buffer sizes
num_nvl_bytes, num_rdma_bytes = 0, 0
for config in (Buffer.get_dispatch_config(group.size()), Buffer.get_combine_config(group.size())):
  num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_size * 2, group.size()), num_nvl_bytes)
  num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_size * 2, group.size()), num_rdma_bytes)

# Create buffer with optimal sizes
buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
          """)

      with st.expander("Low-Latency Configuration"):
          st.code("""
# Low-Latency Mode Configuration
import os
import torch.distributed as dist
from deep_ep import Buffer

# Environment variables for optimal RDMA performance
os.environ['NVSHMEM_DISABLE_P2P'] = '1'
os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'
os.environ['NVSHMEM_IBGDA_NIC_HANDLER'] = 'gpu'
os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'] = f'{num_experts_per_rank}'
os.environ['NVSHMEM_QP_DEPTH'] = '1024'
os.environ['NVSHMEM_IB_SL'] = '5'  # Set appropriate service level for your fabric

# Calculate buffer size for low-latency mode
rdma_size = Buffer.get_low_latency_rdma_size_hint(
  num_max_dispatch_tokens_per_rank=128,
  hidden=hidden_size,
  num_ranks=group.size(),
  num_experts=num_experts
)

# Create buffer in low-latency mode
buffer = Buffer(
  group, 
  0,  # No NVLink buffer needed for low-latency
  rdma_size,
  low_latency_mode=True,
  num_qps_per_rank=num_experts // group.size()
)
          """)

      with st.expander("Communication-Computation Overlap"):
          st.code("""
# Communication-Computation Overlap Example
import torch
from deep_ep import Buffer

def expert_parallel_forward(hidden_states, router_logits, local_experts, buffer):
  batch_size, seq_len, hidden_size = hidden_states.shape
  
  # Get router scores and top-k experts
  router_probs = torch.softmax(router_logits, dim=-1)
  topk_probs, topk_indices = torch.topk(router_probs, k=2, dim=-1)
  
  # Reshape for routing
  hidden_flat = hidden_states.reshape(-1, hidden_size)
  topk_probs = topk_probs.reshape(-1, 2)
  topk_indices = topk_indices.reshape(-1, 2)
  
  # Get dispatch layout
  num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event = \
      buffer.get_dispatch_layout(topk_indices, num_experts, async_finish=True)
  
  # Dispatch tokens to experts with async_finish for overlapping
  recv_hidden_states, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, 
dispatch_event = \
      buffer.dispatch(
          hidden_flat, 
          topk_idx=topk_indices,
          topk_weights=topk_probs,
          num_tokens_per_rank=num_tokens_per_rank,
          num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
          is_token_in_rank=is_token_in_rank,
          num_tokens_per_expert=num_tokens_per_expert,
          async_finish=True,
          allocate_on_comm_stream=True
      )
  
  # Do some other computation while dispatch is happening
  other_computation_result = some_other_function(hidden_states)
  
  # Process tokens with local experts
  expert_outputs = []
  offset = 0
  for i, expert in enumerate(local_experts):
      num_tokens_for_expert = num_recv_tokens_per_expert_list[i]
      if num_tokens_for_expert == 0:
          continue
      
      expert_input = recv_hidden_states[offset:offset + num_tokens_for_expert]
      offset += num_tokens_for_expert
      
      expert_output = expert(expert_input)
      expert_outputs.append(expert_output)
  
  all_expert_outputs = torch.cat(expert_outputs, dim=0)
  
  # Combine with async_finish
  combined_hidden_states, _, combine_event = buffer.combine(
      all_expert_outputs, 
      handle, 
      topk_weights=recv_topk_weights,
      async_finish=True
  )
  
  # Do more computation while combine is happening
  another_result = another_function(other_computation_result)
  
  # Final output processing after combine is done
  output = combined_hidden_states.reshape(batch_size, seq_len, hidden_size)
  return output, combine_event
          """)
  elif st.session_state.benchmark_results is not None:
      df = st.session_state.benchmark_results

      # Generate recommendations based on benchmark results
      st.markdown('<h2 class="section-header">Benchmark-Based Recommendations</h2>', unsafe_allow_html=True)

      # Find best performing configurations
      best_bw_row = df.loc[df["bandwidth_GBs"].idxmax()]
      best_util_row = df.loc[df["utilization"].idxmax()]

      # Analyze intranode vs internode performance
      intranode_df = df[df["intranode"] == True]
      internode_df = df[df["intranode"] == False]

      intranode_util = intranode_df["utilization"].mean() if len(intranode_df) > 0 else 0
      internode_util = internode_df["utilization"].mean() if len(internode_df) > 0 else 0

      # Generate recommendations
      if intranode_util > internode_util + 20:
          st.markdown('<div class="warning-box">', unsafe_allow_html=True)
          st.markdown("""
          **Optimize Internode Communication**
          
          Your benchmark shows significantly better performance for intranode communication compared to 
internode.
          Consider these optimizations:
          
          1. Verify RDMA configuration and ensure proper GPU Direct RDMA setup
          2. Adjust NVSHMEM parameters for your network topology
          3. Enable IBGDA with `NVSHMEM_IB_ENABLE_IBGDA=1` and `NVSHMEM_IBGDA_NIC_HANDLER=gpu`
          4. Consider using FP8 precision for dispatch to reduce bandwidth requirements
          """)
          st.markdown('</div>', unsafe_allow_html=True)

      # Token size recommendation
      token_util = df.groupby("tokens")["utilization"].mean()
      best_token_size = token_util.idxmax()

      st.markdown('<div class="info-box">', unsafe_allow_html=True)
      st.markdown("""
      **Optimal Token Batch Size**
      
      Based on your benchmark results, a token batch size of **{}** provides the best bandwidth
      utilization ({:.1f}%). Consider:
      
      1. Adjusting your batch size to match this optimal value when possible
      2. For prefilling, aim for larger token counts close to this value
      3. For decoding, consider accumulating multiple requests to reach optimal batch sizes
      """.format(best_token_size, token_util[best_token_size]))
      st.markdown('</div>', unsafe_allow_html=True)

      # GPU count recommendation
      gpu_util = df.groupby("gpus_involved")["utilization"].mean()
      best_small_config = gpu_util.iloc[:len(gpu_util)//2].idxmax() if len(gpu_util) > 1 else
gpu_util.idxmax()

      st.markdown('<div class="success-box">', unsafe_allow_html=True)
      st.markdown("""
      **Optimal EP Group Size**
      
      For best efficiency, consider an expert parallelism group size of **{}** GPUs, which
      achieved {:.1f}% bandwidth utilization. If you need more experts, consider
      using multiple EP groups of this size rather than a single larger group.
      """.format(best_small_config, gpu_util[best_small_config]))
      st.markdown('</div>', unsafe_allow_html=True)

      # Code configuration example
      st.markdown('<h2 class="section-header">Configuration Examples</h2>', unsafe_allow_html=True)

      with st.expander("Optimal Configuration Code"):
          st.code("""
# Optimal DeepEP Configuration Based on Benchmark Results
import torch.distributed as dist
from deep_ep import Buffer

# Set optimal number of SMs based on your GPU model
Buffer.set_num_sms(24)  # Adjust based on your specific GPU

# Define optimal process group size
def create_optimal_process_groups(world_size):
  optimal_group_size = {}
  num_groups = world_size // optimal_group_size
  
  groups = []
  for i in range(num_groups):
      ranks = list(range(i * optimal_group_size, (i + 1) * optimal_group_size))
      groups.append(dist.new_group(ranks))
  
  return groups

# Get process groups
world_size = dist.get_world_size()
ep_groups = create_optimal_process_groups(world_size)
local_ep_group = ep_groups[dist.get_rank() // {}]

# Create buffer with optimal configuration
buffer = Buffer(local_ep_group, num_nvl_bytes, num_rdma_bytes)

# Optimal batch size for highest efficiency
recommended_batch_size = {}
          """.format(best_small_config, best_small_config, best_token_size))

      with st.expander("Performance Monitoring Code"):
          st.code("""
# Performance Monitoring for DeepEP
import time
import torch
import pandas as pd
from deep_ep import Buffer

class DeepEPMonitor:
  def __init__(self):
      self.records = []
  
  def record_operation(self, operation, tokens, hidden_size, gpus_involved, 
                      start_time, end_time, data_size_gb):
      duration_ms = (end_time - start_time) * 1000
      bandwidth_gbs = data_size_gb / ((end_time - start_time))
      
      self.records.append({
          "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
          "operation": operation,
          "tokens": tokens,
          "hidden_size": hidden_size,
          "gpus_involved": gpus_involved,
          "duration_ms": duration_ms,
          "bandwidth_GBs": bandwidth_gbs,
          "data_size_gb": data_size_gb
      })
  
  def get_statistics(self):
      df = pd.DataFrame(self.records)
      if len(df) == 0:
          return None
      
      stats = {
          "avg_bandwidth": df["bandwidth_GBs"].mean(),
          "max_bandwidth": df["bandwidth_GBs"].max(),
          "avg_duration": df["duration_ms"].mean(),
          "min_duration": df["duration_ms"].min(),
          "operation_counts": df["operation"].value_counts().to_dict()
      }
      
      return stats
  
  def reset(self):
      self.records = []

# Usage example
monitor = DeepEPMonitor()

# Wrap DeepEP operations for monitoring
def monitored_dispatch(buffer, x, topk_idx, topk_weights, **kwargs):
  tokens = x.size(0)
  hidden_size = x.size(1)
  gpus_involved = buffer.group.size()
  
  data_size_gb = (tokens * hidden_size * 2) / (1024 * 1024 * 1024)  # BF16
  
  start_time = time.time()
  result = buffer.dispatch(x, topk_idx, topk_weights, **kwargs)
  end_time = time.time()
  
  monitor.record_operation("Dispatch", tokens, hidden_size, gpus_involved, 
                          start_time, end_time, data_size_gb)
  
  return result

# Print periodic performance reports
if len(monitor.records) > 0 and len(monitor.records) % 100 == 0:
  stats = monitor.get_statistics()
  print(f"DeepEP Performance: {stats['avg_bandwidth']:.1f} GB/s avg, " 
        f"{stats['avg_duration']:.2f} ms avg latency")
          """)
  else:
      st.info("Run the Log Analysis or Communication Benchmark first to get personalized recommendations")

      # General recommendations
      st.markdown('<h2 class="section-header">General Optimization Guidelines</h2>', unsafe_allow_html=True)

      with st.expander("Buffer Configuration"):
          st.markdown("""
          ### Optimal Buffer Configuration
          
          1. **SM Allocation**
             - Use `Buffer.set_num_sms(N)` to control GPU resources for communication
             - For H100 GPUs, values between 20-28 often work well
             - Adjust based on your specific workload needs
          
          2. **Buffer Sizing**
             - Use the sizing hint functions to properly size buffers:
               ```python
               num_nvl_bytes = config.get_nvl_buffer_size_hint(hidden_size, group.size())
               num_rdma_bytes = config.get_rdma_buffer_size_hint(hidden_size, group.size())
               ```
             - For low-latency mode, use:
               ```python
               rdma_size = Buffer.get_low_latency_rdma_size_hint(
                   num_max_dispatch_tokens_per_rank, hidden, group.size(), num_experts
               )
               ```
          
          3. **Expert Alignment**
             - Use the `expert_alignment` parameter to optimize memory access patterns
             - Typically should be set to a multiple of 8 or 16
          """)

      with st.expander("Communication Optimizations"):
          st.markdown("""
          ### Communication Performance Tuning
          
          1. **NVLink Optimization**
             - Ensure GPUs within a node are directly connected via NVLink
             - Check topology with `nvidia-smi topo -m`
             - Place frequently communicating experts on the same node
          
          2. **RDMA Tuning**
             - Enable GPU Direct RDMA for best performance
             - Set appropriate environment variables:
               ```
               NVSHMEM_IB_ENABLE_IBGDA=1
               NVSHMEM_IBGDA_NIC_HANDLER=gpu
               NVSHMEM_QP_DEPTH=1024
               ```
             - Use virtual lanes for traffic isolation: `NVSHMEM_IB_SL=<value>`
          
          3. **Precision Controls**
             - Use FP8 for dispatch operations to reduce bandwidth requirements
             - Keep BF16 for combine operations where precision is more important
          
          4. **Overlap Strategies**
             - Use `async_finish=True` with appropriate event management
             - Use hook-based background communication for low-latency mode:
               ```python
               recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(...)
               # Do computation
               hook()  # Finalize receiving
               ```
          """)

      with st.expander("Load Balancing"):
          st.markdown("""
          ### Expert Load Balancing
          
          1. **Capacity-Based Routing**

