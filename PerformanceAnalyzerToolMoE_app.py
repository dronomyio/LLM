import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import json
import io
import base64
from datetime import datetime

# Page configuration
st.set_page_config(
   page_title="MoEPerf Analyzer",
   page_icon="📊",
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
       margin-top: 2rem;
       margin-bottom: 1rem;
   }
   .subsection-header {
       font-size: 1.4rem;
       color: #5c4caf;
       margin-top: 1.5rem;
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
       background-color: white;
       padding: 1rem;
       border-radius: 0.5rem;
       box-shadow: 0 0.15rem 1.75rem rgba(0, 0, 0, 0.1);
       margin: 0.5rem 0;
   }
   .stTabs [data-baseweb="tab-list"] {
       gap: 1px;
   }
   .stTabs [data-baseweb="tab"] {
       height: 4rem;
       white-space: pre-wrap;
       background-color: white;
       border-radius: 4px 4px 0 0;
       gap: 1px;
       padding-top: 10px;
       padding-bottom: 10px;
   }
   .stTabs [aria-selected="true"] {
       background-color: #e8f4f8;
       border-bottom-color: transparent;
   }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
   st.session_state.model_loaded = False
if 'profile_data' not in st.session_state:
   st.session_state.profile_data = None
if 'model_config' not in st.session_state:
   st.session_state.model_config = None
if 'analysis_results' not in st.session_state:
   st.session_state.analysis_results = None

# Helper functions and classes
class MoEModelAnalyzer:
   """Class for analyzing MoE model architecture and performance"""

   def __init__(self, model_config, profile_data=None):
       self.model_config = model_config
       self.profile_data = profile_data
       self.analysis_results = {
           "basic_stats": {},
           "communication": {},
           "computation": {},
           "memory": {},
           "bottlenecks": [],
           "recommendations": []
       }

   def analyze_model_architecture(self):
       """Analyze the basic model architecture"""
       config = self.model_config

       # Basic model stats
       params_per_expert = config.get("expert_size", 0) * config.get("ffn_dim", 0) * 8 / 1e9  # Billions
       total_expert_params = params_per_expert * config.get("num_experts", 0)
       shared_params = config.get("shared_params", 0) / 1e9  # Billions
       total_params = total_expert_params + shared_params

       active_params_per_token = shared_params + params_per_expert * config.get("top_k", 2)
       activation_ratio = active_params_per_token / total_params

       self.analysis_results["basic_stats"] = {
           "total_params": total_params,
           "expert_params": total_expert_params,
           "shared_params": shared_params,
           "active_params_per_token": active_params_per_token,
           "activation_ratio": activation_ratio,
           "experts_per_gpu": config.get("experts_per_gpu", config.get("num_experts", 0) /
config.get("num_gpus", 1)),
           "tokens_per_batch": config.get("batch_size", 0) * config.get("seq_length", 0),
           "moe_layers": config.get("num_moe_layers", 0)
       }

       return self.analysis_results["basic_stats"]

   def analyze_communication(self):
       """Analyze communication patterns and bottlenecks"""
       config = self.model_config
       profile = self.profile_data

       # Calculate theoretical communication overhead
       token_hidden_size = config.get("hidden_size", 0)
       tokens_per_batch = config.get("batch_size", 0) * config.get("seq_length", 0)
       top_k = config.get("top_k", 2)
       precision_bytes = 2  # BF16
       moe_layers = config.get("num_moe_layers", 0)

       # Calculate bytes transferred per batch
       bytes_per_token = token_hidden_size * precision_bytes
       bytes_per_dispatch = tokens_per_batch * bytes_per_token * top_k
       bytes_per_combine = bytes_per_dispatch
       total_bytes_per_moe_layer = bytes_per_dispatch + bytes_per_combine
       total_communication = total_bytes_per_moe_layer * moe_layers

       # Estimate communication time
       num_gpus = config.get("num_gpus", 1)
       nvlink_bw = 900  # GB/s
       rdma_bw = 50     # GB/s

       # Estimate intra-node vs inter-node communication ratio
       gpus_per_node = config.get("gpus_per_node", min(8, num_gpus))
       total_possible_connections = (num_gpus * (num_gpus - 1)) // 2
       intranode_connections = (num_gpus // gpus_per_node) * ((gpus_per_node * (gpus_per_node - 1)) // 2)
       internode_connections = total_possible_connections - intranode_connections

       # Estimate communication ratios (very simplified model)
       intranode_ratio = intranode_connections / total_possible_connections if total_possible_connections > 0 else 0
       internode_ratio = 1 - intranode_ratio

       # Convert to GB for bandwidth calculation
       total_gb = total_communication / (1024 * 1024 * 1024)
       intranode_gb = total_gb * intranode_ratio
       internode_gb = total_gb * internode_ratio

       # Estimate time
       intranode_time = intranode_gb / nvlink_bw * 1000 if nvlink_bw > 0 else 0  # ms
       internode_time = internode_gb / rdma_bw * 1000 if rdma_bw > 0 else 0      # ms
       total_comm_time = intranode_time + internode_time

       # If we have profiling data, use that instead of estimates
       actual_comm_time = None
       if profile and "communication_time" in profile:
           actual_comm_time = profile["communication_time"]
           utilization = actual_comm_time / total_comm_time if total_comm_time > 0 else 0
       else:
           utilization = 0.85  # Assumed utilization
           actual_comm_time = total_comm_time / utilization

       self.analysis_results["communication"] = {
           "bytes_per_token": bytes_per_token,
           "bytes_per_dispatch": bytes_per_dispatch,
           "bytes_per_combine": bytes_per_combine,
           "total_bytes_per_moe_layer": total_bytes_per_moe_layer,
           "total_communication_gb": total_gb,
           "intranode_ratio": intranode_ratio,
           "internode_ratio": internode_ratio,
           "intranode_gb": intranode_gb,
           "internode_gb": internode_gb,
           "estimated_comm_time": total_comm_time,
           "actual_comm_time": actual_comm_time,
           "utilization": utilization
       }

       return self.analysis_results["communication"]

   def analyze_computation(self):
       """Analyze computation patterns and performance"""
       config = self.model_config
       profile = self.profile_data

       # Model parameters
       hidden_size = config.get("hidden_size", 0)
       ffn_dim = config.get("ffn_dim", 0)
       batch_size = config.get("batch_size", 0)
       seq_length = config.get("seq_length", 0)
       num_experts = config.get("num_experts", 0)
       experts_per_gpu = config.get("experts_per_gpu", num_experts / config.get("num_gpus", 1))
       top_k = config.get("top_k", 2)
       moe_layers = config.get("num_moe_layers", 0)

       # FLOPs calculation for one expert forward pass
       # 2 matrix multiplications: hidden→ffn and ffn→hidden
       expert_flops = 2 * batch_size * seq_length * hidden_size * ffn_dim * 2  # *2 for mul+add

       # Only top_k experts per token
       active_ratio = top_k / num_experts
       active_experts_per_token = top_k
       total_active_experts = batch_size * seq_length * active_experts_per_token

       # Total FLOPs for MoE layers
       total_moe_flops = expert_flops * active_ratio * moe_layers

       # GPU utilization and load balance
       avg_experts_per_gpu = num_experts / config.get("num_gpus", 1)

       # Expert loading statistics
       # This is a simplified model; in reality, need trace data
       tokens_per_expert = batch_size * seq_length * top_k / num_experts
       load_imbalance = config.get("load_imbalance_factor", 0.2)  # Default assumption

       # Calculate FLOPS with imbalance
       max_tokens_per_expert = tokens_per_expert * (1 + load_imbalance)
       min_tokens_per_expert = tokens_per_expert * (1 - load_imbalance)

       # Compute utilization
       gpu_tflops = config.get("gpu_tflops", 312)  # H100 default
       total_compute_time_balanced = total_moe_flops / (gpu_tflops * 1e12) * 1000  # ms
       total_compute_time_imbalanced = total_compute_time_balanced * (1 + load_imbalance)

       # If we have profiling data, use actual compute time
       actual_compute_time = None
       if profile and "computation_time" in profile:
           actual_compute_time = profile["computation_time"]
           compute_efficiency = total_compute_time_balanced / actual_compute_time if actual_compute_time > 0 else 0
       else:
           compute_efficiency = 0.8  # Assumed efficiency
           actual_compute_time = total_compute_time_balanced / compute_efficiency

       self.analysis_results["computation"] = {
           "expert_flops": expert_flops,
           "active_ratio": active_ratio,
           "total_active_experts": total_active_experts,
           "total_moe_flops": total_moe_flops,
           "tokens_per_expert": tokens_per_expert,
           "max_tokens_per_expert": max_tokens_per_expert,
           "min_tokens_per_expert": min_tokens_per_expert,
           "load_imbalance": load_imbalance,
           "total_compute_time_balanced": total_compute_time_balanced,
           "total_compute_time_imbalanced": total_compute_time_imbalanced,
           "actual_compute_time": actual_compute_time,
           "compute_efficiency": compute_efficiency
       }

       return self.analysis_results["computation"]

   def analyze_memory(self):
       """Analyze memory usage and footprint"""
       config = self.model_config
       profile = self.profile_data

       # Model parameters
       hidden_size = config.get("hidden_size", 0)
       ffn_dim = config.get("ffn_dim", 0)
       num_experts = config.get("num_experts", 0)
       experts_per_gpu = config.get("experts_per_gpu", num_experts / config.get("num_gpus", 1))

       # Memory for expert parameters
       bytes_per_param = 2  # BF16
       expert_params = 2 * hidden_size * ffn_dim  # weights for two matrices
       expert_memory = expert_params * bytes_per_param

       # Memory per GPU for experts
       expert_memory_per_gpu = expert_memory * experts_per_gpu

       # Memory for activations
       batch_size = config.get("batch_size", 0)
       seq_length = config.get("seq_length", 0)
       tokens_per_gpu = batch_size * seq_length / config.get("num_gpus", 1)

       # Activation memory
       token_hidden_memory = hidden_size * bytes_per_param
       token_ffn_memory = ffn_dim * bytes_per_param

       # Memory per expert during forward pass
       activation_memory_per_token = (token_hidden_memory + token_ffn_memory)

       # Buffer memory for dispatch/combine
       top_k = config.get("top_k", 2)
       dispatch_buffer = tokens_per_gpu * token_hidden_memory * top_k * 2  # *2 for input and output

       # Total memory per GPU
       total_memory_params = expert_memory_per_gpu
       total_memory_activations = tokens_per_gpu * activation_memory_per_token * 2  # *2 for fwd/bwd
       total_memory_buffers = dispatch_buffer
       total_memory = total_memory_params + total_memory_activations + total_memory_buffers

       # Convert to GB
       total_memory_gb = total_memory / (1024 * 1024 * 1024)
       params_memory_gb = total_memory_params / (1024 * 1024 * 1024)
       activations_memory_gb = total_memory_activations / (1024 * 1024 * 1024)
       buffers_memory_gb = total_memory_buffers / (1024 * 1024 * 1024)

       # Memory efficiency (what % of GPU memory is used)
       gpu_memory = config.get("gpu_memory", 80)  # H100 default in GB
       memory_efficiency = total_memory_gb / gpu_memory

       self.analysis_results["memory"] = {
           "expert_params": expert_params,
           "expert_memory": expert_memory,
           "expert_memory_per_gpu": expert_memory_per_gpu,
           "activation_memory_per_token": activation_memory_per_token,
           "dispatch_buffer": dispatch_buffer,
           "total_memory_params_gb": params_memory_gb,
           "total_memory_activations_gb": activations_memory_gb,
           "total_memory_buffers_gb": buffers_memory_gb,
           "total_memory_gb": total_memory_gb,
           "memory_efficiency": memory_efficiency,
       }

       return self.analysis_results["memory"]

   def identify_bottlenecks(self):
       """Identify performance bottlenecks"""
       bottlenecks = []

       # Communication vs Computation
       comm = self.analysis_results.get("communication", {})
       comp = self.analysis_results.get("computation", {})

       if comm and comp:
           comm_time = comm.get("actual_comm_time", 0)
           comp_time = comp.get("actual_compute_time", 0)

           # Is this communication-bound?
           if comm_time > comp_time * 0.8:
               bottlenecks.append({
                   "type": "communication",
                   "severity": "high" if comm_time > comp_time * 1.2 else "medium",
                   "description": "Communication-bound performance",
                   "details": f"Communication time ({comm_time:.2f}ms) is significantly higher than computation time ({comp_time:.2f}ms)",
                   "ratio": comm_time / comp_time if comp_time > 0 else float('inf')
               })

           # Or computation-bound?
           elif comp_time > comm_time * 1.2:
               bottlenecks.append({
                   "type": "computation",
                   "severity": "medium",
                   "description": "Computation-bound performance",
                   "details": f"Computation time ({comp_time:.2f}ms) is significantly higher than communication time ({comm_time:.2f}ms)",
                   "ratio": comp_time / comm_time if comm_time > 0 else float('inf')
               })

       # Check bandwidth utilization
       if comm and comm.get("utilization", 1) < 0.7:
           bottlenecks.append({
               "type": "bandwidth_utilization",
               "severity": "high" if comm.get("utilization", 1) < 0.5 else "medium",
               "description": "Low bandwidth utilization",
               "details": f"Network bandwidth utilization is only {comm.get('utilization', 0)*100:.1f}% of theoretical maximum",
               "utilization": comm.get("utilization", 0)
           })

       # Check expert load imbalance
       if comp and comp.get("load_imbalance", 0) > 0.3:
           bottlenecks.append({
               "type": "load_imbalance",
               "severity": "high" if comp.get("load_imbalance", 0) > 0.5 else "medium",
               "description": "High expert load imbalance",
               "details": f"Expert load varies by {comp.get('load_imbalance', 0)*100:.1f}%, causing underutilization of compute resources",
               "imbalance": comp.get("load_imbalance", 0)
           })

       # Check memory efficiency
       mem = self.analysis_results.get("memory", {})
       if mem and mem.get("memory_efficiency", 0) > 0.9:
           bottlenecks.append({
               "type": "memory",
               "severity": "high" if mem.get("memory_efficiency", 0) > 0.95 else "medium",
               "description": "High memory utilization",
               "details": f"Memory usage ({mem.get('total_memory_gb', 0):.1f} GB) is close to GPU capacity",
               "efficiency": mem.get("memory_efficiency", 0)
           })
       elif mem and mem.get("memory_efficiency", 0) < 0.4:
           bottlenecks.append({
               "type": "memory",
               "severity": "low",
               "description": "Low memory utilization",
               "details": f"Memory usage ({mem.get('total_memory_gb', 0):.1f} GB) is significantly below GPU capacity",
               "efficiency": mem.get("memory_efficiency", 0)
           })

       # Check if RDMA is the bottleneck
       if comm and comm.get("internode_ratio", 0) > 0.7:
           bottlenecks.append({
               "type": "internode_communication",
               "severity": "high",
               "description": "High internode communication ratio",
               "details": f"{comm.get('internode_ratio', 0)*100:.1f}% of communication is over slower RDMA links",
               "ratio": comm.get("internode_ratio", 0)
           })

       self.analysis_results["bottlenecks"] = bottlenecks
       return bottlenecks

   def generate_recommendations(self):
       """Generate optimization recommendations based on analysis"""
       recommendations = []
       bottlenecks = self.analysis_results.get("bottlenecks", [])

       for bottleneck in bottlenecks:
           if bottleneck["type"] == "communication":
               recommendations.append({
                   "category": "Communication",
                   "title": "Optimize Communication Patterns",
                   "priority": bottleneck["severity"],
                   "suggestions": [
                       "Increase expert capacity to reduce number of experts needed",
                       "Use FP8 or other reduced precision for dispatch to reduce bandwidth requirements",
                       "Implement expert sharding to minimize cross-node communication",
                       "Consider using DeepEP with RDMA optimization for expert communication",
                       "Adjust topology to maximize intra-node communication"
                   ]
               })

           elif bottleneck["type"] == "bandwidth_utilization":
               recommendations.append({
                   "category": "Network",
                   "title": "Improve Bandwidth Utilization",
                   "priority": bottleneck["severity"],
                   "suggestions": [
                       "Tune DeepEP buffer sizes for your specific model dimensions",
                       "Set appropriate SM allocation for communication kernels",
                       "Implement background communication-computation overlapping",
                       "Configure proper network settings (NVSHMEM_IB_ENABLE_IBGDA=1, NVSHMEM_IBGDA_NIC_HANDLER=gpu)",
                       "Use async_finish=True with custom event management"
                   ]
               })

           elif bottleneck["type"] == "load_imbalance":
               recommendations.append({
                   "category": "Routing",
                   "title": "Improve Expert Load Balancing",
                   "priority": bottleneck["severity"],
                   "suggestions": [
                       "Implement auxiliary load balancing loss during training",
                       "Use capacity factors in router to account for expert loads",
                       "Consider Expert Choice routing instead of Token Choice",
                       "Implement token dropping strategies for experts over capacity",
                       "Fine-tune router temperature parameter"
                   ]
               })

           elif bottleneck["type"] == "memory" and bottleneck["efficiency"] > 0.9:
               recommendations.append({
                   "category": "Memory",
                   "title": "Optimize Memory Usage",
                   "priority": bottleneck["severity"],
                   "suggestions": [
                       "Reduce expert size or number of experts per GPU",
                       "Use activation checkpointing in non-MoE layers",
                       "Implement gradient accumulation to reduce batch size",
                       "Consider using gradient sharding or ZeRO techniques",
                       "Use mixed precision training to reduce memory footprint"
                   ]
               })

           elif bottleneck["type"] == "internode_communication":
               recommendations.append({
                   "category": "Topology",
                   "title": "Optimize for Internode Communication",
                   "priority": bottleneck["severity"],
                   "suggestions": [
                       "Implement hierarchical expert routing to minimize cross-node traffic",
                       "Group related experts on the same node through topology-aware design",
                       "Consider expert replication for high-use experts",
                       "Use advanced RDMA optimization through DeepEP",
                       "Configure InfiniBand adaptive routing and quality of service"
                   ]
               })

       # General recommendations if no specific bottlenecks
       if not recommendations:
           recommendations.append({
               "category": "General",
               "title": "General Optimization Strategies",
               "priority": "medium",
               "suggestions": [
                   "Implement efficient communication-computation overlapping",
                   "Use DeepEP's specialized communication kernels for MoE operations",
                   "Profile with DeepSpeed or PyTorch Profiler to identify specific bottlenecks",
                   "Explore different batch sizes and sequence lengths for optimal throughput",
                   "Consider different expert configurations and top-k values for better efficiency"
               ]
           })

       self.analysis_results["recommendations"] = recommendations
       return recommendations

   def run_full_analysis(self):
       """Run all analysis steps"""
       self.analyze_model_architecture()
       self.analyze_communication()
       self.analyze_computation()
       self.analyze_memory()
       self.identify_bottlenecks()
       self.generate_recommendations()
       return self.analysis_results

def load_sample_model():
   """Load a sample model for demonstration"""
   return {
       "name": "MoE-1T Sample Model",
       "hidden_size": 4096,
       "ffn_dim": 16384,
       "expert_size": 1.0,  # in million parameters
       "num_experts": 128,
       "experts_per_gpu": 8,
       "top_k": 2,
       "batch_size": 64,
       "seq_length": 2048,
       "num_gpus": 16,
       "gpus_per_node": 8,
       "num_moe_layers": 16,
       "shared_params": 4e9,  # 4B shared parameters
       "gpu_tflops": 312,  # H100 FP16
       "gpu_memory": 80,   # H100 80GB
       "load_imbalance_factor": 0.3
   }

def load_sample_profile():
   """Load sample profiling data for demonstration"""
   return {
       "dispatch_times": [12.5, 13.2, 12.8, 13.5, 12.9],
       "combine_times": [10.8, 11.2, 10.9, 11.5, 11.1],
       "computation_times": [45.2, 46.1, 45.8, 46.5, 45.9],
       "communication_time": 24.3,  # ms
       "computation_time": 45.9,    # ms
       "batch_time": 86.2,          # ms
       "bandwidth_utilization": 0.72,
       "expert_loads": [125, 143, 112, 138, 152, 119, 128, 145, 131, 122, 140, 133, 126, 148, 120, 129]
   }

def format_number(num, precision=2):
   """Format large numbers with K, M, B suffixes"""
   if num is None:
       return "N/A"

   if isinstance(num, str):
       return num

   if abs(num) >= 1e9:
       return f"{num/1e9:.{precision}f}B"
   elif abs(num) >= 1e6:
       return f"{num/1e6:.{precision}f}M"
   elif abs(num) >= 1e3:
       return f"{num/1e3:.{precision}f}K"
   else:
       return f"{num:.{precision}f}"

def create_model_overview_chart(model_config, analysis):
   """Create a visual overview of the model architecture"""
   # Create a Sankey diagram showing parameter distribution
   basic_stats = analysis["basic_stats"]

   # Define nodes
   nodes = [
       {"name": "Total Parameters"},
       {"name": "Expert Parameters"},
       {"name": "Shared Parameters"},
       {"name": "Active Parameters"}
   ]

   # Define links
   links = [
       {
           "source": 0, "target": 1,
           "value": basic_stats["expert_params"],
           "label": f"{format_number(basic_stats['expert_params'])}B Expert"
       },
       {
           "source": 0, "target": 2,
           "value": basic_stats["shared_params"],
           "label": f"{format_number(basic_stats['shared_params'])}B Shared"
       },
       {
           "source": 1, "target": 3,
           "value": basic_stats["expert_params"] * (basic_stats["activation_ratio"] -
(basic_stats["shared_params"] / basic_stats["total_params"])),
           "label": "Active Experts"
       },
       {
           "source": 2, "target": 3,
           "value": basic_stats["shared_params"],
           "label": "Always Active"
       }
   ]

   # Create Sankey diagram
   fig = go.Figure(data=[go.Sankey(
       node=dict(
           pad=15,
           thickness=20,
           line=dict(color="black", width=0.5),
           label=[node["name"] for node in nodes],
           color=["rgba(31, 119, 180, 0.8)", "rgba(255, 127, 14, 0.8)",
                  "rgba(44, 160, 44, 0.8)", "rgba(214, 39, 40, 0.8)"]
       ),
       link=dict(
           source=[link["source"] for link in links],
           target=[link["target"] for link in links],
           value=[link["value"] for link in links],
           label=[link["label"] for link in links],
           color=["rgba(31, 119, 180, 0.4)", "rgba(44, 160, 44, 0.4)",
                  "rgba(255, 127, 14, 0.4)", "rgba(44, 160, 44, 0.4)"]
       )
   )])

   fig.update_layout(
       title_text="Model Parameter Distribution",
       font_size=12,
       height=500
   )

   return fig

def create_performance_breakdown_chart(analysis):
   """Create a chart showing performance breakdown"""
   comm = analysis["communication"]
   comp = analysis["computation"]

   # Create stacked bar chart of time components
   categories = ["Total Batch Time"]
   comm_time = comm.get("actual_comm_time", 0)
   comp_time = comp.get("actual_compute_time", 0)
   other_time = (comm_time + comp_time) * 0.15  # Estimate for other operations
   total_time = comm_time + comp_time + other_time

   fig = go.Figure()

   # Add communication component
   fig.add_trace(go.Bar(
       name="Communication",
       x=categories,
       y=[comm_time],
       marker_color='rgba(55, 83, 109, 0.7)',
       text=[f"{comm_time:.1f}ms"],
       textposition='inside'
   ))

   # Add computation component
   fig.add_trace(go.Bar(
       name="Computation",
       x=categories,
       y=[comp_time],
       marker_color='rgba(26, 118, 255, 0.7)',
       text=[f"{comp_time:.1f}ms"],
       textposition='inside'
   ))

   # Add other component
   fig.add_trace(go.Bar(
       name="Other",
       x=categories,
       y=[other_time],
       marker_color='rgba(177, 194, 214, 0.7)',
       text=[f"{other_time:.1f}ms"],
       textposition='inside'
   ))

   # Update layout
   fig.update_layout(
       title="Performance Breakdown",
       xaxis_title="",
       yaxis_title="Time (ms)",
       barmode='stack',
       height=400,
       uniformtext_minsize=10,
       uniformtext_mode='hide'
   )

   # Add percentage annotations
   annotations = []

   annotations.append(dict(
       x=0, y=total_time + 10,
       text=f"Total: {total_time:.1f}ms",
       showarrow=False,
       font=dict(size=14)
   ))

   annotations.append(dict(
       x=1.2, y=comm_time/2,
       text=f"{comm_time/total_time*100:.1f}%",
       showarrow=False
   ))

   annotations.append(dict(
       x=1.2, y=comm_time + comp_time/2,
       text=f"{comp_time/total_time*100:.1f}%",
       showarrow=False
   ))

   annotations.append(dict(
       x=1.2, y=comm_time + comp_time + other_time/2,
       text=f"{other_time/total_time*100:.1f}%",
       showarrow=False
   ))

   fig.update_layout(annotations=annotations)

   return fig

def create_communication_chart(analysis):
   """Create a chart showing communication breakdown"""
   comm = analysis["communication"]

   # Create pie chart showing intranode vs internode communication
   intranode_gb = comm.get("intranode_gb", 0)
   internode_gb = comm.get("internode_gb", 0)

   labels = ['Intranode (NVLink)', 'Internode (RDMA)']
   values = [intranode_gb, internode_gb]

   fig = go.Figure(data=[go.Pie(
       labels=labels,
       values=values,
       hole=.4,
       marker=dict(colors=['rgba(55, 126, 184, 0.7)', 'rgba(228, 26, 28, 0.7)'])
   )])

   fig.update_layout(
       title="Communication Data Distribution",
       annotations=[dict(
           text=f"{comm.get('total_communication_gb', 0):.2f} GB",
           x=0.5, y=0.5,
           font_size=14,
           showarrow=False
       )],
       height=350
   )

   return fig

def create_memory_chart(analysis):
   """Create a chart showing memory usage breakdown"""
   mem = analysis["memory"]

   # Create pie chart showing memory components
   labels = ['Expert Parameters', 'Activations', 'Communication Buffers']
   values = [
       mem.get("total_memory_params_gb", 0),
       mem.get("total_memory_activations_gb", 0),
       mem.get("total_memory_buffers_gb", 0)
   ]

   fig = go.Figure(data=[go.Pie(
       labels=labels,
       values=values,
       hole=.4,
       marker=dict(colors=['rgba(55, 126, 184, 0.7)', 'rgba(77, 175, 74, 0.7)', 'rgba(152, 78, 163, 0.7)'])
   )])

   fig.update_layout(
       title="GPU Memory Usage Breakdown",
       annotations=[dict(
           text=f"{mem.get('total_memory_gb', 0):.1f} GB",
           x=0.5, y=0.5,
           font_size=14,
           showarrow=False
       )],
       height=350
   )

   return fig

def create_expert_load_chart(profile_data=None):
   """Create a chart showing expert load distribution"""
   if not profile_data or "expert_loads" not in profile_data:
       # Generate random data for demonstration
       expert_loads = np.random.normal(loc=130, scale=20, size=16).astype(int).tolist()
   else:
       expert_loads = profile_data["expert_loads"]

   # Calculate statistics
   avg_load = sum(expert_loads) / len(expert_loads)
   load_imbalance = (max(expert_loads) - min(expert_loads)) / avg_load

   # Create expert IDs
   expert_ids = [f"E{i}" for i in range(len(expert_loads))]

   fig = go.Figure()

   # Add bar chart
   fig.add_trace(go.Bar(
       x=expert_ids,
       y=expert_loads,
       marker_color='rgba(55, 126, 184, 0.7)',
       text=expert_loads,
       textposition='auto'
   ))

   # Add average line
   fig.add_trace(go.Scatter(
       x=expert_ids,
       y=[avg_load] * len(expert_ids),
       mode='lines',
       name='Average',
       line=dict(color='red', width=2, dash='dash')
   ))

   # Update layout
   fig.update_layout(
       title=f"Expert Load Distribution (Imbalance: {load_imbalance:.2f})",
       xaxis_title="Expert ID",
       yaxis_title="Number of Tokens",
       height=400,
       showlegend=True
   )

   return fig

def get_file_download_link(df, filename, text):
   """Generate a link to download a dataframe as CSV"""
   csv = df.to_csv(index=False)
   b64 = base64.b64encode(csv.encode()).decode()
   href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
   return href

def get_json_download_link(data, filename, text):
   """Generate a link to download a dictionary as JSON"""
   json_str = json.dumps(data, indent=2)
   b64 = base64.b64encode(json_str.encode()).decode()
   href = f'<a href="data:file/json;base64,{b64}" download="{filename}">{text}</a>'
   return href

# Main app layout
st.markdown('<h1 class="main-header">MoEPerf: MoE Architecture Performance Analyzer</h1>',
unsafe_allow_html=True)

st.markdown("""
This tool analyzes the performance characteristics of Mixture-of-Experts (MoE) model architectures,
identifying bottlenecks and providing optimization recommendations for distributed training and inference.
""")

# Sidebar with model configuration
with st.sidebar:
   st.header("Model Configuration")

   # Input method selection
   input_method = st.radio(
       "Input Method",
       ["Sample Model", "Manual Configuration", "Upload Config JSON"]
   )

   if input_method == "Sample Model":
       if st.button("Load Sample Model"):
           st.session_state.model_config = load_sample_model()
           st.session_state.profile_data = load_sample_profile()
           st.session_state.model_loaded = True

   elif input_method == "Manual Configuration":
       with st.form("model_config_form"):
           st.subheader("Basic Configuration")
           model_name = st.text_input("Model Name", "My MoE Model")
           hidden_size = st.number_input("Hidden Size", min_value=128, value=4096, step=128)
           ffn_dim = st.number_input("FFN Dimension", min_value=512, value=16384, step=512)

           st.subheader("Expert Configuration")
           num_experts = st.number_input("Number of Experts", min_value=1, value=128)
           top_k = st.number_input("Top-k Experts", min_value=1, value=2)

           st.subheader("Hardware Configuration")
           num_gpus = st.number_input("Number of GPUs", min_value=1, value=16)
           gpus_per_node = st.number_input("GPUs per Node", min_value=1, max_value=16, value=8)

           st.subheader("Training/Inference Setup")
           batch_size = st.number_input("Batch Size", min_value=1, value=64)
           seq_length = st.number_input("Sequence Length", min_value=1, value=2048)
           num_moe_layers = st.number_input("Number of MoE Layers", min_value=1, value=16)

           # Calculate experts per GPU
           experts_per_gpu = num_experts / num_gpus

           # Advanced parameters
           with st.expander("Advanced Parameters"):
               shared_params = st.number_input(
                   "Shared Parameters (billions)",
                   min_value=0.0, value=4.0, step=0.1
               ) * 1e9
               expert_size = st.number_input(
                   "Expert Size Multiplier",
                   min_value=0.1, value=1.0, step=0.1
               )
               gpu_tflops = st.number_input("GPU TFLOPS (FP16)", min_value=1, value=312)
               gpu_memory = st.number_input("GPU Memory (GB)", min_value=1, value=80)
               load_imbalance = st.slider("Load Imbalance Factor", 0.0, 1.0, 0.3)

           submit_button = st.form_submit_button("Configure Model")

           if submit_button:
               st.session_state.model_config = {
                   "name": model_name,
                   "hidden_size": hidden_size,
                   "ffn_dim": ffn_dim,
                   "expert_size": expert_size,
                   "num_experts": num_experts,
                   "experts_per_gpu": experts_per_gpu,
                   "top_k": top_k,
                   "batch_size": batch_size,
                   "seq_length": seq_length,
                   "num_gpus": num_gpus,
                   "gpus_per_node": gpus_per_node,
                   "num_moe_layers": num_moe_layers,
                   "shared_params": shared_params,
                   "gpu_tflops": gpu_tflops,
                   "gpu_memory": gpu_memory,
                   "load_imbalance_factor": load_imbalance
               }
               st.session_state.model_loaded = True

   elif input_method == "Upload Config JSON":
       uploaded_file = st.file_uploader("Upload model configuration", type=["json"])

       if uploaded_file is not None:
           try:
               model_config = json.load(uploaded_file)
               st.session_state.model_config = model_config
               st.session_state.model_loaded = True
               st.success("Model configuration loaded successfully!")
           except Exception as e:
               st.error(f"Error loading model configuration: {e}")

   # Option to upload profiling data
   if st.session_state.model_loaded:
       st.header("Profiling Data (Optional)")

       profiling_method = st.radio(
           "Profiling Data",
           ["Theoretical Estimates", "Sample Profiling Data", "Upload Profiling JSON"]
       )

       if profiling_method == "Sample Profiling Data":
           if st.button("Load Sample Profiling Data"):
               st.session_state.profile_data = load_sample_profile()

       elif profiling_method == "Upload Profiling JSON":
           uploaded_profile = st.file_uploader("Upload profiling data", type=["json"])

           if uploaded_profile is not None:
               try:
                   profile_data = json.load(uploaded_profile)
                   st.session_state.profile_data = profile_data
                   st.success("Profiling data loaded successfully!")
               except Exception as e:
                   st.error(f"Error loading profiling data: {e}")

       else:
           st.session_state.profile_data = None

   # Run analysis button
   if st.session_state.model_loaded:
       if st.button("Run Analysis"):
           with st.spinner("Running performance analysis..."):
               analyzer = MoEModelAnalyzer(st.session_state.model_config, st.session_state.profile_data)
               st.session_state.analysis_results = analyzer.run_full_analysis()
               time.sleep(1)  # For visual effect
               st.sidebar.success("Analysis complete!")

# Main content area
if not st.session_state.model_loaded:
   st.info("Configure your model in the sidebar to begin analysis.")

   # Display info about the tool
   st.markdown('<h2 class="section-header">About MoEPerf Analyzer</h2>', unsafe_allow_html=True)

   col1, col2 = st.columns([3, 2])

   with col1:
       st.markdown("""
       <div class="info-box">
       <p>MoEPerf is a comprehensive performance analysis tool for Mixture-of-Experts (MoE) model 
architectures, focusing on:</p>
       <ul>
           <li><strong>Communication Analysis</strong>: Identifies bottlenecks in token routing between 
experts</li>
           <li><strong>Computation Efficiency</strong>: Analyzes expert utilization and load balancing</li>
           <li><strong>Memory Footprint</strong>: Examines parameter and activation memory requirements</li>
           <li><strong>Optimization Recommendations</strong>: Provides tailored suggestions for your 
model</li>
       </ul>
       </div>
       """, unsafe_allow_html=True)

   with col2:
       st.image("https://i.imgur.com/6RRHMLL.png", width=300)

   st.markdown("""
   ### Key Features

   1. **Architecture Analyzer**: Evaluates model size, expert distribution, and activation patterns
   2. **Performance Estimator**: Calculates theoretical throughput and identifies bottlenecks
   3. **Communication Analyzer**: Examines data movement patterns between experts
   4. **Memory Profiler**: Analyzes GPU memory requirements and efficiency
   5. **Recommendation Engine**: Provides optimization strategies based on analysis
   """)

   st.markdown('<h2 class="section-header">Getting Started</h2>', unsafe_allow_html=True)

   st.markdown("""
   To use this tool:
   1. Configure your model using one of the sidebar input methods
   2. Optionally provide profiling data for more accurate analysis
   3. Click "Run Analysis" to generate performance insights
   4. Explore the detailed analysis and recommendations
   """)

elif st.session_state.analysis_results is None:
   st.info("Click 'Run Analysis' in the sidebar to analyze your model.")

   # Display model configuration summary
   st.markdown('<h2 class="section-header">Model Configuration Summary</h2>', unsafe_allow_html=True)

   model = st.session_state.model_config

   col1, col2, col3 = st.columns(3)

   with col1:
       st.markdown('<div class="metric-card">', unsafe_allow_html=True)
       st.metric("Model Name", model.get("name", "Unnamed Model"))
       st.markdown('</div>', unsafe_allow_html=True)

       st.markdown('<div class="metric-card">', unsafe_allow_html=True)
       st.metric("Number of Experts", model.get("num_experts", 0))
       st.markdown('</div>', unsafe_allow_html=True)

   with col2:
       st.markdown('<div class="metric-card">', unsafe_allow_html=True)
       st.metric("Hidden Size", model.get("hidden_size", 0))
       st.markdown('</div>', unsafe_allow_html=True)

       st.markdown('<div class="metric-card">', unsafe_allow_html=True)
       st.metric("Top-k Experts", model.get("top_k", 0))
       st.markdown('</div>', unsafe_allow_html=True)

   with col3:
       st.markdown('<div class="metric-card">', unsafe_allow_html=True)
       st.metric("Number of GPUs", model.get("num_gpus", 0))
       st.markdown('</div>', unsafe_allow_html=True)

       st.markdown('<div class="metric-card">', unsafe_allow_html=True)
       experts_per_gpu = model.get("experts_per_gpu", model.get("num_experts", 0) / max(1,
model.get("num_gpus", 1)))
       st.metric("Experts per GPU", f"{experts_per_gpu:.1f}")
       st.markdown('</div>', unsafe_allow_html=True)

   # Display table with all configuration parameters
   st.markdown('<h3 class="subsection-header">Detailed Configuration</h3>', unsafe_allow_html=True)

   # Convert dict to dataframe for cleaner display
   model_df = pd.DataFrame([
       {"Parameter": k, "Value": v}
       for k, v in model.items()
   ])

   st.dataframe(model_df, use_container_width=True)

else:
   # Display analysis results
   analysis = st.session_state.analysis_results

   # Download buttons for analysis results
   col1, col2, col3 = st.columns(3)

   with col1:
       model_json_link = get_json_download_link(
           st.session_state.model_config,
           "model_config.json",
           "Download Model Config"
       )
       st.markdown(model_json_link, unsafe_allow_html=True)

   with col2:
       analysis_json_link = get_json_download_link(
           analysis,
           "moe_analysis.json",
           "Download Full Analysis"
       )
       st.markdown(analysis_json_link, unsafe_allow_html=True)

   with col3:
       # Convert recommendations to DataFrame for CSV
       if analysis["recommendations"]:
           recs_df = pd.DataFrame([
               {
                   "Category": rec["category"],
                   "Title": rec["title"],
                   "Priority": rec["priority"],
                   "Suggestion": suggestion
               }
               for rec in analysis["recommendations"]
               for suggestion in rec["suggestions"]
           ])

           recs_link = get_file_download_link(
               recs_df,
               "moe_recommendations.csv",
               "Download Recommendations"
           )
           st.markdown(recs_link, unsafe_allow_html=True)

   # Create tabs for different sections of the analysis
   tabs = st.tabs([
       "Overview", "Communication Analysis", "Computation Analysis",
       "Memory Analysis", "Expert Load Analysis", "Recommendations"
   ])

   # Overview Tab
   with tabs[0]:
       st.markdown('<h2 class="section-header">Model Overview</h2>', unsafe_allow_html=True)

       # Basic model stats
       basic_stats = analysis["basic_stats"]

       col1, col2, col3 = st.columns(3)

       with col1:
           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Total Parameters", f"{basic_stats['total_params']:.2f}B")
           st.markdown('</div>', unsafe_allow_html=True)

           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Expert Parameters", f"{basic_stats['expert_params']:.2f}B")
           st.markdown('</div>', unsafe_allow_html=True)

       with col2:
           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Shared Parameters", f"{basic_stats['shared_params']:.2f}B")
           st.markdown('</div>', unsafe_allow_html=True)

           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Active Params per Token", f"{basic_stats['active_params_per_token']:.2f}B")
           st.markdown('</div>', unsafe_allow_html=True)

       with col3:
           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Activation Ratio", f"{basic_stats['activation_ratio']*100:.1f}%")
           st.markdown('</div>', unsafe_allow_html=True)

           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("MoE Layers", f"{basic_stats['moe_layers']}")
           st.markdown('</div>', unsafe_allow_html=True)

       # Performance summary
       st.markdown('<h3 class="subsection-header">Performance Summary</h3>', unsafe_allow_html=True)

       col1, col2 = st.columns(2)

       with col1:
           # Model architecture diagram
           model_chart = create_model_overview_chart(st.session_state.model_config, analysis)
           st.plotly_chart(model_chart, use_container_width=True)

       with col2:
           # Performance breakdown chart
           perf_chart = create_performance_breakdown_chart(analysis)
           st.plotly_chart(perf_chart, use_container_width=True)

       # Bottlenecks section
       st.markdown('<h3 class="subsection-header">Identified Bottlenecks</h3>', unsafe_allow_html=True)

       if not analysis["bottlenecks"]:
           st.success("No significant bottlenecks identified in your model configuration.")
       else:
           for bottleneck in analysis["bottlenecks"]:
               severity_class = {
                   "high": "error-box",
                   "medium": "warning-box",
                   "low": "info-box"
               }.get(bottleneck["severity"], "info-box")

               st.markdown(f'<div class="{severity_class}">', unsafe_allow_html=True)
               st.markdown(f"**{bottleneck['description']}**")
               st.markdown(bottleneck["details"])
               st.markdown('</div>', unsafe_allow_html=True)

   # Communication Analysis Tab
   with tabs[1]:
       st.markdown('<h2 class="section-header">Communication Analysis</h2>', unsafe_allow_html=True)

       comm = analysis["communication"]

       # Communication metrics
       col1, col2, col3 = st.columns(3)

       with col1:
           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Total Communication", f"{comm['total_communication_gb']:.2f} GB")
           st.markdown('</div>', unsafe_allow_html=True)

           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Communication Time", f"{comm['actual_comm_time']:.2f} ms")
           st.markdown('</div>', unsafe_allow_html=True)

       with col2:
           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Bandwidth Utilization", f"{comm['utilization']*100:.1f}%")
           st.markdown('</div>', unsafe_allow_html=True)

           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Data per MoE Layer", f"{comm['total_bytes_per_moe_layer']/1e9:.2f} GB")
           st.markdown('</div>', unsafe_allow_html=True)

       with col3:
           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Intranode Ratio", f"{comm['intranode_ratio']*100:.1f}%")
           st.markdown('</div>', unsafe_allow_html=True)

           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Internode Ratio", f"{comm['internode_ratio']*100:.1f}%")
           st.markdown('</div>', unsafe_allow_html=True)

       # Communication charts
       st.markdown('<h3 class="subsection-header">Communication Breakdown</h3>', unsafe_allow_html=True)

       comm_chart = create_communication_chart(analysis)
       st.plotly_chart(comm_chart, use_container_width=True)

       # Additional metrics table
       st.markdown('<h3 class="subsection-header">Detailed Communication Metrics</h3>',
unsafe_allow_html=True)

       comm_df = pd.DataFrame([
           {"Metric": "Bytes per Token", "Value": format_number(comm["bytes_per_token"])},
           {"Metric": "Bytes per Dispatch", "Value": format_number(comm["bytes_per_dispatch"])},
           {"Metric": "Bytes per Combine", "Value": format_number(comm["bytes_per_combine"])},
           {"Metric": "Total Bytes per MoE Layer", "Value": format_number(comm["total_bytes_per_moe_layer"])},
           {"Metric": "Intranode Data", "Value": f"{comm['intranode_gb']:.2f} GB"},
           {"Metric": "Internode Data", "Value": f"{comm['internode_gb']:.2f} GB"},
           {"Metric": "Estimated Communication Time", "Value": f"{comm['estimated_comm_time']:.2f} ms"},
           {"Metric": "Actual Communication Time", "Value": f"{comm['actual_comm_time']:.2f} ms"},
           {"Metric": "Bandwidth Utilization", "Value": f"{comm['utilization']*100:.1f}%"}
       ])

       st.dataframe(comm_df, use_container_width=True)

       # Communication optimization tips
       st.markdown('<h3 class="subsection-header">Optimization Tips</h3>', unsafe_allow_html=True)

       if comm['utilization'] < 0.7:
           st.markdown('<div class="warning-box">', unsafe_allow_html=True)
           st.markdown("""
           **Bandwidth Utilization Opportunity**
           
           Your model is only utilizing {:.1f}% of the available network bandwidth. Consider:
           
           1. Using DeepEP's specialized communication kernels for MoE operations
           2. Setting appropriate buffer sizes and SM allocation for your model dimensions
           3. Implementing communication-computation overlapping with async_finish=True
           4. Using FP8 precision for dispatch operations to reduce bandwidth requirements
           """.format(comm['utilization']*100))
           st.markdown('</div>', unsafe_allow_html=True)

       if comm['internode_ratio'] > 0.6:
           st.markdown('<div class="warning-box">', unsafe_allow_html=True)
           st.markdown("""
           **High Internode Communication**
           
           {:.1f}% of your communication is over slower internode (RDMA) links. Consider:
           
           1. Implementing expert sharding to increase locality of token routing
           2. Using hierarchical routing to minimize cross-node communication
           3. Group frequently co-activated experts on the same node
           4. Enable GPU Direct RDMA optimizations for better performance
           """.format(comm['internode_ratio']*100))
           st.markdown('</div>', unsafe_allow_html=True)

   # Computation Analysis Tab
   with tabs[2]:
       st.markdown('<h2 class="section-header">Computation Analysis</h2>', unsafe_allow_html=True)

       comp = analysis["computation"]

       # Computation metrics
       col1, col2, col3 = st.columns(3)

       with col1:
           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Computation Time", f"{comp['actual_compute_time']:.2f} ms")
           st.markdown('</div>', unsafe_allow_html=True)

           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("MoE Layer FLOPs", f"{comp['total_moe_flops']/1e12:.2f} TFLOPs")
           st.markdown('</div>', unsafe_allow_html=True)

       with col2:
           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Active Experts Ratio", f"{comp['active_ratio']*100:.1f}%")
           st.markdown('</div>', unsafe_allow_html=True)

           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Load Imbalance", f"{comp['load_imbalance']*100:.1f}%")
           st.markdown('</div>', unsafe_allow_html=True)

       with col3:
           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Total Active Experts", f"{comp['total_active_experts']}")
           st.markdown('</div>', unsafe_allow_html=True)

           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Compute Efficiency", f"{comp['compute_efficiency']*100:.1f}%")
           st.markdown('</div>', unsafe_allow_html=True)

       # Expert utilization chart
       st.markdown('<h3 class="subsection-header">Expert Capacity Analysis</h3>', unsafe_allow_html=True)

       col1, col2 = st.columns(2)

       with col1:
           # Create load capacity chart
           load_labels = ['Minimum Load', 'Average Load', 'Maximum Load']
           load_values = [
               comp['min_tokens_per_expert'],
               comp['tokens_per_expert'],
               comp['max_tokens_per_expert']
           ]

           load_fig = go.Figure(data=[
               go.Bar(
                   x=load_labels,
                   y=load_values,
                   marker_color=['rgba(55, 126, 184, 0.7)', 'rgba(77, 175, 74, 0.7)', 'rgba(228, 26, 28, 0.7)'],
                   text=[f"{val:.1f}" for val in load_values],
                   textposition='auto'
               )
           ])

           load_fig.update_layout(
               title="Token Distribution per Expert",
               xaxis_title="Load Level",
               yaxis_title="Tokens per Expert",
               height=400
           )

           st.plotly_chart(load_fig, use_container_width=True)

       with col2:
           # Create computation time chart
           time_labels = ['Balanced', 'Imbalanced', 'Actual']
           time_values = [
               comp['total_compute_time_balanced'],
               comp['total_compute_time_imbalanced'],
               comp['actual_compute_time']
           ]

           time_fig = go.Figure(data=[
               go.Bar(
                   x=time_labels,
                   y=time_values,
                   marker_color=['rgba(55, 126, 184, 0.7)', 'rgba(228, 26, 28, 0.7)', 'rgba(77, 175, 74, 0.7)'],
                   text=[f"{val:.2f} ms" for val in time_values],
                   textposition='auto'
               )
           ])

           time_fig.update_layout(
               title="Computation Time Impact of Load Imbalance",
               xaxis_title="Scenario",
               yaxis_title="Time (ms)",
               height=400
           )

           st.plotly_chart(time_fig, use_container_width=True)

       # Additional metrics table
       st.markdown('<h3 class="subsection-header">Detailed Computation Metrics</h3>', unsafe_allow_html=True)

       comp_df = pd.DataFrame([
           {"Metric": "Expert FLOPs (per layer)", "Value": format_number(comp["expert_flops"])},
           {"Metric": "Active Ratio (top-k / num_experts)", "Value": f"{comp['active_ratio']*100:.1f}%"},
           {"Metric": "Total Active Experts", "Value": format_number(comp["total_active_experts"])},
           {"Metric": "Total MoE FLOPs", "Value": f"{comp['total_moe_flops']/1e12:.2f} TFLOPs"},
           {"Metric": "Average Tokens per Expert", "Value": f"{comp['tokens_per_expert']:.1f}"},
           {"Metric": "Maximum Tokens per Expert", "Value": f"{comp['max_tokens_per_expert']:.1f}"},
           {"Metric": "Minimum Tokens per Expert", "Value": f"{comp['min_tokens_per_expert']:.1f}"},
           {"Metric": "Load Imbalance Factor", "Value": f"{comp['load_imbalance']*100:.1f}%"},
           {"Metric": "Balanced Compute Time", "Value": f"{comp['total_compute_time_balanced']:.2f} ms"},
           {"Metric": "Imbalanced Compute Time", "Value": f"{comp['total_compute_time_imbalanced']:.2f} ms"},
           {"Metric": "Actual Compute Time", "Value": f"{comp['actual_compute_time']:.2f} ms"},
           {"Metric": "Compute Efficiency", "Value": f"{comp['compute_efficiency']*100:.1f}%"}
       ])

       st.dataframe(comp_df, use_container_width=True)

       # Computation optimization tips
       st.markdown('<h3 class="subsection-header">Optimization Tips</h3>', unsafe_allow_html=True)

       if comp['load_imbalance'] > 0.2:
           st.markdown('<div class="warning-box">', unsafe_allow_html=True)
           st.markdown("""
           **Expert Load Imbalance Detected**
           
           Your model has a load imbalance of {:.1f}%, which reduces compute efficiency. Consider:
           
           1. Implementing auxiliary load balancing loss during training
           2. Using capacity factors to balance expert assignment
           3. Implementing expert choice routing instead of token choice
           4. Fine-tuning router temperature parameter
           """.format(comp['load_imbalance']*100))
           st.markdown('</div>', unsafe_allow_html=True)

       if comp['compute_efficiency'] < 0.75:
           st.markdown('<div class="warning-box">', unsafe_allow_html=True)
           st.markdown("""
           **Low Compute Efficiency**
           
           Your compute efficiency is only {:.1f}%. Consider:
           
           1. Using communication-computation overlapping techniques
           2. Adjusting batch size and sequence length for better GPU utilization
           3. Profiling with PyTorch Profiler to identify specific computation bottlenecks
           4. Checking for GPU frequency throttling or other hardware limitations
           """.format(comp['compute_efficiency']*100))
           st.markdown('</div>', unsafe_allow_html=True)

   # Memory Analysis Tab
   with tabs[3]:
       st.markdown('<h2 class="section-header">Memory Analysis</h2>', unsafe_allow_html=True)

       mem = analysis["memory"]

       # Memory metrics
       col1, col2, col3 = st.columns(3)

       with col1:
           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Total Memory", f"{mem['total_memory_gb']:.1f} GB")
           st.markdown('</div>', unsafe_allow_html=True)

           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Parameter Memory", f"{mem['total_memory_params_gb']:.1f} GB")
           st.markdown('</div>', unsafe_allow_html=True)

       with col2:
           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Activation Memory", f"{mem['total_memory_activations_gb']:.1f} GB")
           st.markdown('</div>', unsafe_allow_html=True)

           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Buffer Memory", f"{mem['total_memory_buffers_gb']:.1f} GB")
           st.markdown('</div>', unsafe_allow_html=True)

       with col3:
           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Memory Efficiency", f"{mem['memory_efficiency']*100:.1f}%")
           st.markdown('</div>', unsafe_allow_html=True)

           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           expert_params = format_number(mem["expert_params"])
           st.metric("Parameters per Expert", expert_params)
           st.markdown('</div>', unsafe_allow_html=True)

       # Memory breakdown chart
       st.markdown('<h3 class="subsection-header">Memory Usage Breakdown</h3>', unsafe_allow_html=True)

       mem_chart = create_memory_chart(analysis)
       st.plotly_chart(mem_chart, use_container_width=True)

       # Memory scaling analysis
       st.markdown('<h3 class="subsection-header">Memory Scaling Analysis</h3>', unsafe_allow_html=True)

       col1, col2 = st.columns(2)

       with col1:
           # Create memory scaling with batch size chart
           batch_sizes = [16, 32, 64, 128, 256]
           mem_by_batch = []

           for bs in batch_sizes:
               # Scale activation and buffer memory
               bs_ratio = bs / st.session_state.model_config["batch_size"]
               act_mem = mem["total_memory_activations_gb"] * bs_ratio
               buf_mem = mem["total_memory_buffers_gb"] * bs_ratio
               total_mem = mem["total_memory_params_gb"] + act_mem + buf_mem
               mem_by_batch.append(total_mem)

           batch_fig = go.Figure(data=[
               go.Bar(
                   x=[str(bs) for bs in batch_sizes],
                   y=mem_by_batch,
                   marker_color='rgba(55, 126, 184, 0.7)',
                   text=[f"{val:.1f} GB" for val in mem_by_batch],
                   textposition='auto'
               )
           ])

           # Add GPU memory limit line
           gpu_memory = st.session_state.model_config.get("gpu_memory", 80)
           batch_fig.add_shape(
               type="line",
               x0=0,
               y0=gpu_memory,
               x1=len(batch_sizes) - 1,
               y1=gpu_memory,
               line=dict(
                   color="red",
                   width=2,
                   dash="dash",
               )
           )

           batch_fig.add_annotation(
               x=len(batch_sizes) - 1,
               y=gpu_memory,
               text=f"GPU Memory Limit ({gpu_memory} GB)",
               showarrow=False,
               yshift=10
           )

           batch_fig.update_layout(
               title="Memory Scaling with Batch Size",
               xaxis_title="Batch Size",
               yaxis_title="Memory (GB)",
               height=400
           )

           st.plotly_chart(batch_fig, use_container_width=True)

       with col2:
           # Create memory scaling with number of experts
           expert_counts = [32, 64, 128, 256, 512]
           mem_by_experts = []

           for ec in expert_counts:
               # Scale expert memory
               expert_ratio = ec / st.session_state.model_config["num_experts"]
               experts_per_gpu = ec / st.session_state.model_config["num_gpus"]
               expert_mem = mem["expert_memory"] * experts_per_gpu / (1024 * 1024 * 1024)
               total_mem = expert_mem + mem["total_memory_activations_gb"] + mem["total_memory_buffers_gb"]
               mem_by_experts.append(total_mem)

           expert_fig = go.Figure(data=[
               go.Bar(
                   x=[str(ec) for ec in expert_counts],
                   y=mem_by_experts,
                   marker_color='rgba(77, 175, 74, 0.7)',
                   text=[f"{val:.1f} GB" for val in mem_by_experts],
                   textposition='auto'
               )
           ])

           # Add GPU memory limit line
           gpu_memory = st.session_state.model_config.get("gpu_memory", 80)
           expert_fig.add_shape(
               type="line",
               x0=0,
               y0=gpu_memory,
               x1=len(expert_counts) - 1,
               y1=gpu_memory,
               line=dict(
                   color="red",
                   width=2,
                   dash="dash",
               )
           )

           expert_fig.add_annotation(
               x=len(expert_counts) - 1,
               y=gpu_memory,
               text=f"GPU Memory Limit ({gpu_memory} GB)",
               showarrow=False,
               yshift=10
           )

           expert_fig.update_layout(
               title="Memory Scaling with Number of Experts",
               xaxis_title="Number of Experts",
               yaxis_title="Memory (GB)",
               height=400
           )

           st.plotly_chart(expert_fig, use_container_width=True)

       # Additional metrics table
       st.markdown('<h3 class="subsection-header">Detailed Memory Metrics</h3>', unsafe_allow_html=True)

       mem_df = pd.DataFrame([
           {"Metric": "Expert Parameters", "Value": format_number(mem["expert_params"])},
           {"Metric": "Expert Memory", "Value": format_number(mem["expert_memory"]) + " bytes"},
           {"Metric": "Expert Memory per GPU", "Value": format_number(mem["expert_memory_per_gpu"]) + " bytes"},
           {"Metric": "Activation Memory per Token", "Value": format_number(mem["activation_memory_per_token"]) + " bytes"},
           {"Metric": "Dispatch Buffer Size", "Value": format_number(mem["dispatch_buffer"]) + " bytes"},
           {"Metric": "Parameter Memory", "Value": f"{mem['total_memory_params_gb']:.2f} GB"},
           {"Metric": "Activation Memory", "Value": f"{mem['total_memory_activations_gb']:.2f} GB"},
           {"Metric": "Buffer Memory", "Value": f"{mem['total_memory_buffers_gb']:.2f} GB"},
           {"Metric": "Total Memory", "Value": f"{mem['total_memory_gb']:.2f} GB"},
           {"Metric": "Memory Efficiency", "Value": f"{mem['memory_efficiency']*100:.1f}%"}
       ])

       st.dataframe(mem_df, use_container_width=True)

       # Memory optimization tips
       st.markdown('<h3 class="subsection-header">Optimization Tips</h3>', unsafe_allow_html=True)

       if mem['memory_efficiency'] > 0.9:
           st.markdown('<div class="warning-box">', unsafe_allow_html=True)
           st.markdown("""**High Memory Utilization**
           
           Your model is using {:.1f}% of available GPU memory. Consider:
           
           1. Reducing batch size or sequence length
           2. Using gradient checkpointing for activations
           3. Implementing expert sharding across GPUs
           4. Using lower precision (FP8) for dispatch operations
           """.format(mem['memory_efficiency']*100))
           st.markdown('</div>', unsafe_allow_html=True)

       if mem['total_memory_buffers_gb'] > mem['total_memory_gb'] * 0.3:
           st.markdown('<div class="info-box">', unsafe_allow_html=True)
           st.markdown("""
           **High Buffer Memory Usage**
           
           Communication buffers are using {:.1f}% of total memory. Consider:
           
           1. Optimizing buffer sizes for your specific workload
           2. Using FP8 precision for dispatch to reduce buffer requirements
           3. Implementing more efficient buffer reuse strategies
           """.format(mem['total_memory_buffers_gb'] / mem['total_memory_gb'] * 100))
           #st.markdown('</div>', unsafe_allow_html



           st.markdown("""
           **High Buffer Memory Usage**
           
           Communication buffers are using {:.1f}% of total memory. Consider:
           
           1. Optimizing buffer sizes for your specific workload
           2. Using FP8 precision for dispatch to reduce buffer requirements
           3. Implementing more efficient buffer reuse strategies
           """.format(mem['total_memory_buffers_gb'] / mem['total_memory_gb'] * 100))
           st.markdown('</div>', unsafe_allow_html=True)

   # Expert Load Analysis Tab
   with tabs[4]:
       st.markdown('<h2 class="section-header">Expert Load Analysis</h2>', unsafe_allow_html=True)

       # Expert load metrics
       comp = analysis["computation"]

       col1, col2, col3 = st.columns(3)

       with col1:
           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Avg Tokens per Expert", f"{comp['tokens_per_expert']:.1f}")
           st.markdown('</div>', unsafe_allow_html=True)

           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Max Tokens per Expert", f"{comp['max_tokens_per_expert']:.1f}")
           st.markdown('</div>', unsafe_allow_html=True)

       with col2:
           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Min Tokens per Expert", f"{comp['min_tokens_per_expert']:.1f}")
           st.markdown('</div>', unsafe_allow_html=True)

           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Load Imbalance", f"{comp['load_imbalance']*100:.1f}%")
           st.markdown('</div>', unsafe_allow_html=True)

       with col3:
           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           st.metric("Active Experts Ratio", f"{comp['active_ratio']*100:.1f}%")
           st.markdown('</div>', unsafe_allow_html=True)

           st.markdown('<div class="metric-card">', unsafe_allow_html=True)
           basic_stats = analysis["basic_stats"]
           st.metric("Experts per GPU", f"{basic_stats['experts_per_gpu']:.1f}")
           st.markdown('</div>', unsafe_allow_html=True)

       # Expert load distribution
       st.markdown('<h3 class="subsection-header">Expert Load Distribution</h3>', unsafe_allow_html=True)

       # Create expert load chart (using either profiling data or simulation)
       expert_load_fig = create_expert_load_chart(st.session_state.profile_data)
       st.plotly_chart(expert_load_fig, use_container_width=True)

       # Create expert utilization histogram
       if st.session_state.profile_data and "expert_loads" in st.session_state.profile_data:
           expert_loads = st.session_state.profile_data["expert_loads"]
       else:
           # Generate random data for demonstration
           expert_loads = np.random.normal(loc=130, scale=20, size=16).astype(int).tolist()

       # Create histogram of expert loads
       hist_fig = px.histogram(
           x=expert_loads,
           nbins=10,
           labels={"x": "Tokens per Expert"},
           title="Expert Load Distribution Histogram"
       )

       hist_fig.update_layout(
           xaxis_title="Tokens per Expert",
           yaxis_title="Number of Experts",
           height=400
       )

       st.plotly_chart(hist_fig, use_container_width=True)

       # Expert routing analysis
       st.markdown('<h3 class="subsection-header">Expert Routing Analysis</h3>', unsafe_allow_html=True)

       col1, col2 = st.columns(2)

       with col1:
           # Create a table showing token distribution
           routing_data = {
               "tokens_per_expert": comp["tokens_per_expert"],
               "active_ratio": comp["active_ratio"],
               "load_imbalance": comp["load_imbalance"],
               "total_active_experts": comp["total_active_experts"]
           }

           tokens_per_gpu = st.session_state.model_config["batch_size"] * st.session_state.model_config["seq_length"] / st.session_state.model_config["num_gpus"]

           # Calculate coefficient of variation
           if expert_loads:
               cv = np.std(expert_loads) / np.mean(expert_loads) if np.mean(expert_loads) > 0 else 0
           else:
               cv = comp["load_imbalance"]

           st.markdown('<div class="info-box">', unsafe_allow_html=True)
           st.markdown(f"""
           **Routing Statistics**
           
           - **Tokens per GPU**: {tokens_per_gpu:.1f}
           - **Tokens per Expert (avg)**: {comp['tokens_per_expert']:.1f}
           - **Coefficient of Variation**: {cv:.2f}
           - **Top-k per Token**: {st.session_state.model_config.get('top_k', 2)}
           - **Active Experts Ratio**: {comp['active_ratio']*100:.1f}%
           """)
           st.markdown('</div>', unsafe_allow_html=True)

       with col2:
           # Expert assignment effectiveness
           model_config = st.session_state.model_config

           # Estimate router quality metrics
           tokens_per_batch = model_config.get("batch_size", 0) * model_config.get("seq_length", 0)
           num_experts = model_config.get("num_experts", 0)
           top_k = model_config.get("top_k", 2)

           # Calculate efficiency metrics
           theoretical_min_cv = 0.0  # perfect balance
           estimated_cv = comp["load_imbalance"]
           random_cv = np.sqrt((num_experts - top_k) / (tokens_per_batch * top_k)) if tokens_per_batch * top_k
> 0 else 0

           efficiency = max(0, (random_cv - estimated_cv) / (random_cv - theoretical_min_cv)) if random_cv >
theoretical_min_cv else 1.0

           st.markdown('<div class="info-box">', unsafe_allow_html=True)
           st.markdown(f"""
           **Router Effectiveness**
           
           - **Random Assignment CV**: {random_cv:.2f}
           - **Current Assignment CV**: {estimated_cv:.2f}
           - **Optimal Assignment CV**: {theoretical_min_cv:.2f}
           - **Router Efficiency**: {efficiency*100:.1f}%
           """)
           st.markdown('</div>', unsafe_allow_html=True)

       # Load balancing strategies
       st.markdown('<h3 class="subsection-header">Load Balancing Strategies</h3>', unsafe_allow_html=True)

       strategies = [
           {
               "name": "Token Choice with Auxiliary Loss",
               "description": "Adds an auxiliary load balancing loss during training to encourage uniform expert usage",
               "efficiency": 0.85,
               "complexity": "Medium",
               "training_impact": "Minor increase in training time"
           },
           {
               "name": "Expert Choice Routing",
               "description": "Experts choose which tokens they want to process rather than tokens choosing experts",
               "efficiency": 0.95,
               "complexity": "Medium",
               "training_impact": "Different convergence characteristics"
           },
           {
               "name": "Balanced Assignment",
               "description": "Use Hungarian algorithm or other assignment methods to ensure balanced expert usage",
               "efficiency": 0.99,
               "complexity": "High",
               "training_impact": "Significant computation overhead"
           },
           {
               "name": "Capacity Factor Tuning",
               "description": "Adjust router capacity factors to limit maximum tokens per expert",
               "efficiency": 0.80,
               "complexity": "Low",
               "training_impact": "Potential token dropping"
           }
       ]

       # Create strategies dataframe
       strategies_df = pd.DataFrame(strategies)

       # Convert to a format more suitable for display
       strategies_display = pd.DataFrame([
           {"Strategy": s["name"],
            "Description": s["description"],
            "Balance Efficiency": f"{s['efficiency']*100:.0f}%",
            "Implementation Complexity": s["complexity"],
            "Training Impact": s["training_impact"]}
           for s in strategies
       ])

       st.dataframe(strategies_display, use_container_width=True)

       # Load balancing recommendation
       if comp['load_imbalance'] > 0.3:
           st.markdown('<div class="warning-box">', unsafe_allow_html=True)
           st.markdown("""
           **High Load Imbalance Detected**
           
           Your expert load imbalance of {:.1f}% is significant and likely impacts performance.
           Based on your configuration, we recommend:
           
           1. Implementing an auxiliary load balancing loss
           2. Consider Expert Choice routing as an alternative to Token Choice
           3. Setting capacity factor = {:.2f} × tokens_per_expert
           4. Monitor expert usage and periodically reset underutilized experts
           """.format(comp['load_imbalance']*100, 1.0 + comp['load_imbalance']))
           st.markdown('</div>', unsafe_allow_html=True)
       else:
           st.markdown('<div class="success-box">', unsafe_allow_html=True)
           st.markdown("""
           **Acceptable Load Balance**
           
           Your load imbalance of {:.1f}% is within reasonable limits. To further improve:
           
           1. Fine-tune router temperature parameter
           2. Implement subtle auxiliary load balancing to maintain balance
           3. Continue monitoring expert usage during training
           """.format(comp['load_imbalance']*100))
           st.markdown('</div>', unsafe_allow_html=True)

   # Recommendations Tab
   with tabs[5]:
       st.markdown('<h2 class="section-header">Optimization Recommendations</h2>', unsafe_allow_html=True)

       recommendations = analysis["recommendations"]

       if not recommendations:
           st.info("No specific recommendations generated. Your model appears to be well-optimized.")
       else:
           # Group recommendations by priority
           high_priority = [r for r in recommendations if r["priority"] == "high"]
           medium_priority = [r for r in recommendations if r["priority"] == "medium"]
           low_priority = [r for r in recommendations if r["priority"] == "low"]

           # Display high priority recommendations
           if high_priority:
               st.markdown('<h3 class="subsection-header">High Priority Optimizations</h3>',
unsafe_allow_html=True)

               for i, rec in enumerate(high_priority):
                   st.markdown('<div class="error-box">', unsafe_allow_html=True)
                   st.markdown(f"**{rec['title']}** ({rec['category']})")
                   st.markdown("Implementation steps:")
                   for step in rec["suggestions"]:
                       st.markdown(f"- {step}")
                   st.markdown('</div>', unsafe_allow_html=True)

           # Display medium priority recommendations
           if medium_priority:
               st.markdown('<h3 class="subsection-header">Medium Priority Optimizations</h3>',unsafe_allow_html=True)

               for i, rec in enumerate(medium_priority):
                   st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                   st.markdown(f"**{rec['title']}** ({rec['category']})")
                   st.markdown("Implementation steps:")
                   for step in rec["suggestions"]:
                       st.markdown(f"- {step}")
                   st.markdown('</div>', unsafe_allow_html=True)

           # Display low priority recommendations
           if low_priority:
               st.markdown('<h3 class="subsection-header">Additional Optimizations</h3>',unsafe_allow_html=True)

               for i, rec in enumerate(low_priority):
                   st.markdown('<div class="info-box">', unsafe_allow_html=True)
                   st.markdown(f"**{rec['title']}** ({rec['category']})")
                   st.markdown("Implementation steps:")
                   for step in rec["suggestions"]:
                       st.markdown(f"- {step}")
                   st.markdown('</div>', unsafe_allow_html=True)

       # Code implementation examples
       st.markdown('<h3 class="subsection-header">Implementation Examples</h3>', unsafe_allow_html=True)

       code_tabs = st.tabs([
           "DeepEP Configuration",
           "Load Balancing",
           "Memory Optimization",
           "Communication Overlap"
       ])

       with code_tabs[0]:
           st.markdown("### DeepEP Optimization Example")
           st.markdown("""
           This example shows how to configure DeepEP for optimal communication performance with your model.
           """)

           st.code("""
           import torch.distributed as dist
           from deep_ep import Buffer
           
           # Get process group
           group = dist.group.WORLD
           
           # Set optimal SM allocation based on model size
           # For H100 GPUs with MoE models, 24 is a good starting point
           Buffer.set_num_sms(24)
           
           # Calculate optimal buffer sizes
           hidden_size = 4096  # Model hidden size dimension
           
           # Get configuration for your specific scale
           dispatch_config = Buffer.get_dispatch_config(group.size())
           combine_config = Buffer.get_combine_config(group.size())
           
           # Calculate buffer size requirements
           num_nvl_bytes = max(
               dispatch_config.get_nvl_buffer_size_hint(hidden_size, group.size()),
               combine_config.get_nvl_buffer_size_hint(hidden_size, group.size())
           )
           
           num_rdma_bytes = max(
               dispatch_config.get_rdma_buffer_size_hint(hidden_size, group.size()),
               combine_config.get_rdma_buffer_size_hint(hidden_size, group.size())
           )
           
           # Create optimized communication buffer
           buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
           
           # Use FP8 for dispatch to reduce bandwidth requirements
           x_fp8, x_scales = convert_to_fp8(hidden_states)
           
           # Dispatch with communication-computation overlap
           recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = \
               buffer.dispatch(
                   (x_fp8, x_scales),  # Use FP8 representation to reduce bandwidth
                   topk_idx=topk_indices,
                   topk_weights=topk_probs,
                   num_tokens_per_rank=num_tokens_per_rank,
                   num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                   is_token_in_rank=is_token_in_rank,
                   num_tokens_per_expert=num_tokens_per_expert,
                   async_finish=True,  # Enable overlapping
                   allocate_on_comm_stream=True
               )
           """)

       with code_tabs[1]:
           st.markdown("### Expert Load Balancing Example")
           st.markdown("""
           This example shows how to implement auxiliary load balancing loss for better expert utilization.
           """)

           st.code("""
           import torch
           import torch.nn.functional as F
           
           def load_balancing_loss(router_logits, expert_indices, num_experts):
               """
               Compute auxiliary load balancing loss to prevent expert imbalance

               Args:
                   router_logits: Raw router outputs, shape [batch_size, seq_len, num_experts]
                   expert_indices: Expert assignments from router, shape [batch_size, seq_len, top_k]
                   num_experts: Total number of experts

               Returns:
                   loss: Auxiliary load balancing loss
               """
               # Get router probabilities
               router_probs = F.softmax(router_logits, dim=-1)  # [batch, seq, num_experts]
               
               # Calculate expert usage from router probabilities (soft assignment)
               # mean across batch dimension
               router_prob_exp = router_probs.sum(dim=[0, 1]) / router_probs.shape[0]  # [num_experts]
               
               # Calculate expert usage from actual routing decisions (hard assignment)
               # Create one-hot encodings of expert assignments
               batch_size, seq_len, top_k = expert_indices.shape
               expert_mask = torch.zeros(
                   batch_size, seq_len, num_experts, 
                   device=expert_indices.device
               )
               
               for k in range(top_k):
                   # For each token, increment count for assigned experts
                   expert_mask.scatter_add_(
                       2, 
                       expert_indices[:, :, k:k+1], 
                       torch.ones_like(expert_indices[:, :, k:k+1], dtype=torch.float)
                   )
               
               # Average over batch dimension
               expert_usage = expert_mask.sum(dim=[0, 1]) / (batch_size * seq_len)  # [num_experts]
               
               # Ideal expert usage is uniform (1/num_experts for each expert)
               ideal_usage = torch.ones_like(expert_usage) / num_experts
               
               # Loss has two components:
               # 1. Variance of router probability distributions (encourages uniform usage)
               # 2. KL divergence between actual usage and ideal usage
               
               # Compute variance loss
               var_loss = torch.mean((router_prob_exp - ideal_usage) ** 2)
               
               # Compute KL divergence (measures difference between distributions)
               # Add small epsilon to avoid log(0)
               kl_loss = F.kl_div(
                   torch.log(expert_usage + 1e-8),
                   ideal_usage,
                   reduction='batchmean'
               )
               
               # Combine losses with appropriate weights
               # Balance between variance loss and KL divergence
               loss = 0.5 * var_loss + 0.5 * kl_loss
               
               return loss
               
           # During training, add auxiliary loss to main loss
           aux_loss = load_balancing_loss(router_logits, expert_indices, num_experts)
           total_loss = main_loss + 0.01 * aux_loss  # aux_loss_weight is typically small (0.01-0.1)
           """)

       with code_tabs[2]:
           st.markdown("### Memory Optimization Example")
           st.markdown("""
           This example shows how to implement memory optimizations for large MoE models.
           """)

           st.code("""
           import torch
           from torch.utils.checkpoint import checkpoint
           
           class MemoryOptimizedMoELayer(torch.nn.Module):
               def __init__(self, hidden_size, ffn_dim, num_experts, top_k=2):
                   super().__init__()
                   self.hidden_size = hidden_size
                   self.ffn_dim = ffn_dim
                   self.num_experts = num_experts
                   self.top_k = top_k
                   
                   # Create router
                   self.router = torch.nn.Linear(hidden_size, num_experts)
                   
                   # Create experts with shared expert classes for memory efficiency
                   self.experts = torch.nn.ModuleList([
                       ExpertFFN(hidden_size, ffn_dim)
                       for _ in range(num_experts)
                   ])
                   
                   # Use half-precision by default
                   self.half_precision = True
               
               def forward(self, hidden_states):
                   # Use checkpointing for router to save memory
                   # This re-computes the forward pass during backward, saving activation memory
                   router_logits = checkpoint(self.router, hidden_states)
                   
                   # Top-k routing
                   routing_weights = torch.softmax(router_logits, dim=-1)
                   routing_weights, indices = torch.topk(routing_weights, self.top_k, dim=-1)
                   
                   # Normalize weights
                   routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
                   
                   # Apply FP8 precision for dispatch phase
                   if self.half_precision:
                       # Convert to BF16 for efficient dispatch
                       orig_dtype = hidden_states.dtype
                       hidden_states = hidden_states.to(torch.bfloat16)
                   
                   # Prepare for dispatch_to_experts function
                   # We'll implement this with DeepEP in a real system
                   expert_outputs = self._dispatch_to_experts(hidden_states, indices, routing_weights)
                   
                   # Restore original precision
                   if self.half_precision:
                       expert_outputs = expert_outputs.to(orig_dtype)
                   
                   return expert_outputs
               
               def _dispatch_to_experts(self, hidden_states, indices, routing_weights):
                   """Memory-efficient implementation of expert dispatch"""
                   batch_size, seq_len, _ = hidden_states.shape
                   hidden_states = hidden_states.reshape(-1, self.hidden_size)  # [batch*seq, hidden]
                   
                   # Process each expert with activation checkpointing
                   results = torch.zeros_like(hidden_states)
                   for expert_idx in range(self.num_experts):
                       # Find tokens routed to this expert (across all top-k slots)
                       mask = (indices == expert_idx).any(dim=-1).reshape(-1)
                       if not mask.any():
                           continue
                           
                       # Get tokens for this expert
                       expert_inputs = hidden_states[mask]
                       
                       # Process with checkpointing to save memory
                       expert_outputs = checkpoint(self.experts[expert_idx], expert_inputs)
                       
                       # Combine with routing weights
                       for k in range(self.top_k):
                           k_mask = (indices[:, :, k] == expert_idx).reshape(-1)
                           k_weights = routing_weights[:, :, k].reshape(-1)[k_mask]
                           results[k_mask] += expert_outputs * k_weights.unsqueeze(1)
                   
                   return results.reshape(batch_size, seq_len, self.hidden_size)
           """)

       with code_tabs[3]:
           st.markdown("### Communication-Computation Overlap Example")
           st.markdown("""
           This example shows how to implement communication-computation overlapping with DeepEP.
           """)

           st.code("""
           import torch
           import torch.distributed as dist
           from deep_ep import Buffer, EventOverlap
           
           def overlapped_moe_forward(hidden_states, router_logits, buffer, local_experts):
               batch_size, seq_len, hidden_size = hidden_states.shape
               
               # Get routing weights and indices
               routing_probs = torch.softmax(router_logits, dim=-1)
               topk_probs, topk_indices = torch.topk(routing_probs, k=2, dim=-1)
               
               # Normalize probabilities
               topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
               
               # Reshape for dispatching
               hidden_flat = hidden_states.reshape(-1, hidden_size)
               topk_probs = topk_probs.reshape(-1, 2)
               topk_indices = topk_indices.reshape(-1, 2)
               
               # Calculate dispatching layout
               num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event =
\
                   buffer.get_dispatch_layout(
                       topk_indices, 
                       num_experts=len(local_experts) * dist.get_world_size(),
                       async_finish=True
                   )
               
               # Start dispatch with async_finish for overlapping
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
               # This will overlap with the communication
               other_output = some_other_layer(hidden_states)
               
               # Process tokens with local experts
               expert_outputs = []
               offset = 0
               for i, expert in enumerate(local_experts):
                   num_tokens_for_expert = num_recv_tokens_per_expert_list[i]
                   if num_tokens_for_expert == 0:
                       continue
                   
                   # Get tokens for this expert
                   expert_input = recv_hidden_states[offset:offset + num_tokens_for_expert]
                   offset += num_tokens_for_expert
                   
                   # Process through expert
                   expert_output = expert(expert_input)
                   expert_outputs.append(expert_output)
               
               # Concatenate outputs from all local experts
               all_expert_outputs = torch.cat(expert_outputs, dim=0)
               
               # Start combine with async_finish
               combined_hidden_states, _, combine_event = buffer.combine(
                   all_expert_outputs, 
                   handle, 
                   topk_weights=recv_topk_weights,
                   async_finish=True
               )
               
               # Do more computation while combine is happening
               final_output = another_layer(other_output)
               
               # Wait for combine to finish (if needed)
               # Typically, you can continue without waiting if you don't need the results immediately
               
               # Reshape output back to original shape
               output = combined_hidden_states.reshape(batch_size, seq_len, hidden_size)
               
               return output, final_output
           """)

       # DeepEP-specific recommendations
       st.markdown('<h3 class="subsection-header">DeepEP-Specific Recommendations</h3>',
unsafe_allow_html=True)

       # Generate specific parameter recommendations for DeepEP
       model_config = st.session_state.model_config

       # Recommended SM allocation
       recommended_sms = min(24, max(8, int(model_config.get("hidden_size", 4096) / 256)))

       # Recommended expert alignment
       recommended_alignment = 16 if model_config.get("hidden_size", 4096) % 16 == 0 else 8

       # Recommended environment variables
       env_vars = {
           "NVSHMEM_IB_ENABLE_IBGDA": "1",
           "NVSHMEM_IBGDA_NIC_HANDLER": "gpu",
           "NVSHMEM_IBGDA_NUM_RC_PER_PE": str(model_config.get("experts_per_gpu", 8)),
           "NVSHMEM_QP_DEPTH": "1024",
           "NVSHMEM_DISABLE_P2P": "1" if model_config.get("num_gpus", 16) > 8 else "0"
       }

       st.markdown('<div class="success-box">', unsafe_allow_html=True)
       st.markdown(f"""
       **DeepEP Configuration Recommendations**
       
       Based on your model configuration, we recommend:
       
       1. **SM Allocation**: `Buffer.set_num_sms({recommended_sms})`
       2. **Expert Alignment**: Use `expert_alignment={recommended_alignment}` in dispatch calls
       3. **Precision**: Use FP8 for dispatch operations and BF16 for combine
       4. **Environment Variables**:
       """)

       for var, val in env_vars.items():
           st.markdown(f"   - `{var}={val}`")

       st.markdown("""
       5. **Buffer Size Optimization**: Use the `get_*_buffer_size_hint` functions for optimal sizing
       6. **Communication Overlapping**: Always use `async_finish=True` with proper event management
       """)
       st.markdown('</div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
   # This code is executed when the script is run directly
   pass
