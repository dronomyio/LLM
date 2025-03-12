import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import random
from collections import defaultdict

st.set_page_config(
    page_title="Expert Parallelism Simulator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .info-box {
        background-color: #e8f4f8;
        border-left: 5px solid #4285f4;
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
    .token-metrics {
        padding: 10px;
        background-color: #f0f7fb;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 0.15rem 1.75rem rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'token_states' not in st.session_state:
    st.session_state.token_states = []
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'experts_states' not in st.session_state:
    st.session_state.experts_states = []

# Class definitions for simulation
class ExpertSystem:
    def __init__(self, num_gpus, experts_per_gpu, tokens_per_gpu, token_dim, routing_type, 
                topk, routing_skew, gpu_topology, comm_type):
        self.num_gpus = num_gpus
        self.experts_per_gpu = experts_per_gpu
        self.tokens_per_gpu = tokens_per_gpu
        self.token_dim = token_dim
        self.routing_type = routing_type
        self.topk = topk
        self.routing_skew = routing_skew
        self.gpu_topology = gpu_topology
        self.comm_type = comm_type

        self.num_tokens = num_gpus * tokens_per_gpu
        self.num_experts = num_gpus * experts_per_gpu

        # Initialize system
        self.tokens = []
        self.experts = []
        self.metrics = {
            "dispatch_times": [],
            "combine_times": [],
            "bandwidth_utilization": [],
            "expert_load": [],
            "nvlink_msgs": 0,
            "rdma_msgs": 0,
            "total_data_transferred": 0
        }

        # Create GPU layout
        self.gpus = []
        if gpu_topology == "Linear":
            # Simple linear arrangement
            self.gpus = [{"id": i, "pos": (i, 0)} for i in range(num_gpus)]
        elif gpu_topology == "Grid":
            # 2D grid arrangement
            cols = int(np.ceil(np.sqrt(num_gpus)))
            rows = int(np.ceil(num_gpus / cols))
            for i in range(num_gpus):
                row = i // cols
                col = i % cols
                self.gpus.append({"id": i, "pos": (col, row)})
        else:  # Hierarchical
            # Assume 4 GPUs per node
            gpus_per_node = 4
            nodes = int(np.ceil(num_gpus / gpus_per_node))
            for i in range(num_gpus):
                node = i // gpus_per_node
                pos_in_node = i % gpus_per_node
                x = node * 2 + (pos_in_node % 2)
                y = pos_in_node // 2
                self.gpus.append({"id": i, "pos": (x, y)})

        # Create experts
        for i in range(self.num_experts):
            gpu_id = i // experts_per_gpu
            gpu_pos = self.gpus[gpu_id]["pos"]

            # Add slight offset for experts within same GPU
            expert_idx = i % experts_per_gpu
            offset_x = (expert_idx % 2) * 0.2 - 0.1
            offset_y = (expert_idx // 2) * 0.2 - 0.2

            pos = (gpu_pos[0] + offset_x, gpu_pos[1] + offset_y)

            self.experts.append({
                "id": i,
                "gpu_id": gpu_id,
                "pos": pos,
                "specialization": np.random.rand(),  # Random specialization value
                "load": 0,  # Current load (number of tokens)
                "tokens": []  # Tokens currently assigned
            })

        # Create tokens
        for i in range(self.num_tokens):
            gpu_id = i // tokens_per_gpu
            gpu_pos = self.gpus[gpu_id]["pos"]

            # Add random offset for tokens within same GPU
            offset_x = (np.random.rand() - 0.5) * 0.4
            offset_y = (np.random.rand() - 0.5) * 0.4

            pos = (gpu_pos[0] + offset_x, gpu_pos[1] + offset_y)

            # Each token has a feature vector with random values that determines
            # which experts it gets routed to
            features = np.random.rand(self.num_experts)

            # Apply routing skew - some patterns are more common
            if routing_skew > 0:
                # Adjust the distribution based on skew
                pattern_type = i % 5  # 5 different patterns
                for j in range(self.num_experts):
                    if j % 5 == pattern_type:
                        features[j] *= (1 + routing_skew)

            self.tokens.append({
                "id": i,
                "gpu_id": gpu_id,
                "orig_pos": pos,
                "curr_pos": pos,
                "features": features,
                "assigned_experts": [],  # Will be filled during routing
                "expert_weights": [],    # Weights for each expert
                "state": "init",         # States: init, dispatched, processed, combined
                "value": np.random.rand(token_dim)  # Token's value/embedding
            })

    def route_tokens(self):
        """Route tokens to experts based on the routing strategy"""
        for token in self.tokens:
            features = token["features"]

            if self.routing_type == "Random":
                # Random routing
                expert_indices = np.random.choice(
                    self.num_experts,
                    size=self.topk,
                    replace=False
                )
                expert_weights = np.ones(self.topk) / self.topk

            elif self.routing_type == "Load-balanced":
                # Consider expert load in routing decision
                expert_indices = np.argsort(
                    [self.experts[i]["load"] for i in range(self.num_experts)]
                )[:self.topk]

                # Equal weights
                expert_weights = np.ones(self.topk) / self.topk

            else:  # TopK routing based on feature match
                # Get top-k experts based on feature match
                expert_indices = np.argsort(-features)[:self.topk]

                # Get weights (normalized)
                expert_values = features[expert_indices]
                expert_weights = expert_values / np.sum(expert_values)

            # Assign experts and weights
            token["assigned_experts"] = expert_indices.tolist()
            token["expert_weights"] = expert_weights.tolist()

    def simulate_dispatch(self):
        """Simulate dispatching tokens to their assigned experts"""
        dispatch_data = []
        total_data = 0

        # Track how many messages go through each link type
        nvlink_messages = 0
        rdma_messages = 0

        # First, route tokens
        self.route_tokens()

        # Track expert load
        expert_loads = [0] * self.num_experts

        for token in self.tokens:
            src_gpu = token["gpu_id"]
            token_size = self.token_dim * 2  # bytes per value (assuming fp16)

            # For each assigned expert
            for i, expert_id in enumerate(token["assigned_experts"]):
                expert = self.experts[expert_id]
                dst_gpu = expert["gpu_id"]

                # Determine communication type
                is_nvlink = self._is_nvlink_connection(src_gpu, dst_gpu)

                # Count message type
                if is_nvlink:
                    nvlink_messages += 1
                else:
                    rdma_messages += 1

                # Record dispatch information
                dispatch_data.append({
                    "token_id": token["id"],
                    "expert_id": expert_id,
                    "src_gpu": src_gpu,
                    "dst_gpu": dst_gpu,
                    "weight": token["expert_weights"][i],
                    "is_nvlink": is_nvlink
                })

                # Update expert load
                expert_loads[expert_id] += 1

                # Update token position (for visualization)
                token["curr_pos"] = expert["pos"]
                token["state"] = "dispatched"

                # Add to expert's token list
                expert["tokens"].append(token["id"])

                # Track data transferred
                total_data += token_size

        # Update expert loads
        for i, load in enumerate(expert_loads):
            self.experts[i]["load"] = load

        # Calculate dispatch time based on network topology
        # This is a simplified model - in reality it would depend on many factors
        nvlink_bandwidth = 900  # GB/s
        rdma_bandwidth = 50    # GB/s

        nvlink_time = 0
        if nvlink_messages > 0:
            nvlink_data = nvlink_messages * self.token_dim * 2 / (1024 * 1024 * 1024)  # GB
            nvlink_time = nvlink_data / nvlink_bandwidth * 1000  # ms

        rdma_time = 0
        if rdma_messages > 0:
            rdma_data = rdma_messages * self.token_dim * 2 / (1024 * 1024 * 1024)  # GB
            rdma_time = rdma_data / rdma_bandwidth * 1000  # ms

        # Total time is max of nvlink and rdma times
        dispatch_time = max(nvlink_time, rdma_time)

        # Calculate utilization
        theoretical_max_bandwidth = nvlink_bandwidth + rdma_bandwidth
        actual_bandwidth = (nvlink_data / nvlink_time * 1000 if nvlink_time > 0 else 0) + \
                           (rdma_data / rdma_time * 1000 if rdma_time > 0 else 0)
        bandwidth_utilization = actual_bandwidth / theoretical_max_bandwidth * 100

        # Update metrics
        self.metrics["dispatch_times"].append(dispatch_time)
        self.metrics["bandwidth_utilization"].append(bandwidth_utilization)
        self.metrics["expert_load"].append(expert_loads)
        self.metrics["nvlink_msgs"] += nvlink_messages
        self.metrics["rdma_msgs"] += rdma_messages
        self.metrics["total_data_transferred"] += total_data

        return dispatch_data, dispatch_time

    def simulate_processing(self):
        """Simulate experts processing their assigned tokens"""
        # In a real system, this would be the computation phase
        # Here we just update token states
        for token in self.tokens:
            if token["state"] == "dispatched":
                token["state"] = "processed"

        # Clear expert token lists (they've been processed)
        for expert in self.experts:
            expert["tokens"] = []

    def simulate_combine(self):
        """Simulate combining results back to original tokens"""
        combine_data = []
        total_data = 0

        # Track how many messages go through each link type
        nvlink_messages = 0
        rdma_messages = 0

        for token in self.tokens:
            if token["state"] != "processed":
                continue

            dst_gpu = token["gpu_id"]
            token_size = self.token_dim * 2  # bytes per value (assuming fp16)

            # For each assigned expert
            for i, expert_id in enumerate(token["assigned_experts"]):
                expert = self.experts[expert_id]
                src_gpu = expert["gpu_id"]

                # Determine communication type
                is_nvlink = self._is_nvlink_connection(src_gpu, dst_gpu)

                # Count message type
                if is_nvlink:
                    nvlink_messages += 1
                else:
                    rdma_messages += 1

                # Record combine information
                combine_data.append({
                    "token_id": token["id"],
                    "expert_id": expert_id,
                    "src_gpu": src_gpu,
                    "dst_gpu": dst_gpu,
                    "weight": token["expert_weights"][i],
                    "is_nvlink": is_nvlink
                })

                # Track data transferred
                total_data += token_size

            # Update token position (for visualization)
            token["curr_pos"] = token["orig_pos"]
            token["state"] = "combined"

        # Calculate combine time based on network topology
        nvlink_bandwidth = 900  # GB/s
        rdma_bandwidth = 50    # GB/s

        nvlink_time = 0
        if nvlink_messages > 0:
            nvlink_data = nvlink_messages * self.token_dim * 2 / (1024 * 1024 * 1024)  # GB
            nvlink_time = nvlink_data / nvlink_bandwidth * 1000  # ms

        rdma_time = 0
        if rdma_messages > 0:
            rdma_data = rdma_messages * self.token_dim * 2 / (1024 * 1024 * 1024)  # GB
            rdma_time = rdma_data / rdma_bandwidth * 1000  # ms

        # Total time is max of nvlink and rdma times
        combine_time = max(nvlink_time, rdma_time)

        # Calculate utilization
        theoretical_max_bandwidth = nvlink_bandwidth + rdma_bandwidth
        actual_bandwidth = (nvlink_data / nvlink_time * 1000 if nvlink_time > 0 else 0) + \
                           (rdma_data / rdma_time * 1000 if rdma_time > 0 else 0)
        bandwidth_utilization = actual_bandwidth / theoretical_max_bandwidth * 100

        # Update metrics
        self.metrics["combine_times"].append(combine_time)
        self.metrics["bandwidth_utilization"].append(bandwidth_utilization)
        self.metrics["nvlink_msgs"] += nvlink_messages
        self.metrics["rdma_msgs"] += rdma_messages
        self.metrics["total_data_transferred"] += total_data

        return combine_data, combine_time

    def reset_tokens(self):
        """Reset tokens to their initial state"""
        for token in self.tokens:
            token["curr_pos"] = token["orig_pos"]
            token["state"] = "init"
            token["assigned_experts"] = []
            token["expert_weights"] = []

        # Reset expert loads
        for expert in self.experts:
            expert["load"] = 0
            expert["tokens"] = []

    def _is_nvlink_connection(self, gpu1, gpu2):
        """Determine if two GPUs are connected via NVLink"""
        if self.comm_type == "All NVLink":
            return True
        elif self.comm_type == "All RDMA":
            return False
        else:  # Realistic
            # In the hierarchical topology, GPUs in the same node (same group of 4) have NVLink
            if self.gpu_topology == "Hierarchical":
                return (gpu1 // 4) == (gpu2 // 4)
            # In other topologies, adjacent GPUs have NVLink
            else:
                return abs(gpu1 - gpu2) == 1 or (gpu1 == 0 and gpu2 == self.num_gpus - 1)

    def get_token_state(self):
        """Return the current state of all tokens for visualization"""
        return [
            {
                "id": token["id"],
                "x": token["curr_pos"][0],
                "y": token["curr_pos"][1],
                "state": token["state"],
                "gpu_id": token["gpu_id"],
                "experts": token["assigned_experts"]
            }
            for token in self.tokens
        ]

    def get_expert_state(self):
        """Return the current state of all experts for visualization"""
        return [
            {
                "id": expert["id"],
                "x": expert["pos"][0],
                "y": expert["pos"][1],
                "gpu_id": expert["gpu_id"],
                "load": expert["load"],
                "tokens": expert["tokens"]
            }
            for expert in self.experts
        ]

    def get_gpu_positions(self):
        """Return GPU positions for visualization"""
        return [
            {
                "id": gpu["id"],
                "x": gpu["pos"][0],
                "y": gpu["pos"][1]
            }
            for gpu in self.gpus
        ]

def visualize_system(token_states, expert_states, gpu_positions, scale=1.0):
    """Create a Plotly figure visualizing the system state"""
    fig = go.Figure()

    # Add GPUs
    gpu_x = [gpu["x"] for gpu in gpu_positions]
    gpu_y = [gpu["y"] for gpu in gpu_positions]
    gpu_text = [f"GPU {gpu['id']}" for gpu in gpu_positions]

    fig.add_trace(go.Scatter(
        x=gpu_x, y=gpu_y,
        mode='markers+text',
        marker=dict(
            size=50*scale,
            color='rgba(66, 135, 245, 0.8)',
            line=dict(width=2, color='DarkSlateGrey')
        ),
        text=gpu_text,
        textposition="bottom center",
        name="GPUs"
    ))

    # Add experts
    expert_x = [expert["x"] for expert in expert_states]
    expert_y = [expert["y"] for expert in expert_states]
    expert_text = [f"E{expert['id']}<br>Load: {expert['load']}" for expert in expert_states]
    expert_size = [20 + min(30, expert["load"] * 3) for expert in expert_states]
    expert_color = ['rgba(50, 205, 50, 0.7)'] * len(expert_states)

    fig.add_trace(go.Scatter(
        x=expert_x, y=expert_y,
        mode='markers+text',
        marker=dict(
            size=expert_size,
            color=expert_color,
            line=dict(width=1.5, color='DarkSlateGrey')
        ),
        text=expert_text,
        textposition="middle center",
        name="Experts"
    ))

    # Add tokens with different colors based on state
    colors = {
        "init": "rgba(100, 149, 237, 0.7)",      # Blue
        "dispatched": "rgba(255, 165, 0, 0.7)",  # Orange
        "processed": "rgba(106, 90, 205, 0.7)",  # Purple
        "combined": "rgba(50, 205, 50, 0.7)"     # Green
    }

    for state in ["init", "dispatched", "processed", "combined"]:
        state_tokens = [t for t in token_states if t["state"] == state]
        if not state_tokens:
            continue

        token_x = [token["x"] for token in state_tokens]
        token_y = [token["y"] for token in state_tokens]
        token_text = [f"T{token['id']}" for token in state_tokens]

        fig.add_trace(go.Scatter(
            x=token_x, y=token_y,
            mode='markers',
            marker=dict(
                size=12*scale,
                color=colors[state],
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=token_text,
            hoverinfo="text",
            name=f"Tokens ({state})"
        ))

    # Layout settings
    x_range = [min(gpu_x) - 1, max(gpu_x) + 1]
    y_range = [min(gpu_y) - 1, max(gpu_y) + 1]

    fig.update_layout(
        showlegend=True,
        xaxis=dict(
            range=x_range,
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            range=y_range,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="x",
            scaleratio=1
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        height=600
    )

    return fig

def visualize_expert_load(expert_states):
    """Create a bar chart of expert loads"""
    expert_ids = [f"E{e['id']}" for e in expert_states]
    expert_loads = [e["load"] for e in expert_states]
    expert_gpus = [e["gpu_id"] for e in expert_states]

    # Create color scale by GPU
    gpu_colors = px.colors.qualitative.Plotly
    colors = [gpu_colors[gpu_id % len(gpu_colors)] for gpu_id in expert_gpus]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=expert_ids,
        y=expert_loads,
        marker_color=colors,
        text=expert_loads,
        textposition='auto'
    ))

    fig.update_layout(
        title="Expert Load Distribution",
        xaxis_title="Expert ID",
        yaxis_title="Number of Tokens",
        height=400
    )

    return fig

def visualize_metrics(metrics):
    """Create charts for performance metrics"""
    figs = []

    # Dispatch and combine times
    if metrics.get("dispatch_times") and metrics.get("combine_times"):
        fig_times = go.Figure()

        fig_times.add_trace(go.Scatter(
            x=list(range(len(metrics["dispatch_times"]))),
            y=metrics["dispatch_times"],
            mode='lines+markers',
            name='Dispatch Time (ms)'
        ))

        fig_times.add_trace(go.Scatter(
            x=list(range(len(metrics["combine_times"]))),
            y=metrics["combine_times"],
            mode='lines+markers',
            name='Combine Time (ms)'
        ))

        fig_times.update_layout(
            title="Operation Latency",
            xaxis_title="Iteration",
            yaxis_title="Time (ms)",
            height=300
        )

        figs.append(fig_times)

    # Bandwidth utilization
    if metrics.get("bandwidth_utilization"):
        fig_bw = go.Figure()

        fig_bw.add_trace(go.Scatter(
            x=list(range(len(metrics["bandwidth_utilization"]))),
            y=metrics["bandwidth_utilization"],
            mode='lines+markers',
            line=dict(color='green'),
            name='Bandwidth Utilization (%)'
        ))

        fig_bw.update_layout(
            title="Bandwidth Utilization",
            xaxis_title="Iteration",
            yaxis_title="Utilization (%)",
            height=300
        )

        figs.append(fig_bw)

    # Message counts pie chart
    if "nvlink_msgs" in metrics and "rdma_msgs" in metrics:
        labels = ['NVLink Messages', 'RDMA Messages']
        values = [metrics["nvlink_msgs"], metrics["rdma_msgs"]]

        fig_msgs = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.3
        )])

        fig_msgs.update_layout(
            title="Communication Message Types",
            height=300
        )

        figs.append(fig_msgs)

    return figs

def initialize_system():
    """Initialize the simulation system based on user parameters"""
    # Get parameters from input widgets
    num_gpus = st.session_state.num_gpus
    experts_per_gpu = st.session_state.experts_per_gpu
    tokens_per_gpu = st.session_state.tokens_per_gpu
    token_dim = st.session_state.token_dim
    routing_type = st.session_state.routing_type
    topk = st.session_state.topk
    routing_skew = st.session_state.routing_skew
    gpu_topology = st.session_state.gpu_topology
    comm_type = st.session_state.comm_type

    # Create the system
    system = ExpertSystem(
        num_gpus, experts_per_gpu, tokens_per_gpu, token_dim,
        routing_type, topk, routing_skew, gpu_topology, comm_type
    )

    # Store initial state
    st.session_state.system = system
    st.session_state.token_states = system.get_token_state()
    st.session_state.expert_states = system.get_expert_state()
    st.session_state.gpu_positions = system.get_gpu_positions()
    st.session_state.step = 0
    st.session_state.dispatch_data = None
    st.session_state.combine_data = None
    st.session_state.metrics = system.metrics

    st.session_state.system_initialized = True
    st.session_state.simulation_running = False

def run_simulation_step():
    """Run one step of the simulation"""
    system = st.session_state.system

    if st.session_state.step == 0:
        # Dispatch phase
        dispatch_data, dispatch_time = system.simulate_dispatch()
        st.session_state.dispatch_data = dispatch_data
        st.session_state.dispatch_time = dispatch_time
        st.session_state.step = 1

    elif st.session_state.step == 1:
        # Processing phase
        system.simulate_processing()
        st.session_state.step = 2

    elif st.session_state.step == 2:
        # Combine phase
        combine_data, combine_time = system.simulate_combine()
        st.session_state.combine_data = combine_data
        st.session_state.combine_time = combine_time
        st.session_state.step = 3

    else:
        # Reset for next iteration
        system.reset_tokens()
        st.session_state.step = 0
        st.session_state.dispatch_data = None
        st.session_state.combine_data = None

    # Update state
    st.session_state.token_states = system.get_token_state()
    st.session_state.expert_states = system.get_expert_state()
    st.session_state.metrics = system.metrics

# Main app layout
st.markdown('<h1 class="main-header">Expert Parallelism Simulator</h1>', unsafe_allow_html=True)

st.markdown("""
This interactive simulator demonstrates how expert parallelism works in Mixture-of-Experts (MoE) models.
Configure your system, visualize token routing, and analyze communication patterns in a distributed GPU 
environment.
""")

# Sidebar configuration
with st.sidebar:
    st.header("System Configuration")

    # System parameters
    st.subheader("Hardware")
    st.session_state.num_gpus = st.slider("Number of GPUs", 2, 16, 8)
    st.session_state.experts_per_gpu = st.slider("Experts per GPU", 1, 8, 2)
    st.session_state.tokens_per_gpu = st.slider("Tokens per GPU", 4, 32, 16)
    st.session_state.token_dim = st.slider("Token Dimension", 512, 8192, 4096, step=512)

    st.subheader("Topology")
    st.session_state.gpu_topology = st.selectbox(
        "GPU Topology",
        ["Linear", "Grid", "Hierarchical"],
        index=2
    )

    st.session_state.comm_type = st.selectbox(
        "Communication Type",
        ["Realistic", "All NVLink", "All RDMA"],
        index=0
    )

    st.subheader("Routing")
    st.session_state.routing_type = st.selectbox(
        "Routing Algorithm",
        ["TopK", "Random", "Load-balanced"],
        index=0
    )

    st.session_state.topk = st.slider("Top-k Experts", 1, 4, 2)
    st.session_state.routing_skew = st.slider("Routing Skew", 0.0, 2.0, 0.5,
                                             help="Higher values make some tokens more likely to be routed to the same experts")

    # Initialize button
    if st.button("Initialize System"):
        initialize_system()

    # Control buttons
    if st.session_state.system_initialized:
        st.subheader("Simulation Control")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Single Step"):
                run_simulation_step()

        with col2:
            if not st.session_state.simulation_running:
                if st.button("Run Auto"):
                    st.session_state.simulation_running = True
            else:
                if st.button("Stop"):
                    st.session_state.simulation_running = False

# Main content area
if not st.session_state.system_initialized:
    # Display initial instructions
    st.info("Configure your system parameters in the sidebar and click 'Initialize System' to begin.")

    # Show a sample visualization
    st.markdown('<h2 class="section-header">About Expert Parallelism</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <p>Expert Parallelism is a technique used in Mixture-of-Experts (MoE) models where:</p>
    <ul>
        <li>The model contains multiple specialized "expert" networks</li>
        <li>Each input token is routed to only a small subset of experts</li>
        <li>Experts are distributed across multiple GPUs</li>
        <li>Tokens must be sent to the correct GPUs (dispatch) and results gathered back (combine)</li>
    </ul>
    <p>This creates a complex all-to-all communication pattern that needs optimization.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h3>Simulation Steps</h3>', unsafe_allow_html=True)
        st.markdown("""
        1. **Dispatch**: Tokens are routed to their assigned experts
        2. **Processing**: Experts process their assigned tokens
        3. **Combine**: Results are sent back to original GPU
        4. **Reset**: System prepares for next batch
        """)

    with col2:
        st.markdown('<h3>Key Performance Factors</h3>', unsafe_allow_html=True)
        st.markdown("""
        - **Communication Bandwidth**: NVLink (~900 GB/s) vs RDMA (~50 GB/s)
        - **Expert Load Balance**: How evenly tokens are distributed
        - **Token Routing Algorithm**: How tokens get assigned to experts
        - **GPU Topology**: Physical arrangement of GPUs
        """)

    # Sample system visualization
    st.markdown('<h3>Sample Visualization</h3>', unsafe_allow_html=True)

    # Create a sample system
    sample_system = ExpertSystem(
        num_gpus=4,
        experts_per_gpu=2,
        tokens_per_gpu=8,
        token_dim=4096,
        routing_type="TopK",
        topk=2,
        routing_skew=0.5,
        gpu_topology="Grid",
        comm_type="Realistic"
    )

    # Get initial state
    sample_tokens = sample_system.get_token_state()
    sample_experts = sample_system.get_expert_state()
    sample_gpus = sample_system.get_gpu_positions()

    # Show visualization
    sample_fig = visualize_system(sample_tokens, sample_experts, sample_gpus)
    st.plotly_chart(sample_fig, use_container_width=True)

else:
    # Auto-advance simulation if running
    if st.session_state.simulation_running:
        run_simulation_step()
        time.sleep(0.5)
        st.rerun()

    # Display system state and metrics
    step_names = ["Initialize", "Dispatch", "Process", "Combine"]
    current_step = step_names[st.session_state.step]

    st.markdown(f'<h2 class="section-header">Current Step: {current_step}</h2>', unsafe_allow_html=True)

    # Show system visualization
    system_fig = visualize_system(
        st.session_state.token_states,
        st.session_state.expert_states,
        st.session_state.gpu_positions
    )
    st.plotly_chart(system_fig, use_container_width=True)

    # Show expert load distribution
    expert_load_fig = visualize_expert_load(st.session_state.expert_states)
    st.plotly_chart(expert_load_fig, use_container_width=True)

    # Metrics section
    st.markdown('<h2 class="section-header">Performance Metrics</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if hasattr(st.session_state, 'dispatch_time'):
            st.metric("Dispatch Time", f"{st.session_state.dispatch_time:.2f} ms")
        else:
            st.metric("Dispatch Time", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if hasattr(st.session_state, 'combine_time'):
            st.metric("Combine Time", f"{st.session_state.combine_time:.2f} ms")
        else:
            st.metric("Combine Time", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if st.session_state.metrics.get("nvlink_msgs") is not None:
            st.metric("NVLink Messages", st.session_state.metrics["nvlink_msgs"])
        else:
            st.metric("NVLink Messages", "0")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if st.session_state.metrics.get("rdma_msgs") is not None:
            st.metric("RDMA Messages", st.session_state.metrics["rdma_msgs"])
        else:
            st.metric("RDMA Messages", "0")
        st.markdown('</div>', unsafe_allow_html=True)

    # Display metric charts
    metric_figs = visualize_metrics(st.session_state.metrics)
    for fig in metric_figs:
        st.plotly_chart(fig, use_container_width=True)

    # Expert utilization analysis
    if st.session_state.expert_states:
        st.markdown('<h2 class="section-header">Expert Utilization Analysis</h2>', unsafe_allow_html=True)

        # Calculate statistics
        loads = [e["load"] for e in st.session_state.expert_states]
        max_load = max(loads) if loads else 0
        min_load = min(loads) if loads else 0
        avg_load = sum(loads) / len(loads) if loads else 0
        load_std = np.std(loads) if loads else 0

        # Load balance metrics
        imbalance = load_std / avg_load if avg_load > 0 else 0

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Max Load", f"{max_load}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Min Load", f"{min_load}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Load", f"{avg_load:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Load Imbalance", f"{imbalance:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Expert analysis and recommendations
        if imbalance > 0.5:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("""
            **High Load Imbalance Detected**
            
            Your experts have a significant load imbalance which may impact performance. Consider:
            
            1. Using load-balanced routing to distribute tokens more evenly
            2. Implementing capacity factors or auxiliary loss for routing
            3. Increasing expert diversity to reduce specialization overlap
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        # Communication efficiency analysis
        nvlink_ratio = st.session_state.metrics.get("nvlink_msgs", 0) / max(1,
(st.session_state.metrics.get("nvlink_msgs", 0) + st.session_state.metrics.get("rdma_msgs", 0)))

        if nvlink_ratio < 0.3:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("""
            **Low NVLink Utilization**
            
            Your system is using more RDMA than NVLink for communication. Consider:
            
            1. Grouping related experts on the same node to increase NVLink usage
            2. Modifying token routing to favor intra-node experts
            3. Using a different GPU topology with more intra-node connections
            """)
            st.markdown('</div>', unsafe_allow_html=True)

    # Token routing details
    with st.expander("Token Routing Details"):
        if st.session_state.dispatch_data:
            df = pd.DataFrame(st.session_state.dispatch_data)
            st.dataframe(df)

