import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import to_rgba

st.title('PX4 Firmware Startup Flow')

# Create tabs for different platforms
tab1, tab2 = st.tabs(['NuttX (Hardware)', 'POSIX (Simulation)'])

with tab1:
    st.header('NuttX Platform Startup Flow')
    
    # Create a function to make a directed graph with matplotlib instead of graphviz
    def create_nuttx_graph():
        G = nx.DiGraph()
        
        # Add nodes with positions
        nodes = [
            ('bootloader', 'Bootloader (verifies firmware)'),
            ('hw_init', 'Hardware Initialization'),
            ('px4_init', 'px4_platform_init()'),
            ('cpp_init', 'C++ Constructors'),
            ('hrt_init', 'HRT (High-Resolution Timer)'),
            ('param_init', 'Parameter System'),
            ('uorb_init', 'uORB Messaging'),
            ('work_queues', 'Work Queues'),
            ('rcs', 'rcS Script Execution'),
            ('sensors', 'Sensor Init'),
            ('estimators', 'EKF2 & Estimators'),
            ('apps', 'Vehicle Apps')
        ]
        
        # Define node positions (x, y)
        pos = {
            'bootloader': (0, 11),
            'hw_init': (0, 10),
            'px4_init': (0, 9),
            'cpp_init': (0, 8),
            'hrt_init': (0, 7),
            'param_init': (0, 6),
            'uorb_init': (0, 5),
            'work_queues': (0, 4),
            'rcs': (0, 3),
            'sensors': (0, 2),
            'estimators': (0, 1),
            'apps': (0, 0)
        }
        
        # Add nodes to graph
        for node_id, node_label in nodes:
            G.add_node(node_id, label=node_label)
        
        # Add edges
        edges = [
            ('bootloader', 'hw_init'),
            ('hw_init', 'px4_init'),
            ('px4_init', 'cpp_init'),
            ('cpp_init', 'hrt_init'),
            ('hrt_init', 'param_init'),
            ('param_init', 'uorb_init'),
            ('uorb_init', 'work_queues'),
            ('work_queues', 'rcs'),
            ('rcs', 'sensors'),
            ('sensors', 'estimators'),
            ('estimators', 'apps')
        ]
        G.add_edges_from(edges)
        
        # Create figure and draw
        plt.figure(figsize=(10, 12))
        
        # Define node colors
        node_colors = {
            'bootloader': 'lightblue',
            'hw_init': 'lightblue',
            'px4_init': 'lightgreen',
            'cpp_init': 'lightgreen',
            'hrt_init': 'lightgreen',
            'param_init': 'orange',
            'uorb_init': 'orange',
            'work_queues': 'orange',
            'rcs': 'red',
            'sensors': 'yellow',
            'estimators': 'yellow',
            'apps': 'pink'
        }
        
        # Extract colors for drawing
        colors = [node_colors[node] for node in G.nodes()]
        
        # Draw the graph
        nx.draw(G, pos, with_labels=False, node_color=colors, node_size=2000, arrows=True, 
                arrowstyle='->', arrowsize=20, edge_color='gray')
        
        # Draw labels with a white background for visibility
        for node, (x, y) in pos.items():
            plt.text(x, y, G.nodes[node]['label'], fontsize=9, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        
        plt.axis('off')
        plt.tight_layout()
        return plt

    # Display the plot
    st.pyplot(create_nuttx_graph().gcf())
    
    # Add explanations
    st.subheader('NuttX Startup Process')
    st.markdown('''
    1. **Bootloader**: Verifies firmware integrity and jumps to application
    2. **Hardware Initialization**: Low-level hardware setup
    3. **px4_platform_init()**: Core platform initialization
    4. **C++ Constructors**: Initialize static C++ objects
    5. **HRT Init**: Setup high-resolution timer
    6. **Parameter System**: Load parameters from storage
    7. **uORB Messaging**: Initialize messaging system for inter-module communication
    8. **Work Queues**: Setup task scheduling system
    9. **rcS Script**: Main startup script that calls other init scripts
    10. **Sensor Initialization**: Start and configure sensors
    11. **Estimators**: Start EKF2 and other state estimators
    12. **Vehicle Apps**: Launch vehicle-specific applications (multicopter, fixed-wing, etc.)
    ''')

with tab2:
    st.header('POSIX/SITL Platform Startup Flow')
    
    # Create a function to make a directed graph with matplotlib for POSIX
    def create_posix_graph():
        G = nx.DiGraph()
        
        # Add nodes with positions
        nodes = [
            ('main', 'main() in platforms/posix'),
            ('args', 'Parse Command Line Arguments'),
            ('dirs', 'Create Directories & Symlinks'),
            ('px4_init_once', 'px4::init_once()'),
            ('px4_init', 'px4::init()'),
            ('daemon', 'Start Daemon Process'),
            ('rcs_posix', 'rcS Script (POSIX version)'),
            ('sim', 'Start Simulator'),
            ('modules', 'Start PX4 Modules')
        ]
        
        # Define node positions (x, y)
        pos = {
            'main': (0, 8),
            'args': (0, 7),
            'dirs': (0, 6),
            'px4_init_once': (0, 5),
            'px4_init': (0, 4),
            'daemon': (0, 3),
            'rcs_posix': (0, 2),
            'sim': (0, 1),
            'modules': (0, 0)
        }
        
        # Add nodes to graph
        for node_id, node_label in nodes:
            G.add_node(node_id, label=node_label)
        
        # Add edges
        edges = [
            ('main', 'args'),
            ('args', 'dirs'),
            ('dirs', 'px4_init_once'),
            ('px4_init_once', 'px4_init'),
            ('px4_init', 'daemon'),
            ('daemon', 'rcs_posix'),
            ('rcs_posix', 'sim'),
            ('sim', 'modules')
        ]
        G.add_edges_from(edges)
        
        # Create figure and draw
        plt.figure(figsize=(10, 10))
        
        # Define node colors
        node_colors = {
            'main': 'lightblue',
            'args': 'lightblue',
            'dirs': 'lightgreen',
            'px4_init_once': 'lightgreen',
            'px4_init': 'lightgreen',
            'daemon': 'orange',
            'rcs_posix': 'red',
            'sim': 'yellow',
            'modules': 'pink'
        }
        
        # Extract colors for drawing
        colors = [node_colors[node] for node in G.nodes()]
        
        # Draw the graph
        nx.draw(G, pos, with_labels=False, node_color=colors, node_size=2000, arrows=True, 
                arrowstyle='->', arrowsize=20, edge_color='gray')
        
        # Draw labels with a white background for visibility
        for node, (x, y) in pos.items():
            plt.text(x, y, G.nodes[node]['label'], fontsize=9, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        
        plt.axis('off')
        plt.tight_layout()
        return plt

    # Display the plot
    st.pyplot(create_posix_graph().gcf())
    
    # Add explanations
    st.subheader('POSIX/SITL Startup Process')
    st.markdown('''
    1. **main()**: Entry point in platforms/posix/src/px4/common/main.cpp
    2. **Parse Arguments**: Process command-line arguments
    3. **Create Directories**: Setup necessary directories and symlinks for simulation
    4. **px4::init_once()**: One-time initialization of platform components
    5. **px4::init()**: Initialize platform-specific components
    6. **Start Daemon**: Create server/client architecture for simulation
    7. **rcS Script (POSIX)**: Execute POSIX-specific initialization scripts from ROMFS/px4fmu_common/init.d-posix/
    8. **Start Simulator**: Initialize the selected simulator (jMAVSim, Gazebo, etc.)
    9. **Start PX4 Modules**: Launch necessary PX4 modules for simulation
    ''')

st.header('Key Components')
st.markdown('''
### uORB Messaging System
- **Purpose**: Inter-module communication
- **Mechanism**: Publish/Subscribe messaging pattern
- **Implementation**: Shared memory on hardware, network sockets in simulation

### Parameter System
- **Purpose**: Configuration storage and management
- **Features**: Persistent storage, parameter update notifications
- **Access**: Available to all modules

### Modules System
- **Purpose**: Encapsulate functionality in independent modules
- **Types**: Drivers, Estimators, Controllers, etc.
- **Management**: Dynamic loading/unloading in POSIX, static linking in NuttX
''')

st.header('Startup Scripts')
st.markdown('''
The rcS scripts handle the following:
- SD card mounting
- Parameter loading
- Airframe configuration
- Sensor drivers
- State estimators (EKF2, etc.)
- Hardware interfaces (PWM, RC, etc.)
- Commander (flight control logic)
- Vehicle-specific setup
- Navigation and mission handling
''')

# Add a run button to simulate execution
if st.button('Simulate PX4 Startup'):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate startup steps
    for i, step in enumerate([
        'Verifying firmware...',
        'Initializing hardware...',
        'Setting up platform...',
        'Loading parameters...',
        'Starting uORB messaging...',
        'Executing rcS script...',
        'Starting sensor drivers...',
        'Initializing estimators...',
        'Starting vehicle applications...',
        'PX4 system ready\!'
    ]):
        # Update progress bar and status text
        progress_bar.progress((i+1)/10)
        status_text.text(step)
        
    st.success('PX4 Firmware started successfully\!')

