import streamlit as st
import graphviz

st.title('PX4 Firmware Startup Flow')

# Create tabs for different platforms
tab1, tab2 = st.tabs(['NuttX (Hardware)', 'POSIX (Simulation)'])

with tab1:
    st.header('NuttX Platform Startup Flow')
    
    # Create a graphviz diagram for NuttX
    nuttx_dot = graphviz.Digraph()
    nuttx_dot.attr(rankdir='TB')
    
    # Add nodes
    nuttx_dot.node('bootloader', 'Bootloader (verifies firmware)', style='filled', fillcolor='lightblue')
    nuttx_dot.node('hw_init', 'Hardware Initialization', style='filled', fillcolor='lightblue')
    nuttx_dot.node('px4_init', 'px4_platform_init()', style='filled', fillcolor='lightgreen')
    nuttx_dot.node('cpp_init', 'C++ Constructors', style='filled', fillcolor='lightgreen')
    nuttx_dot.node('hrt_init', 'HRT (High-Resolution Timer)', style='filled', fillcolor='lightgreen')
    nuttx_dot.node('param_init', 'Parameter System', style='filled', fillcolor='orange')
    nuttx_dot.node('uorb_init', 'uORB Messaging', style='filled', fillcolor='orange')
    nuttx_dot.node('work_queues', 'Work Queues', style='filled', fillcolor='orange')
    nuttx_dot.node('rcs', 'rcS Script Execution', style='filled', fillcolor='red')
    nuttx_dot.node('sensors', 'Sensor Init', style='filled', fillcolor='yellow')
    nuttx_dot.node('estimators', 'EKF2 & Estimators', style='filled', fillcolor='yellow')
    nuttx_dot.node('apps', 'Vehicle Apps', style='filled', fillcolor='pink')
    
    # Add edges
    nuttx_dot.edge('bootloader', 'hw_init')
    nuttx_dot.edge('hw_init', 'px4_init')
    nuttx_dot.edge('px4_init', 'cpp_init')
    nuttx_dot.edge('cpp_init', 'hrt_init')
    nuttx_dot.edge('hrt_init', 'param_init')
    nuttx_dot.edge('param_init', 'uorb_init')
    nuttx_dot.edge('uorb_init', 'work_queues')
    nuttx_dot.edge('work_queues', 'rcs')
    nuttx_dot.edge('rcs', 'sensors')
    nuttx_dot.edge('sensors', 'estimators')
    nuttx_dot.edge('estimators', 'apps')
    
    st.graphviz_chart(nuttx_dot)
    
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
    
    # Create a graphviz diagram for POSIX
    posix_dot = graphviz.Digraph()
    posix_dot.attr(rankdir='TB')
    
    # Add nodes
    posix_dot.node('main', 'main() in platforms/posix/src/px4/common/main.cpp', style='filled', fillcolor='lightblue')
    posix_dot.node('args', 'Parse Command Line Arguments', style='filled', fillcolor='lightblue')
    posix_dot.node('dirs', 'Create Directories & Symlinks', style='filled', fillcolor='lightgreen')
    posix_dot.node('px4_init_once', 'px4::init_once()', style='filled', fillcolor='lightgreen')
    posix_dot.node('px4_init', 'px4::init()', style='filled', fillcolor='lightgreen')
    posix_dot.node('daemon', 'Start Daemon Process', style='filled', fillcolor='orange')
    posix_dot.node('rcs_posix', 'rcS Script (POSIX version)', style='filled', fillcolor='red')
    posix_dot.node('sim', 'Start Simulator', style='filled', fillcolor='yellow')
    posix_dot.node('modules', 'Start PX4 Modules', style='filled', fillcolor='pink')
    
    # Add edges
    posix_dot.edge('main', 'args')
    posix_dot.edge('args', 'dirs')
    posix_dot.edge('dirs', 'px4_init_once')
    posix_dot.edge('px4_init_once', 'px4_init')
    posix_dot.edge('px4_init', 'daemon')
    posix_dot.edge('daemon', 'rcs_posix')
    posix_dot.edge('rcs_posix', 'sim')
    posix_dot.edge('sim', 'modules')
    
    st.graphviz_chart(posix_dot)
    
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

