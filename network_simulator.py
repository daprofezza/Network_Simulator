import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import io
import numpy as np
from matplotlib.patches import FancyBboxPatch
import math

# Page configuration
st.set_page_config(
    page_title="Network Diagram Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .critical-path-display {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255,107,107,0.3);
    }
    .info-box {
        background: #e8f4fd;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stDataFrame {
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-radius: 10px;
    }
    .sidebar .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

class NetworkDiagramApp:
    def __init__(self):
        if 'activities' not in st.session_state:
            st.session_state.activities = []
        if 'graph' not in st.session_state:
            st.session_state.graph = nx.DiGraph()
        if 'critical_path' not in st.session_state:
            st.session_state.critical_path = []
        if 'critical_path_nodes' not in st.session_state:
            st.session_state.critical_path_nodes = []
        if 'project_duration' not in st.session_state:
            st.session_state.project_duration = 0
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}

    def add_activity(self, activity, start_node, end_node, duration):
        """Add an activity to the network with validation"""
        try:
            # Clean inputs
            activity = activity.strip().upper()
            start_node = start_node.strip()
            end_node = end_node.strip()
            
            # Validation
            if not all([activity, start_node, end_node]):
                return False, "All fields must be filled"
            
            if duration <= 0:
                return False, "Duration must be positive"
            
            if start_node == end_node:
                return False, "Start and end nodes must be different"
            
            # Check for duplicate activity names
            existing_activities = [act[0] for act in st.session_state.activities]
            if activity in existing_activities:
                return False, f"Activity '{activity}' already exists"
            
            # Check for duplicate edges
            if st.session_state.graph.has_edge(start_node, end_node):
                return False, "Connection between these nodes already exists"
            
            # Add to graph and activities list
            st.session_state.graph.add_edge(start_node, end_node, 
                                          duration=duration, activity=activity)
            st.session_state.activities.append((activity, start_node, end_node, duration))
            
            # Clear previous analysis
            st.session_state.critical_path = []
            st.session_state.critical_path_nodes = []
            st.session_state.project_duration = 0
            st.session_state.analysis_results = {}
            
            return True, f"Activity '{activity}' added successfully"
            
        except Exception as e:
            return False, f"Error: {str(e)}"

    def remove_last_activity(self):
        """Remove the last added activity"""
        if not st.session_state.activities:
            return False, "No activities to remove"
        
        try:
            last_activity = st.session_state.activities.pop()
            activity, start, end, duration = last_activity
            st.session_state.graph.remove_edge(start, end)
            
            # Clear analysis
            st.session_state.critical_path = []
            st.session_state.critical_path_nodes = []
            st.session_state.project_duration = 0
            st.session_state.analysis_results = {}
            
            return True, f"Activity '{activity}' removed"
        except Exception as e:
            # Restore activity if error occurs
            if 'last_activity' in locals():
                st.session_state.activities.append(last_activity)
            return False, f"Error removing activity: {str(e)}"

    def clear_all_activities(self):
        """Clear all activities and reset"""
        st.session_state.activities = []
        st.session_state.graph = nx.DiGraph()
        st.session_state.critical_path = []
        st.session_state.critical_path_nodes = []
        st.session_state.project_duration = 0
        st.session_state.analysis_results = {}
        return True, "All activities cleared"

    def find_critical_path_sequence(self, graph, critical_edges, start_nodes):
        """Find the correct sequence of critical path from start to end"""
        if not critical_edges:
            return []
        
        # Build adjacency list for critical edges only
        critical_graph = {}
        for u, v in critical_edges:
            if u not in critical_graph:
                critical_graph[u] = []
            critical_graph[u].append(v)
        
        # Find the longest path through critical activities
        def dfs_longest_path(node, visited, path):
            if node in visited:
                return path
            
            visited.add(node)
            current_path = path + [node]
            
            if node not in critical_graph:
                return current_path
            
            longest = current_path
            for neighbor in critical_graph[node]:
                candidate_path = dfs_longest_path(neighbor, visited.copy(), current_path)
                if len(candidate_path) > len(longest):
                    longest = candidate_path
            
            return longest
        
        # Try from each start node and find the longest path
        longest_critical_path = []
        for start in start_nodes:
            if start in critical_graph or any(start == u for u, v in critical_edges):
                path = dfs_longest_path(start, set(), [])
                if len(path) > len(longest_critical_path):
                    longest_critical_path = path
        
        return longest_critical_path

    def compute_critical_path(self):
        """Fixed critical path computation using proper CPM algorithm"""
        try:
            if not st.session_state.activities:
                return False, "No activities to analyze"

            graph = st.session_state.graph
            
            # Verify graph is acyclic
            if not nx.is_directed_acyclic_graph(graph):
                return False, "Network contains cycles - invalid for CPM analysis"
            
            # Get all nodes and sort topologically
            nodes = list(graph.nodes())
            if not nodes:
                return False, "No nodes in the network"
            
            topo_order = list(nx.topological_sort(graph))
            
            # Find start and end nodes
            start_nodes = [n for n in nodes if graph.in_degree(n) == 0]
            end_nodes = [n for n in nodes if graph.out_degree(n) == 0]
            
            if not start_nodes or not end_nodes:
                return False, "Network must have clear start and end nodes"
            
            # Initialize time dictionaries
            es = {node: 0 for node in nodes}  # Early Start
            ef = {node: 0 for node in nodes}  # Early Finish
            ls = {node: float('inf') for node in nodes}  # Late Start
            lf = {node: float('inf') for node in nodes}  # Late Finish
            
            # Forward pass - calculate ES and EF
            for node in topo_order:
                if node in start_nodes:
                    es[node] = 0
                else:
                    # ES = max(EF of all predecessors)
                    max_ef = 0
                    for pred in graph.predecessors(node):
                        pred_ef = ef[pred]
                        max_ef = max(max_ef, pred_ef)
                    es[node] = max_ef
                
                # EF = ES + duration of outgoing activities (max if multiple)
                if graph.out_degree(node) == 0:
                    ef[node] = es[node]  # End node
                else:
                    max_duration = 0
                    for succ in graph.successors(node):
                        duration = graph[node][succ]['duration']
                        max_duration = max(max_duration, duration)
                    ef[node] = es[node] + max_duration
            
            # Project duration is the maximum EF of end nodes
            project_duration = max([ef[node] for node in end_nodes])
            
            # Backward pass - calculate LF and LS
            for node in reversed(topo_order):
                if node in end_nodes:
                    lf[node] = ef[node]
                else:
                    # LF = min(LS of all successors)
                    min_ls = float('inf')
                    for succ in graph.successors(node):
                        succ_ls = ls[succ]
                        min_ls = min(min_ls, succ_ls)
                    lf[node] = min_ls
                
                # LS = LF - duration of outgoing activities (max if multiple)
                if graph.out_degree(node) == 0:
                    ls[node] = lf[node]  # End node
                else:
                    max_duration = 0
                    for succ in graph.successors(node):
                        duration = graph[node][succ]['duration']
                        max_duration = max(max_duration, duration)
                    ls[node] = lf[node] - max_duration
            
            # Calculate activity-specific values and identify critical activities
            activity_analysis = []
            critical_activities = []
            critical_edges = []
            
            for u, v, data in graph.edges(data=True):
                activity = data['activity']
                duration = data['duration']
                
                # Activity times
                activity_es = es[u]
                activity_ef = activity_es + duration
                activity_ls = ls[v] - duration
                activity_lf = ls[v]
                
                # Total Float = LS - ES = LF - EF
                total_float = activity_ls - activity_es
                
                activity_analysis.append({
                    'Activity': activity,
                    'Start_Node': u,
                    'End_Node': v,
                    'Duration': duration,
                    'ES': activity_es,
                    'EF': activity_ef,
                    'LS': activity_ls,
                    'LF': activity_lf,
                    'Float': round(total_float, 2)
                })
                
                # Critical activities have zero or near-zero float
                if abs(total_float) < 0.001:
                    critical_activities.append(activity)
                    critical_edges.append((u, v))
            
            # Find the proper critical path sequence
            critical_path_nodes = self.find_critical_path_sequence(graph, critical_edges, start_nodes)
            
            # Update session state
            st.session_state.critical_path = critical_edges
            st.session_state.critical_path_nodes = critical_path_nodes
            st.session_state.project_duration = project_duration
            st.session_state.analysis_results = {
                'activities': activity_analysis,
                'critical_activities': critical_activities,
                'node_times': {'es': es, 'ef': ef, 'ls': ls, 'lf': lf},
                'start_nodes': start_nodes,
                'end_nodes': end_nodes
            }
            
            return True, f"Analysis complete! Project duration: {project_duration} days"

        except Exception as e:
            return False, f"Error in analysis: {str(e)}"

    def calculate_hierarchical_layout(self):
        """Calculate layout to minimize edge crossings using hierarchical positioning"""
        if not st.session_state.activities:
            return {}
        
        graph = st.session_state.graph
        
        # Try to use graphviz-style hierarchical layout if available
        try:
            pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
            return pos
        except:
            pass
        
        # Fallback: Custom hierarchical layout
        try:
            # Group nodes by their topological level
            topo_order = list(nx.topological_sort(graph))
            levels = {}
            node_levels = {}
            
            # Assign levels based on longest path from source
            for node in topo_order:
                if not list(graph.predecessors(node)):
                    level = 0
                else:
                    level = max([node_levels[pred] for pred in graph.predecessors(node)]) + 1
                
                node_levels[node] = level
                if level not in levels:
                    levels[level] = []
                levels[level].append(node)
            
            # Position nodes
            pos = {}
            level_width = 4.0
            node_spacing = 2.0
            
            for level, nodes in levels.items():
                x = level * level_width
                num_nodes = len(nodes)
                
                if num_nodes == 1:
                    pos[nodes[0]] = (x, 0)
                else:
                    start_y = -(num_nodes - 1) * node_spacing / 2
                    for i, node in enumerate(nodes):
                        y = start_y + i * node_spacing
                        pos[node] = (x, y)
            
            return pos
            
        except:
            # Final fallback: spring layout with parameters to reduce crossings
            return nx.spring_layout(graph, seed=42, k=2.0, iterations=100)

    def draw_network_diagram(self):
        """Draw enhanced network diagram with ES/LF values and animated critical path"""
        if not st.session_state.activities:
            return None

        # Create figure with clean styling
        fig, ax = plt.subplots(figsize=(16, 12))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        graph = st.session_state.graph
        
        # Use hierarchical layout to minimize crossings
        pos = self.calculate_hierarchical_layout()
        
        # Define enhanced colors with gradients
        regular_node_color = '#e8f5e8'
        critical_node_color = '#ffe6e6'
        node_border_color = '#2e7d32'
        critical_border_color = '#c62828'
        regular_edge_color = '#78909c'
        critical_edge_color = '#e53935'
        critical_path_highlight = '#ff1744'
        
        # Get critical elements
        critical_edges = set(st.session_state.critical_path)
        critical_nodes = set(st.session_state.critical_path_nodes)
        
        # Draw nodes with enhanced styling and larger size for ES/LF values
        all_nodes = list(graph.nodes())
        regular_nodes = [n for n in all_nodes if n not in critical_nodes]
        
        # Regular nodes
        if regular_nodes:
            nx.draw_networkx_nodes(graph, pos, nodelist=regular_nodes,
                                 node_color=regular_node_color,
                                 node_size=3500,
                                 edgecolors=node_border_color,
                                 linewidths=3,
                                 ax=ax,
                                 alpha=0.9)
        
        # Critical nodes with pulsing effect (using different alpha and colors)
        if critical_nodes:
            # Main critical nodes
            nx.draw_networkx_nodes(graph, pos, nodelist=list(critical_nodes),
                                 node_color=critical_node_color,
                                 node_size=3500,
                                 edgecolors=critical_border_color,
                                 linewidths=4,
                                 ax=ax,
                                 alpha=0.95)
            
            # Add a glow effect for critical nodes
            nx.draw_networkx_nodes(graph, pos, nodelist=list(critical_nodes),
                                 node_color=critical_path_highlight,
                                 node_size=4000,
                                 edgecolors='none',
                                 ax=ax,
                                 alpha=0.2)
        
        # Draw edges with enhanced styling
        regular_edges = [(u, v) for u, v in graph.edges() if (u, v) not in critical_edges]
        
        # Regular edges
        if regular_edges:
            nx.draw_networkx_edges(graph, pos, edgelist=regular_edges,
                                 edge_color=regular_edge_color,
                                 width=2.5,
                                 arrows=True,
                                 arrowsize=30,
                                 arrowstyle='-|>',
                                 node_size=3500,
                                 ax=ax,
                                 connectionstyle="arc3,rad=0.1",
                                 alpha=0.8)
        
        # Critical edges with multiple layers for glow effect
        if critical_edges:
            # Outer glow
            nx.draw_networkx_edges(graph, pos, edgelist=list(critical_edges),
                                 edge_color=critical_path_highlight,
                                 width=8,
                                 arrows=True,
                                 arrowsize=35,
                                 arrowstyle='-|>',
                                 node_size=3500,
                                 ax=ax,
                                 connectionstyle="arc3,rad=0.1",
                                 alpha=0.3)
            
            # Main critical path
            nx.draw_networkx_edges(graph, pos, edgelist=list(critical_edges),
                                 edge_color=critical_edge_color,
                                 width=5,
                                 arrows=True,
                                 arrowsize=32,
                                 arrowstyle='-|>',
                                 node_size=3500,
                                 ax=ax,
                                 connectionstyle="arc3,rad=0.1",
                                 alpha=0.9)
        
        # Enhanced node labels with ES and LF values
        if st.session_state.analysis_results:
            node_times = st.session_state.analysis_results.get('node_times', {})
            es_values = node_times.get('es', {})
            lf_values = node_times.get('lf', {})
            
            # Create enhanced labels
            enhanced_labels = {}
            for node in all_nodes:
                es = es_values.get(node, 0)
                lf = lf_values.get(node, 0)
                enhanced_labels[node] = f"{node}\nES:{es}\nLF:{lf}"
        else:
            enhanced_labels = {node: str(node) for node in all_nodes}
        
        # Draw node labels with better formatting
        for node, (x, y) in pos.items():
            label = enhanced_labels[node]
            color = 'darkred' if node in critical_nodes else 'darkgreen'
            weight = 'bold' if node in critical_nodes else 'normal'
            
            ax.text(x, y, label, 
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=12,
                   fontweight=weight,
                   color=color,
                   bbox=dict(boxstyle="round,pad=0.1", 
                           facecolor='white', 
                           edgecolor='none',
                           alpha=0.8))
        
        # Enhanced edge labels with activity and duration
        edge_labels = {}
        for u, v, data in graph.edges(data=True):
            activity = data['activity']
            duration = data['duration']
            is_critical = (u, v) in critical_edges
            
            if is_critical:
                edge_labels[(u, v)] = f"⭐ {activity}\n({duration}d)"
            else:
                edge_labels[(u, v)] = f"{activity}\n({duration}d)"
        
        # Position edge labels
        for (u, v), label in edge_labels.items():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2 + 0.2  # Slight offset
            
            is_critical = (u, v) in critical_edges
            bbox_color = '#ffcdd2' if is_critical else 'white'
            text_color = '#b71c1c' if is_critical else '#37474f'
            font_weight = 'bold' if is_critical else 'normal'
            
            ax.text(x, y, label,
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=11,
                   fontweight=font_weight,
                   color=text_color,
                   bbox=dict(boxstyle="round,pad=0.4",
                           facecolor=bbox_color,
                           edgecolor='gray' if not is_critical else '#f44336',
                           alpha=0.95,
                           linewidth=2 if is_critical else 1))
        
        # Enhanced title
        ax.set_title("Project Network Diagram with Critical Path Analysis", 
                    fontsize=20, 
                    fontweight='bold', 
                    pad=40,
                    color='#1a237e')
        
        # Add subtitle with project info
        if st.session_state.project_duration:
            ax.text(0.5, 0.95, f"Project Duration: {st.session_state.project_duration} days", 
                   transform=ax.transAxes,
                   horizontalalignment='center',
                   fontsize=14,
                   color='#d32f2f',
                   weight='bold')
        
        ax.axis('off')
        
        # Enhanced legend with more details
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=regular_node_color, 
                   markeredgecolor=node_border_color,
                   markersize=18, markeredgewidth=3,
                   label='Regular Node (ES/LF)'),
            Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=critical_node_color, 
                   markeredgecolor=critical_border_color,
                   markersize=18, markeredgewidth=4,
                   label='Critical Node (ES/LF)'),
            Line2D([0], [0], color=regular_edge_color, 
                   linewidth=3, label='Regular Activity'),
            Line2D([0], [0], color=critical_edge_color, 
                   linewidth=5, label='Critical Path Activity'),
            Line2D([0], [0], marker='*', color=critical_path_highlight, 
                   markersize=15, linewidth=0,
                   label='Critical Path Highlight')
        ]
        
        ax.legend(handles=legend_elements, 
                 loc='upper left', 
                 fontsize=12,
                 frameon=True,
                 fancybox=True,
                 shadow=True,
                 title="Legend",
                 title_fontsize=14)
        
        # Add critical path sequence as text
        if st.session_state.critical_path_nodes:
            path_text = " → ".join(str(node) for node in st.session_state.critical_path_nodes)
            ax.text(0.02, 0.02, f"Critical Path: {path_text}", 
                   transform=ax.transAxes,
                   fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.5",
                           facecolor='#ffebee',
                           edgecolor='#f44336',
                           alpha=0.9),
                   color='#b71c1c',
                   weight='bold')
        
        plt.tight_layout()
        return fig

    def add_dummy_data(self):
        """Add sample project data"""
        if st.session_state.activities:
            return False, "Clear existing data first"
        
        # Sample project: Software Development
        dummy_activities = [
            ("A", "1", "2", 3),  # Requirements Analysis
            ("B", "1", "3", 4),  # System Design
            ("C", "2", "4", 2),  # Database Setup
            ("D", "3", "4", 5),  # Development
            ("E", "4", "5", 3),  # Testing
            ("F", "5", "6", 1)   # Deployment
        ]
        
        for activity, start, end, duration in dummy_activities:
            st.session_state.graph.add_edge(start, end, 
                                          duration=duration, 
                                          activity=activity)
            st.session_state.activities.append((activity, start, end, duration))
        
        return True, "Sample project loaded: Software Development"

def main():
    app = NetworkDiagramApp()
    
    # Clean header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Network Diagram Simulator</h1>
        <p>Critical Path Method (CPM) Analysis Tool</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for input - cleaner layout
    with st.sidebar:
        st.markdown("### 📝 Add New Activity")
        
        with st.form("activity_form", clear_on_submit=True):
            activity = st.text_input("Activity ID", placeholder="A", help="Unique identifier (A, B, C, etc.)")
            
            col1, col2 = st.columns(2)
            with col1:
                start_node = st.text_input("Start", placeholder="1", help="Starting node")
            with col2:
                end_node = st.text_input("End", placeholder="2", help="Ending node")
            
            duration = st.number_input("Duration (days)", min_value=1, value=1, help="Time to complete activity")
            
            submitted = st.form_submit_button("➕ Add Activity", type="primary", use_container_width=True)
        
        if submitted:
            success, message = app.add_activity(activity, start_node, end_node, duration)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
        
        st.markdown("---")
        
        # Action buttons - cleaner layout
        st.markdown("### 🛠️ Actions")
        
        if st.button("🎲 Load Sample Data", use_container_width=True):
            success, message = app.add_dummy_data()
            if success:
                st.success(message)
                st.rerun()
            else:
                st.warning(message)
        
        if st.button("🔍 Analyze Critical Path", use_container_width=True, type="primary"):
            success, message = app.compute_critical_path()
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("↩️ Undo", use_container_width=True):
                success, message = app.remove_last_activity()
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.warning(message)
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                success, message = app.clear_all_activities()
                st.success(message)
                st.rerun()

    # Main content - cleaner layout
    if not st.session_state.activities:
        st.markdown("""
        <div class="info-box">
            <h3>🚀 Getting Started</h3>
            <p><strong>Step 1:</strong> Add activities using the sidebar form</p>
            <p><strong>Step 2:</strong> Click "Analyze Critical Path" to run CPM analysis</p>
            <p><strong>Step 3:</strong> View results and network diagram</p>
            <br>
            <p>💡 <strong>Tip:</strong> Use "Load Sample Data" to see an example project</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Results section - much cleaner
    if st.session_state.analysis_results:
        # Project metrics in a clean row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Project Duration", f"{st.session_state.project_duration} days")
        
        with col2:
            critical_count = len(st.session_state.analysis_results.get('critical_activities', []))
            total_count = len(st.session_state.activities)
            st.metric("Critical Activities", f"{critical_count}/{total_count}")
        
        with col3:
            if 'activities' in st.session_state.analysis_results:
                avg_float = np.mean([act['Float'] for act in st.session_state.analysis_results['activities']])
                st.metric("Average Float", f"{avg_float:.1f} days")
        
        with col4:
            st.metric("Total Activities", total_count)
        
        # Critical path display - show node sequence
        if st.session_state.critical_path_nodes:
            # Show both node sequence and activity sequence
            node_path_str = " → ".join(str(node) for node in st.session_state.critical_path_nodes)
            activity_path_str = " → ".join(st.session_state.analysis_results['critical_activities'])
            
            st.markdown(f"""
            <div class="critical-path-display">
                🔑 Critical Path (Nodes): {node_path_str}<br>
                🔑 Critical Path (Activities): {activity_path_str}
            </div>
            """, unsafe_allow_html=True)

    # Network diagram
    st.markdown("### 🕸️ Network Diagram")
    fig = app.draw_network_diagram()
    if fig:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    
    # Data table - only if analysis is done
    if st.session_state.analysis_results:
        st.markdown("### 📊 Activity Analysis")
        
        # Create comprehensive dataframe with all analysis elements
        analysis_data = st.session_state.analysis_results['activities']
        df = pd.DataFrame(analysis_data)
        
        # Calculate additional derived values
        df['Free_Float'] = df['Float']  # In simple networks, Total Float = Free Float
        df['Critical'] = df['Float'] == 0
        df['Earliest_Duration'] = df['EF'] - df['ES']
        df['Latest_Duration'] = df['LF'] - df['LS']
        
        # Reorder and rename columns for comprehensive display
        display_df = df[['Activity', 'Start_Node', 'End_Node', 'Duration', 'ES', 'EF', 'LS', 'LF', 'Float', 'Free_Float', 'Critical']].copy()
        display_df.columns = ['Activity', 'Start Node', 'End Node', 'Duration (days)', 'Early Start (ES)', 
                             'Early Finish (EF)', 'Late Start (LS)', 'Late Finish (LF)', 
                             'Total Float', 'Free Float', 'Critical?']
        
        # Advanced styling function
        def style_analysis_table(row):
            styles = [''] * len(row)
            if row['Critical?']:
                # Critical activities - red highlighting
                styles = ['background-color: #ffcdd2; font-weight: bold; color: #b71c1c'] * len(row)
            elif row['Total Float'] <= 1:
                # Near-critical activities - yellow highlighting
                styles = ['background-color: #fff9c4; color: #f57f17'] * len(row)
            else:
                # Regular activities - light green
                styles = ['background-color: #e8f5e8; color: #2e7d32'] * len(row)
            return styles
        
        # Apply enhanced styling
        styled_df = display_df.style.apply(style_analysis_table, axis=1)
        
        # Add formatting for numeric columns
        styled_df = styled_df.format({
            'Duration (days)': '{:.0f}',
            'Early Start (ES)': '{:.1f}',
            'Early Finish (EF)': '{:.1f}',
            'Late Start (LS)': '{:.1f}',
            'Late Finish (LF)': '{:.1f}',
            'Total Float': '{:.1f}',
            'Free Float': '{:.1f}',
            'Critical?': lambda x: '⭐ YES' if x else 'No'
        })
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Add summary statistics
        st.markdown("#### 📈 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            critical_activities = df[df['Critical']]['Activity'].tolist()
            st.info(f"**Critical Activities:**\n{', '.join(critical_activities)}")
        
        with col2:
            max_float = df['Float'].max()
            activities_with_max_float = df[df['Float'] == max_float]['Activity'].tolist()
            st.info(f"**Most Flexible Activity:**\n{', '.join(activities_with_max_float)}\n(Float: {max_float:.1f} days)")
        
        with col3:
            avg_duration = df['Duration'].mean()
            st.info(f"**Average Activity Duration:**\n{avg_duration:.1f} days")
        
        with col4:
            total_work = df['Duration'].sum()
            st.info(f"**Total Work Content:**\n{total_work:.0f} person-days")
        
        # Critical path efficiency metrics
        st.markdown("#### ⚡ Critical Path Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            critical_work = df[df['Critical']]['Duration'].sum()
            efficiency = (critical_work / total_work) * 100
            st.metric("Critical Path Efficiency", f"{efficiency:.1f}%", 
                     help="Percentage of total work on critical path")
        
        with col2:
            non_critical_activities = len(df[~df['Critical']])
            float_distribution = df[~df['Critical']]['Float'].describe()
            if non_critical_activities > 0:
                avg_float = float_distribution['mean']
                st.metric("Average Float (Non-Critical)", f"{avg_float:.1f} days",
                         help="Average slack time for non-critical activities")
            else:
                st.metric("Average Float (Non-Critical)", "N/A",
                         help="All activities are critical")
        
        # Download options
        col1, col2 = st.columns(2)
        
        with col1:
            if fig:
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                img_buffer.seek(0)
                
                st.download_button(
                    label="💾 Download Diagram",
                    data=img_buffer.getvalue(),
                    file_name="network_diagram.png",
                    mime="image/png",
                    use_container_width=True
                )
        
        with col2:
            csv_buffer = io.StringIO()
            display_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📤 Export Data",
                data=csv_buffer.getvalue(),
                file_name="project_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    else:
        # Simple activity list before analysis
        st.markdown("### 📋 Current Activities")
        simple_df = pd.DataFrame(st.session_state.activities, 
                               columns=['Activity', 'Start', 'End', 'Duration'])
        st.dataframe(simple_df, use_container_width=True, hide_index=True)

    # Updated footer with requested watermark
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "Network Diagram Simulator | Web Application | Developed by J. Inigo Papu Vinodhan, Asst. Prof., BBA Dept., St. Joseph's College, Trichy"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
