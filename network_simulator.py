import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import io
import numpy as np
from matplotlib.patches import FancyBboxPatch

# Page configuration
st.set_page_config(
    page_title="Network Diagram Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .critical-path-display {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-box {
        background: #e8f4fd;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .path-highlight {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
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
            activity = activity.strip().upper()
            start_node = start_node.strip()
            end_node = end_node.strip()
            
            if not all([activity, start_node, end_node]):
                return False, "All fields must be filled"
            if duration <= 0:
                return False, "Duration must be positive"
            if start_node == end_node:
                return False, "Start and end nodes must be different"
                
            existing_activities = [act[0] for act in st.session_state.activities]
            if activity in existing_activities:
                return False, f"Activity '{activity}' already exists"
                
            if st.session_state.graph.has_edge(start_node, end_node):
                return False, "Connection between these nodes already exists"
            
            # Test for cycles before adding
            temp_graph = st.session_state.graph.copy()
            temp_graph.add_edge(start_node, end_node, duration=duration, activity=activity)
            if not nx.is_directed_acyclic_graph(temp_graph):
                return False, "Adding this activity would create a cycle"
            
            st.session_state.graph.add_edge(start_node, end_node,
                                          duration=duration, activity=activity)
            st.session_state.activities.append((activity, start_node, end_node, duration))
            
            # Reset analysis
            st.session_state.critical_path = []
            st.session_state.critical_path_nodes = []
            st.session_state.project_duration = 0
            st.session_state.analysis_results = {}
            
            return True, f"Activity '{activity}' added successfully"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def compute_critical_path(self):
        """CORRECTED: Proper CPM algorithm with critical path from node 1"""
        try:
            if not st.session_state.activities:
                return False, "No activities to analyze"
                
            graph = st.session_state.graph
            
            if not nx.is_directed_acyclic_graph(graph):
                return False, "Network contains cycles - invalid for CPM analysis"
            
            nodes = list(graph.nodes())
            if not nodes:
                return False, "No nodes in the network"
            
            # Find start and end nodes
            start_nodes = [n for n in nodes if graph.in_degree(n) == 0]
            end_nodes = [n for n in nodes if graph.out_degree(n) == 0]
            
            if not start_nodes or not end_nodes:
                return False, "Network must have clear start and end nodes"
            
            # Prioritize node '1' as starting node if it exists and is a start node
            if '1' in start_nodes:
                primary_start = '1'
            else:
                primary_start = start_nodes[0]
            
            # Initialize time values for all nodes
            es = {node: 0 for node in nodes}  # Early Start
            ef = {node: 0 for node in nodes}  # Early Finish
            ls = {node: float('inf') for node in nodes}  # Late Start
            lf = {node: float('inf') for node in nodes}  # Late Finish
            
            # FORWARD PASS - Calculate ES and EF
            topo_order = list(nx.topological_sort(graph))
            
            # Calculate Early Start and Early Finish
            for node in topo_order:
                if graph.in_degree(node) == 0:
                    es[node] = 0
                    ef[node] = 0
                else:
                    # ES = max(ES + duration) of all predecessors
                    max_ef = 0
                    for pred in graph.predecessors(node):
                        activity_duration = graph[pred][node]['duration']
                        pred_finish = es[pred] + activity_duration
                        max_ef = max(max_ef, pred_finish)
                    es[node] = max_ef
                    ef[node] = es[node]
            
            # Project duration = max ES of end nodes
            project_duration = max(es[node] for node in end_nodes)
            
            # BACKWARD PASS - Calculate LS and LF
            reverse_topo = list(reversed(topo_order))
            
            # Initialize end nodes with project duration
            for node in end_nodes:
                lf[node] = es[node]  # LF = EF for end nodes
                ls[node] = es[node]  # LS = ES for end nodes
            
            for node in reverse_topo:
                if graph.out_degree(node) == 0:
                    continue
                else:
                    # LF = min(LS of successors)
                    min_ls = float('inf')
                    for succ in graph.successors(node):
                        activity_duration = graph[node][succ]['duration']
                        succ_start = ls[succ] - activity_duration
                        min_ls = min(min_ls, succ_start + activity_duration)
                    lf[node] = min_ls
                    ls[node] = lf[node]
            
            # Recalculate LS properly
            for node in reverse_topo:
                if graph.out_degree(node) == 0:
                    continue
                else:
                    min_start = float('inf')
                    for succ in graph.successors(node):
                        activity_duration = graph[node][succ]['duration']
                        required_start = ls[succ] - activity_duration
                        min_start = min(min_start, required_start)
                    ls[node] = min_start
                    lf[node] = ls[node]
            
            # Calculate activity analysis
            activity_analysis = []
            critical_activities = []
            critical_edges = []
            
            for u, v, data in graph.edges(data=True):
                activity = data['activity']
                duration = data['duration']
                
                # Activity times
                activity_es = es[u]
                activity_ef = es[u] + duration
                activity_ls = ls[v] - duration
                activity_lf = ls[v]
                
                # Total Float = LS - ES or LF - EF
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
                
                # Critical activities have zero float
                if abs(total_float) < 0.001:
                    critical_activities.append(activity)
                    critical_edges.append((u, v))
            
            # Find critical path starting from node 1 (or primary start node)
            critical_path_nodes = self.find_critical_path_from_start(graph, primary_start, 
                                                                   critical_edges, end_nodes)
            
            # Store results
            st.session_state.critical_path = critical_edges
            st.session_state.critical_path_nodes = critical_path_nodes
            st.session_state.project_duration = project_duration
            st.session_state.analysis_results = {
                'activities': activity_analysis,
                'critical_activities': critical_activities,
                'node_times': {'es': es, 'ef': ef, 'ls': ls, 'lf': lf},
                'start_nodes': start_nodes,
                'end_nodes': end_nodes,
                'primary_start': primary_start
            }
            
            return True, f"Analysis complete! Project duration: {project_duration} days"
            
        except Exception as e:
            return False, f"Error in analysis: {str(e)}"

    def find_critical_path_from_start(self, graph, start_node, critical_edges, end_nodes):
        """Find critical path starting from specified start node"""
        def dfs_critical_path(current_node, path, visited_edges):
            if current_node in end_nodes:
                return path
            
            # Find critical successors
            for next_node in graph.successors(current_node):
                edge = (current_node, next_node)
                if edge in critical_edges and edge not in visited_edges:
                    new_visited = visited_edges.copy()
                    new_visited.add(edge)
                    result = dfs_critical_path(next_node, path + [next_node], new_visited)
                    if result:
                        return result
            return None
        
        critical_path = dfs_critical_path(start_node, [start_node], set())
        return critical_path if critical_path else [start_node]

    def get_critical_activities_sequence(self):
        """Get the sequence of critical activities from the critical path"""
        if not st.session_state.critical_path_nodes or len(st.session_state.critical_path_nodes) < 2:
            return []
        
        graph = st.session_state.graph
        critical_activities_sequence = []
        
        for i in range(len(st.session_state.critical_path_nodes) - 1):
            current_node = st.session_state.critical_path_nodes[i]
            next_node = st.session_state.critical_path_nodes[i + 1]
            
            if graph.has_edge(current_node, next_node):
                activity = graph[current_node][next_node]['activity']
                duration = graph[current_node][next_node]['duration']
                critical_activities_sequence.append((activity, duration))
        
        return critical_activities_sequence

    def draw_network_diagram(self):
        """Enhanced network diagram with proper critical path highlighting from node 1"""
        if not st.session_state.activities:
            return None
            
        fig, ax = plt.subplots(figsize=(16, 12))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        graph = st.session_state.graph
        
        # Use hierarchical layout for better visualization
        try:
            pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
        except:
            # Fallback to spring layout if graphviz not available
            pos = nx.spring_layout(graph, seed=42, k=3.0, iterations=100)
        
        # Color scheme
        regular_node_color = '#e8f5e8'
        critical_node_color = '#ffcdd2'
        start_node_color = '#81c784'  # Special color for starting node
        node_border_color = '#2e7d32'
        critical_border_color = '#c62828'
        start_border_color = '#388e3c'
        regular_edge_color = '#78909c'
        critical_edge_color = '#e53935'
        
        critical_edges = set(st.session_state.critical_path)
        critical_nodes = set(st.session_state.critical_path_nodes)
        
        # Get primary start node
        primary_start = st.session_state.analysis_results.get('primary_start', '1')
        
        all_nodes = list(graph.nodes())
        regular_nodes = [n for n in all_nodes if n not in critical_nodes and n != primary_start]
        critical_non_start = [n for n in critical_nodes if n != primary_start]
        
        # Draw nodes with different colors
        if regular_nodes:
            nx.draw_networkx_nodes(graph, pos, nodelist=regular_nodes,
                                 node_color=regular_node_color,
                                 node_size=4000,
                                 edgecolors=node_border_color,
                                 linewidths=3, ax=ax)
        
        if critical_non_start:
            nx.draw_networkx_nodes(graph, pos, nodelist=critical_non_start,
                                 node_color=critical_node_color,
                                 node_size=4000,
                                 edgecolors=critical_border_color,
                                 linewidths=4, ax=ax)
        
        # Highlight the primary start node specially
        if primary_start in all_nodes:
            nx.draw_networkx_nodes(graph, pos, nodelist=[primary_start],
                                 node_color=start_node_color,
                                 node_size=5000,
                                 edgecolors=start_border_color,
                                 linewidths=5, ax=ax)
        
        # Draw edges
        regular_edges = [(u, v) for u, v in graph.edges() if (u, v) not in critical_edges]
        if regular_edges:
            nx.draw_networkx_edges(graph, pos, edgelist=regular_edges,
                                 edge_color=regular_edge_color,
                                 width=2, arrows=True, arrowsize=20,
                                 arrowstyle='-|>', ax=ax)
        
        if critical_edges:
            nx.draw_networkx_edges(graph, pos, edgelist=list(critical_edges),
                                 edge_color=critical_edge_color,
                                 width=4, arrows=True, arrowsize=25,
                                 arrowstyle='-|>', ax=ax)
        
        # Node labels with ES values
        if st.session_state.analysis_results:
            node_times = st.session_state.analysis_results.get('node_times', {})
            es_values = node_times.get('es', {})
            
            for node, (x, y) in pos.items():
                es = es_values.get(node, 0)
                if node == primary_start:
                    label = f"{node}\nSTART\nES:{es:.0f}"
                    color = 'darkgreen'
                    bbox_color = start_node_color
                elif node in critical_nodes:
                    label = f"{node}\nES:{es:.0f}"
                    color = 'darkred'
                    bbox_color = critical_node_color
                else:
                    label = f"{node}\nES:{es:.0f}"
                    color = 'darkgreen'
                    bbox_color = regular_node_color
                
                ax.text(x, y, label, ha='center', va='center',
                       fontsize=10, fontweight='bold', color=color,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=bbox_color, 
                               alpha=0.7, edgecolor=color))
        else:
            nx.draw_networkx_labels(graph, pos, ax=ax)
        
        # Edge labels with activity info
        edge_labels = {}
        for u, v, data in graph.edges(data=True):
            activity = data['activity']
            duration = data['duration']
            is_critical = (u, v) in critical_edges
            if is_critical:
                edge_labels[(u, v)] = f"{activity}({duration})★"
            else:
                edge_labels[(u, v)] = f"{activity}({duration})"
        
        # Position edge labels
        for (u, v), label in edge_labels.items():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2 + 0.1
            is_critical = (u, v) in critical_edges
            color = '#b71c1c' if is_critical else '#37474f'
            weight = 'bold' if is_critical else 'normal'
            ax.text(x, y, label, ha='center', va='center',
                   fontsize=9, fontweight=weight, color=color,
                   bbox=dict(boxstyle="round,pad=0.2", 
                           facecolor='#ffcdd2' if is_critical else 'white', 
                           alpha=0.8))
        
        ax.set_title("CPM Network Diagram - Critical Path Analysis\n(Green = Start Node, Red = Critical Path)",
                    fontsize=16, fontweight='bold', pad=30)
        
        if st.session_state.project_duration:
            ax.text(0.5, 0.02, f"Project Duration: {st.session_state.project_duration} days",
                   transform=ax.transAxes, ha='center', fontsize=14,
                   color='#d32f2f', weight='bold')
        
        ax.axis('off')
        plt.tight_layout()
        return fig

    def add_dummy_data(self):
        """Add sample project data for testing"""
        if st.session_state.activities:
            return False, "Clear existing data first"
        
        dummy_activities = [
            ("A", "1", "2", 3),
            ("B", "1", "3", 4), 
            ("C", "2", "4", 2),
            ("D", "3", "4", 5),
            ("E", "4", "5", 3),
        ]
        
        for activity, start, end, duration in dummy_activities:
            st.session_state.graph.add_edge(start, end,
                                          duration=duration, activity=activity)
            st.session_state.activities.append((activity, start, end, duration))
        
        return True, "Sample project loaded successfully"

    def clear_all_activities(self):
        """Clear all activities and reset"""
        st.session_state.activities = []
        st.session_state.graph = nx.DiGraph()
        st.session_state.critical_path = []
        st.session_state.critical_path_nodes = []
        st.session_state.project_duration = 0
        st.session_state.analysis_results = {}
        return True, "All activities cleared"

    def remove_last_activity(self):
        """Remove the last added activity"""
        if not st.session_state.activities:
            return False, "No activities to remove"
        try:
            last_activity = st.session_state.activities.pop()
            activity, start, end, duration = last_activity
            st.session_state.graph.remove_edge(start, end)
            # Reset analysis
            st.session_state.critical_path = []
            st.session_state.critical_path_nodes = []
            st.session_state.project_duration = 0
            st.session_state.analysis_results = {}
            return True, f"Activity '{activity}' removed"
        except Exception as e:
            return False, f"Error removing activity: {str(e)}"


def main():
    app = NetworkDiagramApp()
    
    st.markdown("""
    <div class="main-header">
        <h1>📊 CPM Network Diagram Simulator</h1>
        <p>Critical Path Method Analysis Tool - Enhanced with Node 1 Focus</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for input
    with st.sidebar:
        st.markdown("### 📝 Add New Activity")
        with st.form("activity_form", clear_on_submit=True):
            activity = st.text_input("Activity ID", placeholder="A")
            col1, col2 = st.columns(2)
            with col1:
                start_node = st.text_input("Start Node", placeholder="1")
            with col2:
                end_node = st.text_input("End Node", placeholder="2")
            duration = st.number_input("Duration (days)", min_value=1, value=1)
            submitted = st.form_submit_button("➕ Add Activity", type="primary")
        
        if submitted:
            success, message = app.add_activity(activity, start_node, end_node, duration)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
        
        st.markdown("---")
        
        # Action buttons
        if st.button("🎲 Load Sample Data"):
            success, message = app.add_dummy_data()
            if success:
                st.success(message)
                st.rerun()
            else:
                st.warning(message)
        
        if st.button("🔍 Analyze Critical Path", type="primary"):
            success, message = app.compute_critical_path()
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("↩️ Undo"):
                success, message = app.remove_last_activity()
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.warning(message)
        with col2:
            if st.button("🗑️ Clear All"):
                success, message = app.clear_all_activities()
                st.success(message)
                st.rerun()
    
    # Main content area
    if not st.session_state.activities:
        st.markdown("""
        <div class="info-box">
            <h3>🚀 Getting Started</h3>
            <p><strong>Step 1:</strong> Add activities using the sidebar form</p>
            <p><strong>Step 2:</strong> Click "Analyze Critical Path" to run CPM analysis</p>
            <p><strong>Step 3:</strong> View results and network diagram</p>
            <br>
            <p>💡 <strong>Tip:</strong> Use "Load Sample Data" to see an example</p>
            <p>🎯 <strong>Note:</strong> Node '1' will be highlighted as the primary start node</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Display current activities
    st.markdown("### 📋 Current Activities")
    df = pd.DataFrame(st.session_state.activities, 
                     columns=['Activity', 'Start', 'End', 'Duration'])
    st.dataframe(df, use_container_width=True)
    
    # Show analysis results if available
    if st.session_state.analysis_results:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Project Duration", f"{st.session_state.project_duration} days")
        with col2:
            critical_count = len(st.session_state.analysis_results.get('critical_activities', []))
            st.metric("Critical Activities", f"{critical_count}")
        with col3:
            total_activities = len(st.session_state.activities)
            st.metric("Total Activities", f"{total_activities}")
        
        # Critical path display with sequence from node 1
        if st.session_state.critical_path_nodes:
            path_str = " → ".join(str(node) for node in st.session_state.critical_path_nodes)
            primary_start = st.session_state.analysis_results.get('primary_start', '1')
            
            st.markdown(f"""
            <div class="critical-path-display">
                🔑 Critical Path from Node {primary_start}: {path_str}<br>
                📊 Total Critical Path Length: {st.session_state.project_duration} days
            </div>
            """, unsafe_allow_html=True)
            
            # Show critical activities sequence
            critical_sequence = app.get_critical_activities_sequence()
            if critical_sequence:
                activities_str = " → ".join([f"{act}({dur})" for act, dur in critical_sequence])
                st.markdown(f"""
                <div class="path-highlight">
                    📋 Critical Activities Sequence: {activities_str}
                </div>
                """, unsafe_allow_html=True)
        
        # Activity analysis table
        st.markdown("### 📊 Activity Analysis")
        analysis_df = pd.DataFrame(st.session_state.analysis_results['activities'])
        
        # Style the dataframe to highlight critical activities
        def highlight_critical(row):
            if row['Float'] <= 0.001:
                return ['background-color: #ffcdd2'] * len(row)
            else:
                return [''] * len(row)
        
        styled_df = analysis_df.style.apply(highlight_critical, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Additional critical path information
        st.markdown("### 🎯 Critical Path Details")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Critical Path Nodes:**")
            for i, node in enumerate(st.session_state.critical_path_nodes):
                node_times = st.session_state.analysis_results['node_times']
                es_time = node_times['es'].get(node, 0)
                if i == 0:
                    st.write(f"🟢 **{node}** (Start Node) - ES: {es_time}")
                else:
                    st.write(f"🔴 **{node}** - ES: {es_time}")
        
        with col2:
            st.markdown("**Critical Activities:**")
            critical_sequence = app.get_critical_activities_sequence()
            for i, (activity, duration) in enumerate(critical_sequence):
                st.write(f"⭐ **Activity {activity}** - Duration: {duration} days")
    
    # Network diagram
    st.markdown("### 🕸️ Network Diagram")
    fig = app.draw_network_diagram()
    if fig:
        st.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
