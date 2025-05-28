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

# CSS Styling (shortened for space)
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
        """CORRECTED: Proper CPM algorithm implementation"""
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
            
            # Initialize time values
            es = {node: 0 for node in nodes}  # Early Start
            ef = {node: 0 for node in nodes}  # Early Finish
            ls = {node: float('inf') for node in nodes}  # Late Start
            lf = {node: float('inf') for node in nodes}  # Late Finish
            
            # FORWARD PASS - Calculate ES and EF
            topo_order = list(nx.topological_sort(graph))
            
            for node in topo_order:
                # ES = max(EF of all predecessors)
                if graph.in_degree(node) == 0:
                    es[node] = 0
                else:
                    es[node] = max(ef[pred] for pred in graph.predecessors(node))
                
                # EF = ES + duration (for this node, we need to consider outgoing activities)
                if graph.out_degree(node) == 0:
                    ef[node] = es[node]  # End nodes have no duration
                else:
                    # For nodes with outgoing activities, EF = ES (node duration is in the edges)
                    ef[node] = es[node]
            
            # Calculate activity-based EF values
            activity_ef = {}
            for u, v, data in graph.edges(data=True):
                duration = data['duration']
                activity_ef[(u, v)] = es[u] + duration
                # Update successor's ES if this path is longer
                if activity_ef[(u, v)] > es[v]:
                    es[v] = activity_ef[(u, v)]
                    ef[v] = es[v]
            
            # Recalculate with proper forward pass
            for node in topo_order:
                if graph.in_degree(node) == 0:
                    es[node] = 0
                    ef[node] = 0
                else:
                    incoming_activities = [(pred, node) for pred in graph.predecessors(node)]
                    if incoming_activities:
                        es[node] = max(es[pred] + graph[pred][node]['duration'] 
                                     for pred in graph.predecessors(node))
                        ef[node] = es[node]
            
            # Project duration = max EF of end nodes
            project_duration = max(ef[node] for node in end_nodes)
            
            # BACKWARD PASS - Calculate LS and LF
            reverse_topo = list(reversed(topo_order))
            
            # Initialize end nodes
            for node in end_nodes:
                lf[node] = ef[node]  # LF = EF for end nodes
                ls[node] = es[node]  # LS = ES for end nodes
            
            for node in reverse_topo:
                if graph.out_degree(node) == 0:
                    # End nodes already initialized
                    continue
                else:
                    # LF = min(LS of all successors)
                    successors = list(graph.successors(node))
                    if successors:
                        lf[node] = min(ls[succ] - graph[node][succ]['duration'] 
                                     for succ in successors)
                        ls[node] = lf[node]
            
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
            
            # Find critical path using longest path algorithm
            # Create a copy with negative weights for longest path
            longest_path_graph = nx.DiGraph()
            for u, v, data in graph.edges(data=True):
                longest_path_graph.add_edge(u, v, weight=-data['duration'])
            
            # Find longest path from each start node to each end node
            critical_path_nodes = []
            max_length = 0
            
            for start in start_nodes:
                for end in end_nodes:
                    try:
                        path = nx.shortest_path(longest_path_graph, start, end, weight='weight')
                        length = -nx.shortest_path_length(longest_path_graph, start, end, weight='weight')
                        if length > max_length:
                            max_length = length
                            critical_path_nodes = path
                    except nx.NetworkXNoPath:
                        continue
            
            # Store results
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

    def draw_network_diagram(self):
        """CORRECTED: Enhanced network diagram with proper CPM visualization"""
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
        node_border_color = '#2e7d32'
        critical_border_color = '#c62828'
        regular_edge_color = '#78909c'
        critical_edge_color = '#e53935'
        
        critical_edges = set(st.session_state.critical_path)
        critical_nodes = set(st.session_state.critical_path_nodes)
        
        all_nodes = list(graph.nodes())
        regular_nodes = [n for n in all_nodes if n not in critical_nodes]
        
        # Draw nodes
        if regular_nodes:
            nx.draw_networkx_nodes(graph, pos, nodelist=regular_nodes,
                                 node_color=regular_node_color,
                                 node_size=4000,
                                 edgecolors=node_border_color,
                                 linewidths=3, ax=ax)
        
        if critical_nodes:
            nx.draw_networkx_nodes(graph, pos, nodelist=list(critical_nodes),
                                 node_color=critical_node_color,
                                 node_size=4000,
                                 edgecolors=critical_border_color,
                                 linewidths=4, ax=ax)
        
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
        
        # Node labels with ES/EF values
        if st.session_state.analysis_results:
            node_times = st.session_state.analysis_results.get('node_times', {})
            es_values = node_times.get('es', {})
            ef_values = node_times.get('ef', {})
            
            for node, (x, y) in pos.items():
                es = es_values.get(node, 0)
                ef = ef_values.get(node, 0)
                label = f"{node}\nES:{es:.0f}"
                color = 'darkred' if node in critical_nodes else 'darkgreen'
                ax.text(x, y, label, ha='center', va='center',
                       fontsize=11, fontweight='bold', color=color,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
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
                   fontsize=10, fontweight=weight, color=color,
                   bbox=dict(boxstyle="round,pad=0.2", 
                           facecolor='#ffcdd2' if is_critical else 'white', 
                           alpha=0.8))
        
        ax.set_title("CPM Network Diagram - Critical Path Analysis",
                    fontsize=18, fontweight='bold', pad=30)
        
        if st.session_state.project_duration:
            ax.text(0.5, 0.95, f"Project Duration: {st.session_state.project_duration} days",
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
        <p>Critical Path Method Analysis Tool</p>
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
        
        # Critical path display
        if st.session_state.critical_path_nodes:
            path_str = " → ".join(str(node) for node in st.session_state.critical_path_nodes)
            activities_str = " → ".join(st.session_state.analysis_results['critical_activities'])
            st.markdown(f"""
            <div class="critical-path-display">
                🔑 Critical Path: {path_str}<br>
                📋 Critical Activities: {activities_str}
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
    
    # Network diagram
    st.markdown("### 🕸️ Network Diagram")
    fig = app.draw_network_diagram()
    if fig:
        st.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
