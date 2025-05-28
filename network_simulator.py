import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import io
import numpy as np
from matplotlib.patches import FancyBboxPatch

# Page configuration
st.set_page_config(
    page_title="CPM Network Diagram Simulator",
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
    .node-critical {
        fill: #ffcdd2;
        stroke: #c62828;
        stroke-width: 3;
    }
    .node-normal {
        fill: #e8f5e8;
        stroke: #2e7d32;
        stroke-width: 2;
    }
    .edge-critical {
        stroke: #e53935;
        stroke-width: 3;
    }
    .edge-normal {
        stroke: #78909c;
        stroke-width: 2;
    }
</style>
""", unsafe_allow_html=True)


class CPMNetworkDiagram:
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

    def validate_activity(self, activity, start_node, end_node, duration):
        """Validate activity before adding to network"""
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
        
        return True, "Validation passed"

    def add_activity(self, activity, start_node, end_node, duration):
        """Add an activity to the network with validation"""
        try:
            activity = activity.strip().upper()
            start_node = start_node.strip()
            end_node = end_node.strip()
            
            valid, msg = self.validate_activity(activity, start_node, end_node, duration)
            if not valid:
                return False, msg
                
            # Test for cycles before adding
            temp_graph = st.session_state.graph.copy()
            temp_graph.add_edge(start_node, end_node, duration=duration, activity=activity)
            if not nx.is_directed_acyclic_graph(temp_graph):
                return False, "Adding this activity would create a cycle"
            
            st.session_state.graph.add_edge(start_node, end_node,
                                          duration=duration, activity=activity)
            st.session_state.activities.append((activity, start_node, end_node, duration))
            
            # Reset analysis
            self.reset_analysis()
            
            return True, f"Activity '{activity}' added successfully"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def reset_analysis(self):
        """Reset all analysis results"""
        st.session_state.critical_path = []
        st.session_state.critical_path_nodes = []
        st.session_state.project_duration = 0
        st.session_state.analysis_results = {}

    def compute_critical_path(self):
        """Proper CPM algorithm implementation with forward and backward passes"""
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
            
            # Initialize time parameters
            es = {node: 0 for node in nodes}  # Early Start
            ef = {node: 0 for node in nodes}  # Early Finish
            ls = {node: float('inf') for node in nodes}  # Late Start
            lf = {node: float('inf') for node in nodes}  # Late Finish
            
            # FORWARD PASS - Calculate ES and EF for nodes
            for node in nx.topological_sort(graph):
                if graph.in_degree(node) == 0:
                    es[node] = 0
                else:
                    es[node] = max(es[pred] + graph[pred][node]['duration'] 
                                for pred in graph.predecessors(node))
                ef[node] = es[node]  # Nodes have zero duration in AON
            
            # Project duration is max EF of end nodes
            project_duration = max(ef[node] for node in end_nodes)
            
            # BACKWARD PASS - Calculate LS and LF for nodes
            for node in reversed(list(nx.topological_sort(graph))):
                if graph.out_degree(node) == 0:
                    lf[node] = project_duration
                else:
                    lf[node] = min(ls[succ] for succ in graph.successors(node))
                ls[node] = lf[node]  # Nodes have zero duration in AON
            
            # Calculate activity times and floats
            activity_analysis = []
            critical_activities = []
            critical_edges = []
            
            for u, v, data in graph.edges(data=True):
                activity = data['activity']
                duration = data['duration']
                
                # Activity times
                activity_es = es[u]
                activity_ef = es[u] + duration
                activity_ls = lf[v] - duration
                activity_lf = lf[v]
                
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
                    'Float': round(total_float, 2),
                    'Critical': abs(total_float) < 0.001
                })
                
                # Identify critical activities
                if abs(total_float) < 0.001:
                    critical_activities.append(activity)
                    critical_edges.append((u, v))
            
            # Find the critical path (longest path)
            longest_path_graph = nx.DiGraph()
            for u, v, data in graph.edges(data=True):
                longest_path_graph.add_edge(u, v, weight=-data['duration'])
            
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
        """Enhanced network diagram with proper CPM visualization"""
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
        
        # Get critical path information
        critical_edges = set(st.session_state.critical_path)
        critical_nodes = set(st.session_state.critical_path_nodes)
        
        # Draw nodes with different styles for critical vs normal
        for node in graph.nodes():
            if node in critical_nodes:
                nx.draw_networkx_nodes(graph, pos, nodelist=[node],
                                     node_color='#ffcdd2',
                                     node_size=4000,
                                     edgecolors='#c62828',
                                     linewidths=3, ax=ax)
            else:
                nx.draw_networkx_nodes(graph, pos, nodelist=[node],
                                     node_color='#e8f5e8',
                                     node_size=4000,
                                     edgecolors='#2e7d32',
                                     linewidths=2, ax=ax)
        
        # Draw edges with different styles for critical vs normal
        for u, v in graph.edges():
            if (u, v) in critical_edges:
                nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)],
                                     edge_color='#e53935',
                                     width=3, arrows=True, arrowsize=25,
                                     arrowstyle='-|>', ax=ax)
            else:
                nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)],
                                     edge_color='#78909c',
                                     width=2, arrows=True, arrowsize=20,
                                     arrowstyle='-|>', ax=ax)
        
        # Node labels with ES/LS values
        if st.session_state.analysis_results:
            node_times = st.session_state.analysis_results.get('node_times', {})
            es_values = node_times.get('es', {})
            ls_values = node_times.get('ls', {})
            
            for node, (x, y) in pos.items():
                es = es_values.get(node, 0)
                ls = ls_values.get(node, 0)
                label = f"{node}\nES:{es:.0f}\nLS:{ls:.0f}"
                color = '#c62828' if node in critical_nodes else '#2e7d32'
                ax.text(x, y, label, ha='center', va='center',
                       fontsize=10, fontweight='bold', color=color,
                       bbox=dict(boxstyle="round,pad=0.3", 
                               facecolor='white', 
                               alpha=0.8))
        
        # Edge labels with activity info
        for u, v, data in graph.edges(data=True):
            activity = data['activity']
            duration = data['duration']
            is_critical = (u, v) in critical_edges
            
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2 + 0.1
            
            ax.text(x, y, f"{activity}({duration})", 
                   ha='center', va='center',
                   fontsize=10, 
                   fontweight='bold' if is_critical else 'normal',
                   color='#b71c1c' if is_critical else '#37474f',
                   bbox=dict(boxstyle="round,pad=0.2", 
                           facecolor='#ffcdd2' if is_critical else 'white', 
                           alpha=0.8))
        
        # Diagram title and project duration
        ax.set_title("CPM Network Diagram (Activity-on-Node)", 
                    fontsize=18, fontweight='bold', pad=30)
        
        if st.session_state.project_duration:
            ax.text(0.5, 0.95, f"Project Duration: {st.session_state.project_duration} days",
                   transform=ax.transAxes, ha='center', fontsize=14,
                   color='#d32f2f', weight='bold')
        
        ax.axis('off')
        plt.tight_layout()
        return fig

    def add_dummy_data(self):
        """Add sample project data for demonstration"""
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
        """Clear all activities and reset analysis"""
        st.session_state.activities = []
        st.session_state.graph = nx.DiGraph()
        self.reset_analysis()
        return True, "All activities cleared"

    def remove_last_activity(self):
        """Remove the last added activity"""
        if not st.session_state.activities:
            return False, "No activities to remove"
        try:
            last_activity = st.session_state.activities.pop()
            activity, start, end, duration = last_activity
            st.session_state.graph.remove_edge(start, end)
            self.reset_analysis()
            return True, f"Activity '{activity}' removed"
        except Exception as e:
            return False, f"Error removing activity: {str(e)}"


def main():
    app = CPMNetworkDiagram()
    
    st.markdown("""
    <div class="main-header">
        <h1>📊 CPM Network Diagram Simulator</h1>
        <p>Critical Path Method Analysis Tool (Activity-on-Node)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for input
    with st.sidebar:
        st.markdown("### 📝 Add New Activity")
        with st.form("activity_form", clear_on_submit=True):
            activity = st.text_input("Activity ID", placeholder="A", max_chars=5)
            col1, col2 = st.columns(2)
            with col1:
                start_node = st.text_input("Start Node", placeholder="1", max_chars=5)
            with col2:
                end_node = st.text_input("End Node", placeholder="2", max_chars=5)
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
            if st.button("↩️ Undo Last Activity"):
                success, message = app.remove_last_activity()
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.warning(message)
        with col2:
            if st.button("🗑️ Clear All"):
                success, message = app.clear_all_activities()
                if success:
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
                     columns=['Activity', 'Start Node', 'End Node', 'Duration'])
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
            if row['Critical']:
                return ['background-color: #ffcdd2'] * len(row)
            return [''] * len(row)
        
        styled_df = analysis_df.style.apply(highlight_critical, axis=1)
        st.dataframe(styled_df, use_container_width=True,
                    column_order=['Activity', 'Start_Node', 'End_Node', 'Duration', 
                                'ES', 'EF', 'LS', 'LF', 'Float', 'Critical'])
    
    # Network diagram
    st.markdown("### 🕸️ Network Diagram")
    fig = app.draw_network_diagram()
    if fig:
        st.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
