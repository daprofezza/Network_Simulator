import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict

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
    .node-info {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
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
        """Enhanced CPM algorithm with proper critical path identification"""
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
            
            # Initialize time values for nodes
            es = {node: 0 for node in nodes}  # Early Start
            ef = {node: 0 for node in nodes}  # Early Finish
            ls = {node: 0 for node in nodes}  # Late Start
            lf = {node: 0 for node in nodes}  # Late Finish
            
            # FORWARD PASS - Calculate ES and EF for nodes
            topo_order = list(nx.topological_sort(graph))
            
            for node in topo_order:
                if graph.in_degree(node) == 0:
                    es[node] = 0
                    ef[node] = 0
                else:
                    # ES = max(EF of all predecessors + duration of connecting activity)
                    max_ef = 0
                    for pred in graph.predecessors(node):
                        activity_duration = graph[pred][node]['duration']
                        pred_completion = ef[pred] + activity_duration
                        max_ef = max(max_ef, pred_completion)
                    es[node] = max_ef
                    ef[node] = max_ef
            
            # Project duration = max EF of end nodes
            project_duration = max(ef[node] for node in end_nodes)
            
            # BACKWARD PASS - Calculate LS and LF for nodes
            reverse_topo = list(reversed(topo_order))
            
            # Initialize end nodes
            for node in end_nodes:
                lf[node] = ef[node]
                ls[node] = es[node]
            
            for node in reverse_topo:
                if graph.out_degree(node) == 0:
                    continue
                else:
                    # LF = min(LS of all successors)
                    min_ls = float('inf')
                    for succ in graph.successors(node):
                        activity_duration = graph[node][succ]['duration']
                        succ_latest_start = ls[succ] - activity_duration
                        min_ls = min(min_ls, succ_latest_start)
                    lf[node] = min_ls
                    ls[node] = min_ls
            
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
            
            # Find critical path through the network
            critical_path_nodes = self.find_critical_path_nodes(graph, critical_edges, start_nodes, end_nodes)
            
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

    def find_critical_path_nodes(self, graph, critical_edges, start_nodes, end_nodes):
        """Find the sequence of nodes that form the critical path"""
        if not critical_edges:
            return []
        
        # Build critical subgraph
        critical_graph = nx.DiGraph()
        for u, v in critical_edges:
            critical_graph.add_edge(u, v)
        
        # Find the longest path through critical activities
        longest_path = []
        max_length = 0
        
        for start in start_nodes:
            if start in critical_graph.nodes():
                for end in end_nodes:
                    if end in critical_graph.nodes():
                        try:
                            if nx.has_path(critical_graph, start, end):
                                # Get all simple paths and find the longest
                                paths = list(nx.all_simple_paths(critical_graph, start, end))
                                for path in paths:
                                    path_length = len(path)
                                    if path_length > max_length:
                                        max_length = path_length
                                        longest_path = path
                        except:
                            continue
        
        return longest_path

    def draw_network_diagram(self):
        """Enhanced network diagram with improved critical path highlighting"""
        if not st.session_state.activities:
            return None
            
        fig, ax = plt.subplots(figsize=(16, 12))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        graph = st.session_state.graph
        
        # Use spring layout with better positioning
        pos = nx.spring_layout(graph, seed=42, k=2.5, iterations=200)
        
        # Color scheme
        regular_node_color = '#e8f5e8'
        critical_node_color = '#ffcdd2'
        start_node_color = '#c8e6c9'
        end_node_color = '#ffecb3'
        node_border_color = '#2e7d32'
        critical_border_color = '#c62828'
        start_border_color = '#388e3c'
        end_border_color = '#f57c00'
        regular_edge_color = '#78909c'
        critical_edge_color = '#e53935'
        
        critical_edges = set(st.session_state.critical_path)
        critical_nodes = set(st.session_state.critical_path_nodes)
        
        # Get node classification
        start_nodes = set()
        end_nodes = set()
        if st.session_state.analysis_results:
            start_nodes = set(st.session_state.analysis_results.get('start_nodes', []))
            end_nodes = set(st.session_state.analysis_results.get('end_nodes', []))
        
        all_nodes = list(graph.nodes())
        
        # Draw nodes with different colors based on their role
        for node in all_nodes:
            if node in critical_nodes:
                color = critical_node_color
                border_color = critical_border_color
                border_width = 4
            elif node in start_nodes:
                color = start_node_color
                border_color = start_border_color
                border_width = 3
            elif node in end_nodes:
                color = end_node_color
                border_color = end_border_color
                border_width = 3
            else:
                color = regular_node_color
                border_color = node_border_color
                border_width = 2
            
            nx.draw_networkx_nodes(graph, pos, nodelist=[node],
                                 node_color=color,
                                 node_size=4500,
                                 edgecolors=border_color,
                                 linewidths=border_width, ax=ax)
        
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
                                 width=5, arrows=True, arrowsize=30,
                                 arrowstyle='-|>', ax=ax)
        
        # Enhanced node labels with timing information
        if st.session_state.analysis_results:
            node_times = st.session_state.analysis_results.get('node_times', {})
            es_values = node_times.get('es', {})
            ls_values = node_times.get('ls', {})
            
            for node, (x, y) in pos.items():
                es = es_values.get(node, 0)
                ls = ls_values.get(node, 0)
                
                if node in critical_nodes:
                    label = f"{node}\nES/LS: {es:.0f}"
                    color = 'darkred'
                    bbox_color = '#ffebee'
                elif node in start_nodes:
                    label = f"{node}\n(START)\nES: {es:.0f}"
                    color = 'darkgreen'
                    bbox_color = '#e8f5e8'
                elif node in end_nodes:
                    label = f"{node}\n(END)\nLS: {ls:.0f}"
                    color = 'darkorange'
                    bbox_color = '#fff3e0'
                else:
                    label = f"{node}\nES: {es:.0f}\nLS: {ls:.0f}"
                    color = 'darkblue'
                    bbox_color = 'white'
                
                ax.text(x, y, label, ha='center', va='center',
                       fontsize=10, fontweight='bold', color=color,
                       bbox=dict(boxstyle="round,pad=0.4", facecolor=bbox_color, alpha=0.9))
        
        # Enhanced edge labels
        edge_labels = {}
        for u, v, data in graph.edges(data=True):
            activity = data['activity']
            duration = data['duration']
            is_critical = (u, v) in critical_edges
            
            if is_critical:
                edge_labels[(u, v)] = f"★ {activity} ({duration}) ★"
            else:
                edge_labels[(u, v)] = f"{activity} ({duration})"
        
        # Position edge labels with better spacing
        for (u, v), label in edge_labels.items():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2 + 0.08
            is_critical = (u, v) in critical_edges
            
            if is_critical:
                color = '#b71c1c'
                weight = 'bold'
                bbox_color = '#ffcdd2'
                fontsize = 11
            else:
                color = '#37474f'
                weight = 'normal'
                bbox_color = 'white'
                fontsize = 9
            
            ax.text(x, y, label, ha='center', va='center',
                   fontsize=fontsize, fontweight=weight, color=color,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=bbox_color, alpha=0.9))
        
        # Enhanced title and information
        title = "CPM Network Diagram - Critical Path Analysis"
        if st.session_state.critical_path_nodes:
            path_str = " → ".join(str(node) for node in st.session_state.critical_path_nodes)
            title += f"\n🔴 Critical Path: {path_str}"
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=40)
        
        if st.session_state.project_duration:
            ax.text(0.5, 0.02, f"Project Duration: {st.session_state.project_duration} days",
                   transform=ax.transAxes, ha='center', fontsize=14,
                   color='#d32f2f', weight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='#ffebee', alpha=0.8))
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=critical_node_color, 
                      markersize=15, markeredgecolor=critical_border_color, markeredgewidth=3,
                      label='Critical Path Nodes'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=start_node_color, 
                      markersize=15, markeredgecolor=start_border_color, markeredgewidth=2,
                      label='Start Nodes'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=end_node_color, 
                      markersize=15, markeredgecolor=end_border_color, markeredgewidth=2,
                      label='End Nodes'),
            plt.Line2D([0], [0], color=critical_edge_color, linewidth=4, label='Critical Activities'),
            plt.Line2D([0], [0], color=regular_edge_color, linewidth=2, label='Regular Activities')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        ax.axis('off')
        plt.tight_layout()
        return fig

    def add_dummy_data(self):
        """Add sample project data for testing critical path from node 1"""
        if st.session_state.activities:
            return False, "Clear existing data first"
        
        # Sample project with clear critical path starting from node 1
        dummy_activities = [
            ("A", "1", "2", 5),  # Activity A starts the critical path
            ("B", "1", "3", 3),
            ("C", "2", "4", 4),  # Part of critical path
            ("D", "3", "4", 2),
            ("E", "4", "5", 6),  # Critical activity
            ("F", "3", "5", 4),
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
        <h1>📊 Enhanced CPM Network Diagram Simulator</h1>
        <p>Critical Path Method Analysis with Proper Node Highlighting</p>
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
            <p><strong>Step 3:</strong> View results with highlighted critical path from node 1</p>
            <br>
            <p>💡 <strong>Tip:</strong> Use "Load Sample Data" to see critical path highlighting</p>
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
        
        # Critical path display with enhanced information
        if st.session_state.critical_path_nodes:
            path_str = " → ".join(str(node) for node in st.session_state.critical_path_nodes)
            activities_str = " → ".join(st.session_state.analysis_results['critical_activities'])
            st.markdown(f"""
            <div class="critical-path-display">
                🔑 Critical Path Nodes: {path_str}<br>
                📋 Critical Activities: {activities_str}<br>
                🎯 Starting from Node: {st.session_state.critical_path_nodes[0] if st.session_state.critical_path_nodes else 'N/A'}
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed critical path analysis
            if st.session_state.critical_path_nodes and 'A' in [act[0] for act in st.session_state.activities]:
                st.markdown("### 🎯 Critical Path Analysis from Starting Node")
                
                # Check if Activity A is critical
                activity_a_critical = 'A' in st.session_state.analysis_results.get('critical_activities', [])
                
                if activity_a_critical:
                    st.success("✅ Activity 'A' is on the critical path!")
                else:
                    st.warning("⚠️ Activity 'A' is not on the critical path.")
                
                # Show path details
                st.markdown(f"""
                <div class="node-info">
                    <strong>Critical Path Details:</strong><br>
                    • Path: {path_str}<br>
                    • Total Duration: {st.session_state.project_duration} days<br>
                    • Critical Activities: {len(st.session_state.analysis_results.get('critical_activities', []))}<br>
                    • Float = 0 for all critical activities
                </div>
                """, unsafe_allow_html=True)
        
        # Activity analysis table with highlighting
        st.markdown("### 📊 Detailed Activity Analysis")
        analysis_df = pd.DataFrame(st.session_state.analysis_results['activities'])
        
        # Enhanced styling for critical activities
        def highlight_critical(row):
            if row['Float'] <= 0.001:
                return ['background-color: #ffcdd2; font-weight: bold'] * len(row)
            else:
                return [''] * len(row)
        
        styled_df = analysis_df.style.apply(highlight_critical, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Float analysis
        st.markdown("### ⏱️ Float Analysis")
        col1, col2 = st.columns(2)
        with col1:
            critical_activities = [act for act in st.session_state.analysis_results['activities'] if act['Float'] <= 0.001]
            st.write("**Critical Activities (Float = 0):**")
            for act in critical_activities:
                st.write(f"• {act['Activity']}: {act['Start_Node']} → {act['End_Node']}")
        
        with col2:
            non_critical_activities = [act for act in st.session_state.analysis_results['activities'] if act['Float'] > 0.001]
            st.write("**Non-Critical Activities (Float > 0):**")
            for act in non_critical_activities:
                st.write(f"• {act['Activity']}: Float = {act['Float']} days")
    
    # Enhanced network diagram
    st.markdown("### 🕸️ Network Diagram with Critical Path Highlighting")
    fig = app.draw_network_diagram()
    if fig:
        st.pyplot(fig)
        plt.close(fig)
        
        # Additional information below diagram
        if st.session_state.critical_path_nodes:
            st.markdown("""
            **Legend:**
            - 🔴 **Red nodes/edges**: Critical path (zero float)
            - 🟢 **Green nodes**: Start nodes
            - 🟡 **Yellow nodes**: End nodes  
            - ⭐ **Starred activities**: Critical activities
            - **ES/LS values**: Early Start / Late Start times
            """)


if __name__ == "__main__":
    main()
