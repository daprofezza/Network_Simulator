import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from matplotlib.patches import ArrowStyle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Network Diagram Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4a6baf 0%, #6c757d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4a6baf;
        margin: 0.5rem 0;
    }
    .critical-path {
        background: linear-gradient(90deg, #ff6b6b 0%, #ff8a5b 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 4px;
        border: 1px solid #c3e6cb;
        margin: 0.5rem 0;
    }
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 4px;
        border: 1px solid #f5c6cb;
        margin: 0.5rem 0;
    }
    .stDataFrame {
        background: white;
        border-radius: 8px;
        padding: 1rem;
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
        if 'project_duration' not in st.session_state:
            st.session_state.project_duration = 0
        if 'float_table' not in st.session_state:
            st.session_state.float_table = []

    def add_activity(self, activity, start_node, end_node, duration):
        """Add an activity to the network"""
        try:
            # Validation
            if not all([activity.strip(), start_node.strip(), end_node.strip()]):
                return False, "All fields must be filled"
            
            if duration <= 0:
                return False, "Duration must be positive"
            
            if start_node == end_node:
                return False, "Start and end nodes must be different"
            
            # Check for duplicate edges
            if (start_node, end_node) in st.session_state.graph.edges:
                return False, "This connection already exists"
            
            # Add to graph and activities list
            st.session_state.graph.add_edge(start_node, end_node, 
                                          duration=duration, activity=activity)
            st.session_state.activities.append((activity, start_node, end_node, duration))
            
            return True, "Activity added successfully"
            
        except Exception as e:
            return False, f"Error: {str(e)}"

    def remove_last_activity(self):
        """Remove the last added activity"""
        if st.session_state.activities:
            last_activity = st.session_state.activities.pop()
            activity, start, end, duration = last_activity
            try:
                st.session_state.graph.remove_edge(start, end)
                # Clear calculations
                st.session_state.critical_path = []
                st.session_state.float_table = []
                st.session_state.project_duration = 0
                return True, "Last activity removed"
            except:
                # If error, add back the activity
                st.session_state.activities.append(last_activity)
                return False, "Error removing activity"
        return False, "No activities to remove"

    def clear_all_activities(self):
        """Clear all activities and reset the graph"""
        st.session_state.activities = []
        st.session_state.graph = nx.DiGraph()
        st.session_state.critical_path = []
        st.session_state.project_duration = 0
        st.session_state.float_table = []
        return True, "All activities cleared"

    def compute_critical_path(self):
        """Compute the critical path using CPM algorithm"""
        try:
            if not st.session_state.activities:
                return False, "No activities to analyze"

            graph = st.session_state.graph
            durations = nx.get_edge_attributes(graph, 'duration')
            
            # Check if graph is acyclic
            if not nx.is_directed_acyclic_graph(graph):
                return False, "Network contains cycles - invalid for CPM"
            
            topo_order = list(nx.topological_sort(graph))
            es, ef, ls, lf = {}, {}, {}, {}

            # Forward pass - calculate Early Start and Early Finish
            for node in topo_order:
                predecessors = list(graph.predecessors(node))
                if predecessors:
                    es[node] = max(ef[pred] for pred in predecessors)
                else:
                    es[node] = 0
                
                successors = list(graph.successors(node))
                if successors:
                    ef[node] = es[node] + max(durations.get((node, succ), 0) for succ in successors)
                else:
                    ef[node] = es[node]

            # Project duration is the maximum EF
            project_duration = max(ef.values()) if ef else 0

            # Backward pass - calculate Late Start and Late Finish
            for node in reversed(topo_order):
                successors = list(graph.successors(node))
                if successors:
                    lf[node] = min(ls[succ] for succ in successors)
                else:
                    lf[node] = project_duration
                
                predecessors = list(graph.predecessors(node))
                if predecessors:
                    incoming_durations = [durations.get((pred, node), 0) for pred in predecessors]
                    ls[node] = lf[node] - max(incoming_durations) if incoming_durations else lf[node]
                else:
                    ls[node] = lf[node]

            # Calculate floats and identify critical activities
            critical_path = []
            float_table = []
            
            for u, v, data in graph.edges(data=True):
                activity = data['activity']
                duration = data['duration']
                
                es_u = es[u]
                ef_u = es_u + duration
                ls_u = ls[u]
                lf_v = lf[v]
                
                # Total float calculation
                total_float = lf_v - ef_u
                
                float_table.append({
                    'Activity': activity,
                    'ES': es_u,
                    'EF': ef_u,
                    'LS': ls_u,
                    'LF': lf_v,
                    'Float': total_float
                })
                
                if total_float == 0:  # Critical activity
                    critical_path.append((u, v))

            # Update session state
            st.session_state.critical_path = critical_path
            st.session_state.project_duration = project_duration
            st.session_state.float_table = float_table

            return True, f"Critical path computed successfully. Project duration: {project_duration} days"

        except Exception as e:
            return False, f"Error computing critical path: {str(e)}"

    def draw_network_diagram(self, highlight_critical=True):
        """Draw the network diagram using matplotlib"""
        if not st.session_state.activities:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#ffffff')

        graph = st.session_state.graph
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(graph, seed=42, k=1.5, iterations=50)
        
        # Color schemes
        base_node_color = '#e6f3ff'
        base_edge_color = '#a0aec0'
        critical_node_color = '#ff6b6b'
        critical_edge_color = '#ff3860'
        
        # Get critical nodes
        critical_edges = set(st.session_state.critical_path)
        critical_nodes = set()
        for u, v in critical_edges:
            critical_nodes.add(u)
            critical_nodes.add(v)
        
        # Draw all nodes
        regular_nodes = [n for n in graph.nodes if n not in critical_nodes]
        
        if regular_nodes:
            nx.draw_networkx_nodes(graph, pos, nodelist=regular_nodes,
                                 ax=ax, node_color=base_node_color,
                                 node_size=1500, edgecolors='#4a6baf', linewidths=2)
        
        if critical_nodes and highlight_critical:
            nx.draw_networkx_nodes(graph, pos, nodelist=list(critical_nodes),
                                 ax=ax, node_color=critical_node_color,
                                 node_size=1500, edgecolors='#d63384', linewidths=3)
        
        # Draw edges
        regular_edges = [(u, v) for u, v in graph.edges if (u, v) not in critical_edges]
        
        if regular_edges:
            nx.draw_networkx_edges(graph, pos, edgelist=regular_edges,
                                 ax=ax, edge_color=base_edge_color,
                                 width=2, arrows=True, arrowsize=20,
                                 arrowstyle='-|>', node_size=1500)
        
        if critical_edges and highlight_critical:
            nx.draw_networkx_edges(graph, pos, edgelist=list(critical_edges),
                                 ax=ax, edge_color=critical_edge_color,
                                 width=3, arrows=True, arrowsize=20,
                                 arrowstyle='-|>', node_size=1500)
        
        # Draw node labels
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=14,
                              font_weight='bold', font_color='#2c3e50')
        
        # Draw edge labels
        edge_labels = {}
        for u, v, data in graph.edges(data=True):
            edge_labels[(u, v)] = f"{data['activity']}\n({data['duration']})"
        
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels,
                                   ax=ax, font_size=10,
                                   bbox=dict(boxstyle='round,pad=0.2',
                                           facecolor='white',
                                           edgecolor='gray',
                                           alpha=0.8))
        
        ax.set_title("Project Network Diagram", fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=base_node_color, edgecolor='#4a6baf', label='Regular Activity'),
            Patch(facecolor=critical_node_color, edgecolor='#d63384', label='Critical Activity')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        return fig

    def add_dummy_data(self):
        """Add sample data for demonstration"""
        if st.session_state.activities:
            return False, "Clear existing data before loading sample data"
        
        dummy_activities = [
            ("A", "1", "2", 4),
            ("B", "1", "3", 3),
            ("C", "2", "4", 2),
            ("D", "3", "4", 5),
            ("E", "4", "5", 1)
        ]
        
        for activity, start, end, duration in dummy_activities:
            st.session_state.graph.add_edge(start, end, duration=duration, activity=activity)
            st.session_state.activities.append((activity, start, end, duration))
        
        return True, "Sample data loaded successfully"

def main():
    app = NetworkDiagramApp()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Network Diagram Simulator</h1>
        <p>Critical Path Method (CPM) Analysis Tool</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for input
    with st.sidebar:
        st.header("📝 Add Activity")
        
        with st.form("activity_form"):
            activity = st.text_input("Activity ID", placeholder="e.g., A, B, C")
            start_node = st.text_input("Start Node", placeholder="e.g., 1, 2")
            end_node = st.text_input("End Node", placeholder="e.g., 2, 3")
            duration = st.number_input("Duration (days)", min_value=1, value=1)
            
            col1, col2 = st.columns(2)
            with col1:
                submit_button = st.form_submit_button("➕ Add Activity", use_container_width=True)
            with col2:
                dummy_button = st.form_submit_button("🎲 Load Sample", use_container_width=True)
        
        # Handle form submissions
        if submit_button:
            success, message = app.add_activity(activity, start_node, end_node, duration)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
        
        if dummy_button:
            success, message = app.add_dummy_data()
            if success:
                st.success(message)
                st.rerun()
            else:
                st.warning(message)
        
        st.divider()
        
        # Action buttons
        st.header("🛠️ Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("↩️ Undo Last", use_container_width=True):
                success, message = app.remove_last_activity()
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.warning(message)
        
        with col2:
            if st.button("🗑️ Clear All", use_container_width=True):
                success, message = app.clear_all_activities()
                st.success(message)
                st.rerun()
        
        if st.button("🔍 Compute Critical Path", use_container_width=True, type="primary"):
            success, message = app.compute_critical_path()
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    # Main content area
    if st.session_state.activities:
        # Create activity dataframe
        activity_df = pd.DataFrame(st.session_state.activities, 
                                 columns=['Activity', 'Start Node', 'End Node', 'Duration'])
        
        # Add computed values if available
        if st.session_state.float_table:
            float_df = pd.DataFrame(st.session_state.float_table)
            # Merge the dataframes
            merged_df = activity_df.merge(float_df[['Activity', 'ES', 'EF', 'LS', 'LF', 'Float']], 
                                        on='Activity', how='left')
            display_df = merged_df
        else:
            # Add empty columns for ES, EF, LS, LF, Float
            for col in ['ES', 'EF', 'LS', 'LF', 'Float']:
                activity_df[col] = ''
            display_df = activity_df
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Activity Table")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("📊 Project Metrics")
            
            if st.session_state.project_duration > 0:
                st.metric("Project Duration", f"{st.session_state.project_duration} days")
                
                # Critical path display
                if st.session_state.critical_path:
                    critical_activities = []
                    for u, v in st.session_state.critical_path:
                        for act, s, e, d in st.session_state.activities:
                            if s == u and e == v:
                                critical_activities.append(act)
                                break
                    
                    st.markdown(f"""
                    <div class="critical-path">
                        🔑 Critical Path<br>
                        {' → '.join(critical_activities)}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Float analysis
                if st.session_state.float_table:
                    critical_count = sum(1 for item in st.session_state.float_table if item['Float'] == 0)
                    total_activities = len(st.session_state.float_table)
                    
                    st.metric("Critical Activities", f"{critical_count}/{total_activities}")
                    
                    avg_float = np.mean([item['Float'] for item in st.session_state.float_table])
                    st.metric("Average Float", f"{avg_float:.1f} days")
            else:
                st.info("🔍 Click 'Compute Critical Path' to analyze the network")
        
        # Display network diagram
        st.subheader("🕸️ Network Diagram")
        
        fig = app.draw_network_diagram(highlight_critical=st.session_state.project_duration > 0)
        if fig:
            st.pyplot(fig, use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Save plot as PNG
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                
                st.download_button(
                    label="💾 Download Diagram (PNG)",
                    data=img_buffer.getvalue(),
                    file_name="network_diagram.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with col2:
                # Export data as CSV
                csv_buffer = io.StringIO()
                display_df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="📤 Export Data (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="project_activities.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            plt.close(fig)  # Clean up memory
    
    else:
        st.info("👈 Add activities using the sidebar to get started, or load sample data to see a demonstration.")
        
        # Show some help text
        with st.expander("ℹ️ How to use this application"):
            st.markdown("""
            **Step 1: Add Activities**
            - Enter a unique Activity ID (e.g., A, B, C)
            - Specify Start and End nodes (e.g., 1, 2, 3)
            - Enter the duration in days
            - Click "Add Activity"
            
            **Step 2: Build Your Network**
            - Add multiple activities that connect to form a project network
            - Activities should connect start-to-finish to show dependencies
            
            **Step 3: Analyze**
            - Click "Compute Critical Path" to perform CPM analysis
            - View the critical path, project duration, and activity floats
            - The diagram will highlight critical activities in red
            
            **Tips:**
            - Use "Load Sample" to see an example project
            - Critical activities have zero float and determine project duration
            - Non-critical activities have positive float (slack time)
            """)

    # Footer
    st.markdown("---")
    st.markdown(
        "© Network Diagram Simulator | Web App Version | "
        "Original by J. Inigo Papu Vinodhan, Asst. Prof., BBA Dept., St. Joseph's College, Trichy"
    )

if __name__ == "__main__":
    main()
