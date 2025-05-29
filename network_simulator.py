import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import io
import numpy as np
# from matplotlib.patches import FancyBboxPatch # Not explicitly used, can be removed if not needed later

# Page configuration
st.set_page_config(
    page_title="SJC Network Diagram Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    .critical-pulse {
        animation: pulse 2s ease-in-out infinite;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #5D4037;
        color: #FFFFFF;
        text-align: center;
        padding: 10px 0;
        font-size: 0.8rem;
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-container { /* This class is defined but not explicitly used on st.metric by default */
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
        if 'critical_path' not in st.session_state: # Stores critical edges (u,v)
            st.session_state.critical_path = []
        if 'critical_path_nodes' not in st.session_state: # Stores all unique nodes on any critical path
            st.session_state.critical_path_nodes = []
        if 'project_duration' not in st.session_state:
            st.session_state.project_duration = 0
        if 'analysis_results' not in st.session_state: # Stores detailed analysis, including critical activity names
            st.session_state.analysis_results = {}

    def add_activity(self, activity_id, start_node, end_node, duration):
        """Add an activity to the network with validation"""
        try:
            activity_id = activity_id.strip().upper()
            start_node = start_node.strip()
            end_node = end_node.strip()
            if not all([activity_id, start_node, end_node]):
                return False, "Activity ID, Start Node, and End Node fields must be filled."
            if duration <= 0:
                return False, "Duration must be a positive number."
            if start_node == end_node:
                return False, "Start and End nodes must be different for an activity."
            
            existing_activity_ids = [act[0] for act in st.session_state.activities]
            if activity_id in existing_activity_ids:
                return False, f"Activity ID '{activity_id}' already exists. Please use a unique ID."
            
            if st.session_state.graph.has_edge(start_node, end_node):
                return False, f"An activity already exists between nodes '{start_node}' and '{end_node}'."

            st.session_state.graph.add_edge(start_node, end_node,
                                            duration=duration, activity=activity_id)
            st.session_state.activities.append((activity_id, start_node, end_node, duration))
            
            st.session_state.critical_path = []
            st.session_state.critical_path_nodes = []
            st.session_state.project_duration = 0
            st.session_state.analysis_results = {}
            return True, f"Activity '{activity_id}' added successfully."
        except Exception as e:
            return False, f"Error adding activity: {str(e)}"

    def remove_last_activity(self):
        """Remove the last added activity"""
        if not st.session_state.activities:
            return False, "No activities to remove."
        try:
            last_activity_tuple = st.session_state.activities.pop()
            activity_id, start, end, _ = last_activity_tuple
            if st.session_state.graph.has_edge(start, end):
                st.session_state.graph.remove_edge(start, end)
            
            nodes_to_remove = [node for node in [start, end] if node in st.session_state.graph and st.session_state.graph.degree(node) == 0]
            for node in nodes_to_remove:
                st.session_state.graph.remove_node(node)

            st.session_state.critical_path = []
            st.session_state.critical_path_nodes = []
            st.session_state.project_duration = 0
            st.session_state.analysis_results = {}
            return True, f"Activity '{activity_id}' removed successfully."
        except Exception as e:
            if 'last_activity_tuple' in locals():
                 st.session_state.activities.append(last_activity_tuple)
            return False, f"Error removing activity: {str(e)}"

    def clear_all_activities(self):
        """Clear all activities and reset the application state"""
        st.session_state.activities = []
        st.session_state.graph = nx.DiGraph()
        st.session_state.critical_path = []
        st.session_state.critical_path_nodes = []
        st.session_state.project_duration = 0
        st.session_state.analysis_results = {}
        return True, "All activities and analysis results have been cleared."

    def compute_critical_path(self):
        """Compute critical path using standard CPM algorithm (Activity on Arrow)."""
        try:
            if not st.session_state.activities:
                return False, "No activities to analyze. Please add activities first."
            
            graph = st.session_state.graph
            if not nx.is_directed_acyclic_graph(graph):
                try:
                    cycles = list(nx.simple_cycles(graph))
                    if cycles:
                        cycle_path = " -> ".join(str(n) for n in cycles[0]) + f" -> {cycles[0][0]}"
                        return False, f"Network contains cycles (e.g., {cycle_path}). CPM cannot be applied to graphs with cycles."
                    else: # Should not happen if is_directed_acyclic_graph is false
                        return False, "Network contains cycles. CPM cannot be applied."
                except Exception: 
                    return False, "Network contains cycles. CPM cannot be applied."

            nodes = list(graph.nodes())
            if not nodes:
                return False, "No nodes in the network graph."

            try:
                topo_order = list(nx.topological_sort(graph))
            except nx.NetworkXUnfeasible:
                 return False, "Graph is not a DAG (contains cycles), cannot perform topological sort."

            start_nodes = [n for n in topo_order if graph.in_degree(n) == 0]
            end_nodes = [n for n in topo_order if graph.out_degree(n) == 0]

            if not start_nodes:
                return False, "Network must have at least one start node (a node with no predecessors)."
            if not end_nodes:
                return False, "Network must have at least one end node (a node with no successors)."

            ET = {node: 0.0 for node in nodes}
            for node in topo_order:
                current_node_et = 0.0
                for pred in graph.predecessors(node):
                    # Ensure duration is treated as float
                    duration_val = float(graph[pred][node]['duration'])
                    current_node_et = max(current_node_et, ET[pred] + duration_val)
                ET[node] = current_node_et
            
            project_duration = 0.0
            if end_nodes:
                project_duration = max(ET[n] for n in end_nodes)
            elif ET:
                project_duration = max(ET.values())

            LT = {node: project_duration for node in nodes}
            for node in reversed(topo_order):
                if graph.out_degree(node) == 0:
                    LT[node] = project_duration 
                    continue
                current_node_lt = float('inf')
                for succ in graph.successors(node):
                    duration_val = float(graph[node][succ]['duration'])
                    current_node_lt = min(current_node_lt, LT[succ] - duration_val)
                LT[node] = current_node_lt
            
            activity_analysis_list = []
            critical_activity_names = []
            critical_edges_for_diagram = []
            epsilon = 1e-9 

            for u, v, data in graph.edges(data=True):
                activity_id = data['activity']
                duration = float(data['duration'])
                
                act_ES = ET[u]
                act_EF = ET[u] + duration
                act_LF = LT[v]
                act_LS = LT[v] - duration
                total_float = act_LS - act_ES

                if abs(total_float) < epsilon:
                    total_float = 0.0 
                    critical_activity_names.append(activity_id)
                    critical_edges_for_diagram.append((u,v))

                activity_analysis_list.append({
                    'Activity': activity_id, 'Start_Node': u, 'End_Node': v, 'Duration': duration,
                    'ES': round(act_ES, 2), 'EF': round(act_EF, 2), 
                    'LS': round(act_LS, 2), 'LF': round(act_LF, 2), 
                    'Total_Float': round(total_float, 2)
                })
            
            all_critical_nodes_set = set()
            for u_crit, v_crit in critical_edges_for_diagram:
                all_critical_nodes_set.add(u_crit)
                all_critical_nodes_set.add(v_crit)
            
            sorted_critical_nodes = sorted(list(all_critical_nodes_set), key=lambda n: topo_order.index(n) if n in topo_order else float('inf'))

            st.session_state.critical_path = critical_edges_for_diagram
            st.session_state.critical_path_nodes = sorted_critical_nodes
            st.session_state.project_duration = round(project_duration, 2)
            st.session_state.analysis_results = {
                'activities': activity_analysis_list,
                'critical_activities': critical_activity_names,
                'node_times': {'es': {k: round(v,2) for k,v in ET.items()}, 
                               'lf': {k: round(v,2) for k,v in LT.items()}},
                'start_nodes': start_nodes,
                'end_nodes': end_nodes
            }
            return True, f"Critical Path Analysis complete! Project Duration: {st.session_state.project_duration} days."
        except Exception as e:
            # import traceback
            # st.error(f"An unexpected error occurred: {str(e)}\n{traceback.format_exc()}") # For debugging
            st.error(f"An unexpected error occurred during analysis: {str(e)}")
            return False, f"Error in analysis: {str(e)}"

    def calculate_hierarchical_layout(self):
        if not st.session_state.activities or not st.session_state.graph.nodes():
            return {}
        graph = st.session_state.graph
        try:
            pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
            return pos
        except Exception: 
            pass 
        try:
            topo_order = list(nx.topological_sort(graph))
            levels = {}
            node_levels = {}
            for node_idx, node in enumerate(topo_order): # use enumerate for fallback if predecessors are complex
                if not list(graph.predecessors(node)):
                    level = 0
                else:
                    # Ensure all predecessors are in node_levels before calculating max
                    pred_levels = [node_levels[pred] for pred in graph.predecessors(node) if pred in node_levels]
                    if not pred_levels: # If predecessors not yet processed (should not happen with topo_order)
                        level = node_idx # Fallback: use topological index as a rough level
                    else:
                        level = max(pred_levels) + 1

                node_levels[node] = level
                if level not in levels:
                    levels[level] = []
                levels[level].append(node)
            
            pos = {}
            level_width_scale = 4.0 
            node_spacing_scale = 2.0
            max_nodes_in_level = max(len(nodes_in_lvl) for nodes_in_lvl in levels.values()) if levels else 1
            if max_nodes_in_level == 0 : max_nodes_in_level = 1 # Avoid division by zero
            
            for level, nodes_in_level in levels.items():
                x_coord = level * level_width_scale
                num_nodes = len(nodes_in_level)
                current_level_node_spacing = node_spacing_scale * (max_nodes_in_level / num_nodes if num_nodes > 0 else 1)
                if num_nodes == 0: continue

                if num_nodes == 1:
                    pos[nodes_in_level[0]] = (x_coord, 0)
                else:
                    start_y = -(num_nodes - 1) * current_level_node_spacing / 2
                    for i, node_item in enumerate(nodes_in_level):
                        y_coord = start_y + i * current_level_node_spacing
                        pos[node_item] = (x_coord, y_coord)
            return pos
        except Exception: 
            return nx.spring_layout(graph, seed=42, k=2.0/np.sqrt(graph.number_of_nodes()) if graph.number_of_nodes() > 0 else 1.0, iterations=100)

    def draw_network_diagram(self):
        if not st.session_state.activities or not st.session_state.graph.nodes():
            # st.warning("No activities to draw the diagram.") # User sees this in main now
            return None
        
        fig, ax = plt.subplots(figsize=(18, 14))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        graph = st.session_state.graph
        pos = self.calculate_hierarchical_layout()

        if not pos:
            # st.error("Could not calculate node positions for the diagram.") # User sees this in main
            plt.close(fig)
            return None

        regular_node_color = '#e0f7fa' 
        critical_node_color = '#ffebee' 
        node_border_color = '#00796b'  
        critical_border_color = '#c62828' 
        regular_edge_color = '#546e7a' 
        critical_edge_color = '#d32f2f' 

        critical_edges_set = set(st.session_state.critical_path)
        critical_nodes_set = set(st.session_state.critical_path_nodes)

        all_graph_nodes = list(graph.nodes())
        valid_pos = {k: v for k, v in pos.items() if k in all_graph_nodes}

        if not valid_pos and all_graph_nodes: # If all positions are somehow invalid
             valid_pos = nx.spring_layout(graph, seed=42)


        nx.draw_networkx_nodes(graph, valid_pos,
                               nodelist=[n for n in all_graph_nodes if n not in critical_nodes_set and n in valid_pos],
                               node_color=regular_node_color, node_size=3800,
                               edgecolors=node_border_color, linewidths=2.5, ax=ax, alpha=0.95)
        if critical_nodes_set:
            nx.draw_networkx_nodes(graph, valid_pos,
                                   nodelist=[n for n in critical_nodes_set if n in all_graph_nodes and n in valid_pos],
                                   node_color=critical_node_color, node_size=4000,
                                   edgecolors=critical_border_color, linewidths=3.5, ax=ax, alpha=1.0)
            nx.draw_networkx_nodes(graph, valid_pos,
                                   nodelist=[n for n in critical_nodes_set if n in all_graph_nodes and n in valid_pos],
                                   node_color='none', node_size=4200, 
                                   edgecolors=critical_edge_color, linewidths=1, ax=ax, alpha=0.5)

        regular_edges_list = [(u, v) for u, v in graph.edges() if (u,v) not in critical_edges_set]
        critical_edges_list = [(u,v) for u,v in critical_edges_set if graph.has_edge(u,v)]

        nx.draw_networkx_edges(graph, valid_pos, edgelist=regular_edges_list,
                               edge_color=regular_edge_color, width=2.0, arrows=True, arrowsize=25,
                               arrowstyle='-|>', node_size=3800, ax=ax, connectionstyle="arc3,rad=0.05", alpha=0.7)
        if critical_edges_list:
            nx.draw_networkx_edges(graph, valid_pos, edgelist=critical_edges_list,
                                   edge_color=critical_edge_color, width=4.0, style='solid', 
                                   arrows=True, arrowsize=30, arrowstyle='-|>', 
                                   node_size=4000, ax=ax, connectionstyle="arc3,rad=0.05", alpha=1.0)
        
        node_labels_custom = {}
        analysis_node_times = st.session_state.analysis_results.get('node_times', {})
        node_ET = analysis_node_times.get('es', {})
        node_LT = analysis_node_times.get('lf', {})

        for node in all_graph_nodes:
            if node not in valid_pos: continue
            et_val = node_ET.get(node, "N/A")
            lt_val = node_LT.get(node, "N/A")
            # Values are already rounded in compute_critical_path if from analysis
            et_str = f"{et_val}" if isinstance(et_val, (int, float)) else str(et_val)
            lt_str = f"{lt_val}" if isinstance(lt_val, (int, float)) else str(lt_val)
            node_labels_custom[node] = f"{node}\nES:{et_str}\nLF:{lt_str}"
        
        for node, label_text in node_labels_custom.items():
            if node not in valid_pos: continue 
            x, y = valid_pos[node]
            is_critical_node = node in critical_nodes_set
            text_color = critical_border_color if is_critical_node else node_border_color
            font_weight = 'bold' if is_critical_node else 'normal'
            ax.text(x, y, label_text, ha='center', va='center', fontsize=10, fontweight=font_weight, color=text_color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='none', alpha=0.7))

        edge_labels_custom = {}
        for u, v, data in graph.edges(data=True):
            activity_id = data.get('activity', '')
            duration = float(data.get('duration', 0))
            is_critical_edge = (u,v) in critical_edges_set
            star = " ⭐" if is_critical_edge else ""
            edge_labels_custom[(u,v)] = f"{activity_id} ({duration:.1f}d){star}" # Format duration

        for (u,v), label_text in edge_labels_custom.items():
            if u not in valid_pos or v not in valid_pos : continue
            x1, y1 = valid_pos[u]
            x2, y2 = valid_pos[v]
            x_mid, y_mid = (x1 + x2) / 2, (y1 + y2) / 2
            angle = np.arctan2(y2 - y1, x2 - x1) if (x2-x1) != 0 or (y2-y1) !=0 else 0
            offset_dx, offset_dy = -np.sin(angle) * 0.15, np.cos(angle) * 0.15 
            
            is_critical_edge = (u,v) in critical_edges_set
            bbox_fc = '#fff0f0' if is_critical_edge else '#f5f5f5'
            text_col = critical_edge_color if is_critical_edge else regular_edge_color
            font_w = 'bold' if is_critical_edge else 'normal'
            
            ax.text(x_mid + offset_dx, y_mid + offset_dy, label_text, ha='center', va='center', fontsize=9, fontweight=font_w, color=text_col,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=bbox_fc, edgecolor='grey', alpha=0.9, lw=1.5 if is_critical_edge else 0.5))

        ax.set_title("Project Network Diagram & Critical Path", fontsize=22, fontweight='bold', pad=30, color='#333')
        
        if st.session_state.project_duration > 0:
            ax.text(0.5, 0.96, f"Total Project Duration: {st.session_state.project_duration:.1f} days", # Format duration
                    transform=ax.transAxes, ha='center', fontsize=14, color=critical_edge_color, weight='bold')

        ax.axis('off')
        plt.margins(0.1)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Regular Node (ID, ES/LF)',
                   markerfacecolor=regular_node_color, markeredgecolor=node_border_color, markersize=15, mew=2.5),
            Line2D([0], [0], marker='o', color='w', label='Critical Node (ID, ES/LF)',
                   markerfacecolor=critical_node_color, markeredgecolor=critical_border_color, markersize=15, mew=3),
            Line2D([0], [0], color=regular_edge_color, lw=2, label='Regular Activity (Name (Duration))'),
            Line2D([0], [0], color=critical_edge_color, lw=3.5, label='Critical Activity (Name (Duration) ⭐)'),
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=11, frameon=True, fancybox=True, shadow=True, title="Legend", title_fontsize=13)

        if st.session_state.critical_path_nodes:
            critical_nodes_str = ", ".join(map(str, st.session_state.critical_path_nodes))
            if len(critical_nodes_str) > 70: 
                 critical_nodes_str = critical_nodes_str[:67] + "..."
            ax.text(0.98, 0.02, f"Critical Nodes: {critical_nodes_str}",
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='#ffebee', edgecolor=critical_border_color, alpha=0.85),
                    color=critical_border_color, weight='bold')

        plt.tight_layout(pad=1.5)
        return fig

    def add_dummy_data(self):
        self.clear_all_activities() 

        dummy_activities = [
            ("PLAN", "S", "1", 2.0), ("DESIGN_A", "1", "2", 3.0), ("DESIGN_B", "1", "3", 4.0),
            ("CODE_A", "2", "4", 5.0), ("CODE_B", "3", "4", 6.0), ("CODE_C", "3", "5", 4.0),
            ("TEST_AB", "4", "6", 3.0), ("TEST_C", "5", "6", 2.0), ("DEPLOY", "6", "F", 1.0)
        ]
        
        temp_activities_list = st.session_state.activities # Should be empty due to clear_all_activities
        temp_graph_obj = st.session_state.graph # Should be empty

        try:
            for activity_id, start, end, duration in dummy_activities:
                if not all([activity_id, start, end]) or duration <= 0 or start == end:
                    raise ValueError(f"Malformed dummy data for activity '{activity_id}'.")

                st.session_state.graph.add_edge(start, end, duration=float(duration), activity=activity_id)
                st.session_state.activities.append((activity_id, start, end, float(duration)))
            
            st.session_state.critical_path = []
            st.session_state.critical_path_nodes = []
            st.session_state.project_duration = 0
            st.session_state.analysis_results = {}
            return True, "Sample project data loaded successfully. Previous data was cleared."
        except ValueError as ve:
             st.session_state.activities = temp_activities_list # Rollback
             st.session_state.graph = temp_graph_obj # Rollback
             return False, str(ve)
        except Exception as e:
             st.session_state.activities = temp_activities_list # Rollback
             st.session_state.graph = temp_graph_obj # Rollback
             return False, f"An error occurred loading dummy data: {e}"


def main():
    app = NetworkDiagramApp()
    st.markdown("""
    <div class="main-header">
        <h1>📊 SJC Network Diagram Simulator</h1>
        <p>A Tool for Critical Path Method (CPM) Analysis and Visualization</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 📝 Add New Activity")
        with st.form("activity_form", clear_on_submit=True):
            activity_id = st.text_input("Activity ID", placeholder="e.g., A or Task1", help="Unique identifier for the activity.")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                start_node = st.text_input("Start Node", placeholder="e.g., 1 or S", help="Starting node/event.")
            with col_s2:
                end_node = st.text_input("End Node", placeholder="e.g., 2 or P1", help="Ending node/event.")
            duration = st.number_input("Duration (days)", min_value=0.1, value=1.0, step=0.1, format="%.1f", help="Time for activity.")
            submitted_add = st.form_submit_button("➕ Add Activity", type="primary", use_container_width=True)

        if submitted_add:
            success, message = app.add_activity(activity_id, start_node, end_node, float(duration))
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

        st.markdown("---")
        st.markdown("### 🛠️ Project Actions")
        if st.button("🎲 Load Sample Data", use_container_width=True, help="Loads a sample project, clearing current data."):
            success, message = app.add_dummy_data()
            if success: st.success(message)
            else: st.error(message) # Use error for load failure
            st.rerun()

        if st.button("🔍 Analyze Critical Path", use_container_width=True, type="primary", help="Performs CPM analysis."):
            if not st.session_state.activities:
                 st.warning("Please add activities or load sample data before analyzing.")
            else:
                success, message = app.compute_critical_path()
                if success: st.success(message)
                else: st.error(message)
                st.rerun() 
        
        st.markdown("---")
        st.markdown("### ⚙️ Edit Project")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            if st.button("↩️ Undo Last Add", use_container_width=True, help="Removes the last added activity."):
                success, message = app.remove_last_activity()
                if success: st.info(message)
                else: st.warning(message)
                st.rerun()
        with col_e2:
            if st.button("🗑️ Clear Project", use_container_width=True, help="Removes all activities and resets analysis."):
                success, message = app.clear_all_activities()
                st.success(message)
                st.rerun()

    if not st.session_state.activities:
        st.markdown("""
        <div class="info-box">
            <h3>🚀 Getting Started with Your Project Network</h3>
            <p>Welcome! Here's how to begin:</p>
            <ul>
                <li><strong>Add Activities:</strong> Use the sidebar form to define activities, their nodes, and durations.</li>
                <li><strong>Load Sample:</strong> Or, click "Load Sample Data" for an example.</li>
                <li><strong>Analyze:</strong> Click "Analyze Critical Path" for CPM calculations.</li>
                <li><strong>View Results:</strong> The diagram, path details, and analysis table will appear below.</li>
            </ul>
            <p>💡 <strong>Tip:</strong> Node and Activity IDs can be alphanumeric. Ensure Activity IDs are unique.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    analysis_results_exist = st.session_state.analysis_results and 'activities' in st.session_state.analysis_results
    
    if analysis_results_exist:
        st.markdown("### 📈 Project Metrics Overview")
        total_activities_count = len(st.session_state.activities)
        critical_activities_list = st.session_state.analysis_results.get('critical_activities', [])
        critical_activities_count = len(critical_activities_list)

        cols_metrics = st.columns(4)
        cols_metrics[0].metric("Project Duration", f"{st.session_state.project_duration:.1f} days" if st.session_state.project_duration else "N/A")
        cols_metrics[1].metric("Total Activities", total_activities_count)
        cols_metrics[2].metric("Critical Activities", f"{critical_activities_count} / {total_activities_count}")
        
        analysis_df_for_metrics = pd.DataFrame(st.session_state.analysis_results['activities'])
        if not analysis_df_for_metrics.empty and 'Total_Float' in analysis_df_for_metrics.columns:
            avg_total_float = analysis_df_for_metrics['Total_Float'].mean()
            cols_metrics[3].metric("Avg. Total Float", f"{avg_total_float:.1f} days")
        else:
            cols_metrics[3].metric("Avg. Total Float", "N/A")

        if st.session_state.critical_path_nodes or critical_activities_list:
            nodes_str = ", ".join(map(str, st.session_state.critical_path_nodes))
            activities_str = ", ".join(map(str, critical_activities_list))
            st.markdown(f"""
            <div class="critical-path-display">
                🔑 Critical Nodes: {nodes_str if nodes_str else "None"}<br>
                🔑 Critical Activities: {activities_str if activities_str else "None"}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("### 🕸️ Network Diagram")
    if st.session_state.activities:
        fig = app.draw_network_diagram()
        if fig:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig) 
        else:
            st.warning("Diagram could not be generated. Please check activity data or try re-analyzing if you have activities loaded.")
    else:
        st.info("Add activities to generate the network diagram.")


    if analysis_results_exist:
        st.markdown("### 📊 Activity Analysis Details")
        analysis_data = st.session_state.analysis_results['activities']
        df = pd.DataFrame(analysis_data)

        node_ET_for_ff = st.session_state.analysis_results['node_times']['es']
        def calculate_free_float(row):
            start_node_et = node_ET_for_ff.get(row['Start_Node'], 0.0)
            end_node_et = node_ET_for_ff.get(row['End_Node'], 0.0)    
            duration = float(row['Duration'])
            ff = end_node_et - (start_node_et + duration)
            ff_rounded = round(ff, 2)
            return 0.0 if abs(ff_rounded) < 1e-9 else ff_rounded

        if not df.empty:
            df['Free_Float'] = df.apply(calculate_free_float, axis=1)
            display_cols = ['Activity', 'Start_Node', 'End_Node', 'Duration', 'ES', 'EF', 'LS', 'LF', 'Total_Float', 'Free_Float']
            df_display = df[display_cols].copy()
            df_display.columns = ['Activity', 'Start Node', 'End Node', 'Duration', 'Early Start', 'Early Finish', 
                                  'Late Start', 'Late Finish', 'Total Float', 'Free Float']
        else:
            df_display = pd.DataFrame(columns=['Activity', 'Start Node', 'End Node', 'Duration', 'Early Start', 'Early Finish', 'Late Start', 'Late Finish', 'Total Float', 'Free Float'])

        def style_analysis_table(row):
            styles = [''] * len(row)
            total_float_val = row['Total Float']
            if abs(total_float_val) < 1e-9 : 
                styles = ['background-color: #ffebee; font-weight: bold; color: #c62828'] * len(row)
            elif total_float_val <= 1.0: 
                styles = ['background-color: #fff9c4; color: #af8600'] * len(row)
            else: 
                styles = ['background-color: #e8f5e9; color: #2e7d32'] * len(row)
            return styles

        if not df_display.empty:
            styled_df = df_display.style.apply(style_analysis_table, axis=1).format({
                'Duration': '{:.1f}', 'Early Start': '{:.1f}', 'Early Finish': '{:.1f}',
                'Late Start': '{:.1f}', 'Late Finish': '{:.1f}',
                'Total Float': '{:.1f}', 'Free Float': '{:.1f}'
            })
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            st.markdown("#### 📈 Summary Statistics from Analysis")
            cols_summary = st.columns(4)
            critical_acts_summary = df_display[abs(df_display['Total Float']) < 1e-9]['Activity'].tolist()
            cols_summary[0].info(f"**Critical Activities:**\n{', '.join(critical_acts_summary) if critical_acts_summary else 'None'}")
            
            max_tf = df_display['Total Float'].max() if not df_display.empty else 0
            most_flexible = df_display[df_display['Total Float'] == max_tf]['Activity'].tolist() if not df_display.empty else []
            cols_summary[1].info(f"**Most Flexible:**\n{', '.join(most_flexible) if most_flexible else 'N/A'}\n(TF: {max_tf:.1f}d)")
            
            avg_dur = df_display['Duration'].mean() if not df_display.empty else 0
            cols_summary[2].info(f"**Avg. Activity Duration:**\n{avg_dur:.1f} days")
            
            total_work_days = df_display['Duration'].sum() if not df_display.empty else 0
            cols_summary[3].info(f"**Total Work (Durations):**\n{total_work_days:.0f} days")

            st.markdown("#### 📥 Export Results")
            dl_cols = st.columns(2)
            if 'fig' in locals() and fig: 
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
                img_buffer.seek(0)
                dl_cols[0].download_button(label="💾 Download Diagram (PNG)", data=img_buffer,
                                       file_name="network_diagram.png", mime="image/png", use_container_width=True)
            else:
                dl_cols[0].markdown("<small>Diagram not available for download.</small>", unsafe_allow_html=True)

            csv_buffer = io.StringIO()
            df_display.to_csv(csv_buffer, index=False)
            dl_cols[1].download_button(label="📤 Export Analysis Data (CSV)", data=csv_buffer.getvalue(),
                                   file_name="project_analysis_data.csv", mime="text/csv", use_container_width=True)
        # Removed "No analysis data to display" here as it might be covered by analysis_results_exist check

    elif st.session_state.activities and not analysis_results_exist:
        st.markdown("### 📋 Current Activities List")
        st.info("Project activities are loaded. Click 'Analyze Critical Path' to perform CPM analysis.")
        simple_df = pd.DataFrame(st.session_state.activities, columns=['Activity ID', 'Start Node', 'End Node', 'Duration (days)'])
        simple_df['Duration (days)'] = simple_df['Duration (days)'].map(lambda x: f"{x:.1f}")
        st.dataframe(simple_df, use_container_width=True, hide_index=True)
 
    # Footer
st.markdown("""
<div class="footer">
    Web Application |  Developed with 🧡 by J. Inigo Papu Vinodhan, Asst. Prof., BBA Dept., St. Joseph's College, Trichy
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
