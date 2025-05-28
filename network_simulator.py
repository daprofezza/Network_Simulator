Okay, I've reviewed your Streamlit code for the Network Diagram Simulator. I'll provide the corrected and enhanced version, addressing the known issues and incorporating other improvements for robustness and clarity.

Here's a summary of the key changes:

1.  **Corrected CPM Logic (`compute_critical_path`)**:
    * Implemented standard Forward Pass to calculate Earliest Event Times (ET) for each node.
    * Implemented standard Backward Pass to calculate Latest Event Times (LT) for each node.
    * Project duration is correctly determined from the ET of end nodes.
    * Activity times (ES, EF, LS, LF) and Total Float are calculated using the standard AoA (Activity on Arrow) definitions.
    * A small epsilon (1e-9) is used for float comparisons to handle potential floating-point inaccuracies when identifying critical activities.

2.  **Critical Path Identification and Storage**:
    * `st.session_state.critical_path`: Now correctly stores a list of all critical edges `(u,v)` (activities with zero total float). This is used for highlighting edges in the diagram.
    * `st.session_state.critical_path_nodes`: Now stores a list of all unique nodes that are part of *any* critical activity, sorted topologically. This is used for highlighting nodes in the diagram and for display.
    * `st.session_state.analysis_results['critical_activities']`: Stores a list of *names* of all critical activities.

3.  **Diagram Highlighting (`draw_network_diagram`)**:
    * Nodes are highlighted based on whether they belong to `st.session_state.critical_path_nodes` (i.e., are part of any critical activity).
    * Edges are highlighted based on whether they are in `st.session_state.critical_path` (i.e., are critical activities).
    * Node labels in the diagram now display `ES` (Earliest Start, which is ET of the node) and `LF` (Latest Finish, which is LT of the node).
    * The critical path text at the bottom of the diagram is updated to list "Critical Nodes" with comma separation.

4.  **Display of Critical Path Information (`main`)**:
    * The "Critical Path Display" section now lists "Critical Nodes" and "Critical Activities" using comma separation, which is more accurate if multiple critical paths or parallel critical activities exist.

5.  **Free Float Calculation**:
    * The `Free_Float` column in the activity analysis table is now correctly calculated as `ET[j] - (ET[i] + Duration(i,j))`.

6.  **Sample Data Loading (`add_dummy_data`)**:
    * Modified to first clear any existing project data using `self.clear_all_activities()` before loading the sample activities. This ensures a clean state.
    * Crucially, it now resets analysis-related session state variables after loading data directly, similar to `add_activity`.

7.  **Error Handling in Layout**:
    * Broad `except:` clauses in `calculate_hierarchical_layout` are replaced with `except Exception:`.

8.  **CSS and UI Enhancements**:
    * Minor adjustments to ensure consistency and use of the styles.
    * The `critical-pulse` animation is applied to critical items for better visual emphasis (though this would require JavaScript or more complex CSS interactions if done directly on SVG elements, the current CSS applies to HTML elements). The provided CSS will pulse elements with the class `critical-pulse`. The Python code for drawing doesn't directly apply this class to matplotlib elements, but the principle is noted. The current highlighting is done via colors and line widths.

Here is the complete, corrected code:

```python
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import io
import numpy as np
# from matplotlib.patches import FancyBboxPatch # Not explicitly used, can be removed if not needed later

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
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    .critical-pulse {
        animation: pulse 2s ease-in-out infinite;
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
                # Check if the existing edge is the same activity or a different one trying to use the same nodes
                # For simplicity, AoA usually assumes one activity between two nodes.
                # If multiple activities can span the same two nodes, this check might need adjustment or
                # the model needs to support parallel activities explicitly (e.g. by different activity IDs on same edge)
                # Current model: activity name is an attribute of the edge.
                return False, f"An activity already exists between nodes '{start_node}' and '{end_node}'."

            st.session_state.graph.add_edge(start_node, end_node,
                                            duration=duration, activity=activity_id)
            st.session_state.activities.append((activity_id, start_node, end_node, duration))
            
            # Reset analysis results as the graph has changed
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
            
            # Clean up nodes if they become isolated (optional, but good practice)
            nodes_to_remove = [node for node in [start, end] if st.session_state.graph.degree(node) == 0]
            for node in nodes_to_remove:
                if node in st.session_state.graph: # Check if still exists
                    st.session_state.graph.remove_node(node)

            st.session_state.critical_path = []
            st.session_state.critical_path_nodes = []
            st.session_state.project_duration = 0
            st.session_state.analysis_results = {}
            return True, f"Activity '{activity_id}' removed successfully."
        except Exception as e:
            # Re-add if removal failed midway (though pop is usually atomic for the list)
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
                # Attempt to find cycles for a more informative message
                try:
                    cycles = list(nx.simple_cycles(graph))
                    cycle_path = " -> ".join(cycles[0]) + f" -> {cycles[0][0]}"
                    return False, f"Network contains cycles (e.g., {cycle_path}). CPM cannot be applied to graphs with cycles."
                except: # Fallback if cycle finding also fails for some reason
                    return False, "Network contains cycles. CPM cannot be applied to graphs with cycles."

            nodes = list(graph.nodes())
            if not nodes:
                return False, "No nodes in the network graph."

            try:
                topo_order = list(nx.topological_sort(graph))
            except nx.NetworkXUnfeasible: # Should be caught by is_directed_acyclic_graph
                 return False, "Graph is not a DAG (contains cycles), cannot perform topological sort."


            start_nodes = [n for n in topo_order if graph.in_degree(n) == 0]
            end_nodes = [n for n in topo_order if graph.out_degree(n) == 0]

            if not start_nodes:
                return False, "Network must have at least one start node (a node with no predecessors)."
            if not end_nodes:
                return False, "Network must have at least one end node (a node with no successors)."

            # Forward Pass: Calculate Earliest Event Times (ET) for nodes
            ET = {node: 0.0 for node in nodes}
            for node in topo_order:
                current_node_et = 0.0
                for pred in graph.predecessors(node):
                    current_node_et = max(current_node_et, ET[pred] + graph[pred][node]['duration'])
                ET[node] = current_node_et
            
            project_duration = 0.0
            if end_nodes: # Ensure end_nodes is not empty
                project_duration = max(ET[n] for n in end_nodes)
            elif ET: # Fallback if end_nodes somehow empty but ET has values
                project_duration = max(ET.values())


            # Backward Pass: Calculate Latest Event Times (LT) for nodes
            LT = {node: project_duration for node in nodes}
            for node in reversed(topo_order):
                if graph.out_degree(node) == 0: # Is an end node
                    LT[node] = project_duration # Or ET[node] if multiple end points define project duration
                                              # Sticking to one project_duration for consistency
                    continue
                current_node_lt = float('inf')
                for succ in graph.successors(node):
                    current_node_lt = min(current_node_lt, LT[succ] - graph[node][succ]['duration'])
                LT[node] = current_node_lt
            
            # Activity Analysis (ES, EF, LS, LF, Float)
            activity_analysis_list = []
            critical_activity_names = []
            critical_edges_for_diagram = []
            epsilon = 1e-9 # For float comparisons

            for u, v, data in graph.edges(data=True):
                activity_id = data['activity']
                duration = data['duration']
                
                act_ES = ET[u]
                act_EF = ET[u] + duration
                act_LF = LT[v]
                act_LS = LT[v] - duration
                total_float = act_LS - act_ES

                if abs(total_float) < epsilon:
                    total_float = 0.0 # Normalize to zero
                    critical_activity_names.append(activity_id)
                    critical_edges_for_diagram.append((u,v))

                activity_analysis_list.append({
                    'Activity': activity_id, 'Start_Node': u, 'End_Node': v, 'Duration': duration,
                    'ES': round(act_ES, 2), 'EF': round(act_EF, 2), 
                    'LS': round(act_LS, 2), 'LF': round(act_LF, 2), 
                    'Total_Float': round(total_float, 2)
                })
            
            # Identify all unique nodes on any critical path
            all_critical_nodes_set = set()
            for u_crit, v_crit in critical_edges_for_diagram:
                all_critical_nodes_set.add(u_crit)
                all_critical_nodes_set.add(v_crit)
            
            # Sort critical nodes topologically for consistent display
            sorted_critical_nodes = sorted(list(all_critical_nodes_set), key=lambda n: topo_order.index(n) if n in topo_order else float('inf'))

            st.session_state.critical_path = critical_edges_for_diagram
            st.session_state.critical_path_nodes = sorted_critical_nodes
            st.session_state.project_duration = round(project_duration, 2)
            st.session_state.analysis_results = {
                'activities': activity_analysis_list,
                'critical_activities': critical_activity_names, # List of names
                'node_times': {'es': ET, 'lf': LT}, # Store raw ET/LT for nodes
                'start_nodes': start_nodes,
                'end_nodes': end_nodes
            }

            return True, f"Critical Path Analysis complete! Project Duration: {st.session_state.project_duration} days."
        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {str(e)}") # More user-friendly
            # traceback.print_exc() # For server-side logging if needed
            return False, f"Error in analysis: {str(e)}"


    def calculate_hierarchical_layout(self):
        """Calculate layout to minimize edge crossings using hierarchical positioning"""
        if not st.session_state.activities or not st.session_state.graph.nodes():
            return {}
        graph = st.session_state.graph
        try:
            # Requires pygraphviz and graphviz installed
            pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
            return pos
        except Exception: # More specific: ImportError if pygraphviz not found, or other Graphviz errors
            # Fallback to manual topological layout if graphviz fails
            pass 
        try:
            topo_order = list(nx.topological_sort(graph))
            levels = {}
            node_levels = {}
            for node in topo_order:
                if not list(graph.predecessors(node)):
                    level = 0
                else:
                    level = max([node_levels.get(pred, -1) for pred in graph.predecessors(node)]) + 1
                node_levels[node] = level
                if level not in levels:
                    levels[level] = []
                levels[level].append(node)
            
            pos = {}
            level_width_scale = 4.0 
            node_spacing_scale = 2.0
            max_nodes_in_level = max(len(nodes_in_lvl) for nodes_in_lvl in levels.values()) if levels else 1
            
            for level, nodes_in_level in levels.items():
                x_coord = level * level_width_scale
                num_nodes = len(nodes_in_level)
                # Adjust y spacing based on number of nodes to prevent overlap
                current_level_node_spacing = node_spacing_scale * (max_nodes_in_level / num_nodes if num_nodes > 0 else 1)

                if num_nodes == 1:
                    pos[nodes_in_level[0]] = (x_coord, 0)
                else:
                    start_y = -(num_nodes - 1) * current_level_node_spacing / 2
                    for i, node in enumerate(nodes_in_level):
                        y_coord = start_y + i * current_level_node_spacing
                        pos[node] = (x_coord, y_coord)
            return pos
        except Exception: # Fallback if topological sort layout also fails
            return nx.spring_layout(graph, seed=42, k=2.0/np.sqrt(graph.number_of_nodes()) if graph.number_of_nodes() > 0 else 1.0, iterations=100)

    def draw_network_diagram(self):
        """Draw enhanced network diagram with ES/LF values and highlighted critical path"""
        if not st.session_state.activities or not st.session_state.graph.nodes():
            st.warning("No activities to draw the diagram.")
            return None
        
        fig, ax = plt.subplots(figsize=(18, 14)) # Increased size slightly
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        graph = st.session_state.graph
        pos = self.calculate_hierarchical_layout()

        if not pos: # If layout calculation failed or returned empty
            st.error("Could not calculate node positions for the diagram.")
            plt.close(fig)
            return None

        regular_node_color = '#e0f7fa' # Lighter blue
        critical_node_color = '#ffebee' # Lighter red
        node_border_color = '#00796b'  # Teal
        critical_border_color = '#c62828' # Darker Red
        regular_edge_color = '#546e7a' # Blue-grey
        critical_edge_color = '#d32f2f' # Material Red
        # critical_path_highlight_pulse = '#ff1744' # Used for CSS, not directly in matplotib

        # Use st.session_state.critical_path for edges and st.session_state.critical_path_nodes for nodes
        critical_edges_set = set(st.session_state.critical_path) # These are (u,v) tuples
        critical_nodes_set = set(st.session_state.critical_path_nodes) # These are node names

        all_graph_nodes = list(graph.nodes()) # Ensure we only try to draw nodes present in the graph
        
        # Filter pos to only include nodes in the graph
        valid_pos = {k: v for k, v in pos.items() if k in all_graph_nodes}
        if len(valid_pos) != len(all_graph_nodes):
            # This case should ideally not happen if pos is generated from the graph
            missing_nodes = [n for n in all_graph_nodes if n not in valid_pos]
            # st.warning(f"Position missing for nodes: {missing_nodes}. Using spring layout for them.")
            # temp_pos = nx.spring_layout(graph.subgraph(missing_nodes), seed=42)
            # valid_pos.update(temp_pos)
            # Fallback to overall spring if too many issues:
            if not valid_pos and all_graph_nodes: # If all positions are somehow invalid
                 valid_pos = nx.spring_layout(graph, seed=42)

        # Draw regular nodes
        nx.draw_networkx_nodes(graph, valid_pos,
                               nodelist=[n for n in all_graph_nodes if n not in critical_nodes_set],
                               node_color=regular_node_color, node_size=3800,
                               edgecolors=node_border_color, linewidths=2.5, ax=ax, alpha=0.95)
        # Draw critical nodes
        if critical_nodes_set:
            nx.draw_networkx_nodes(graph, valid_pos,
                                   nodelist=[n for n in critical_nodes_set if n in all_graph_nodes], # Ensure node exists
                                   node_color=critical_node_color, node_size=4000,
                                   edgecolors=critical_border_color, linewidths=3.5, ax=ax, alpha=1.0)
            # Optional: Add a subtle glow or outer ring for critical nodes (can be faked with another draw call)
            nx.draw_networkx_nodes(graph, valid_pos,
                                   nodelist=[n for n in critical_nodes_set if n in all_graph_nodes],
                                   node_color='none', node_size=4200, # Slightly larger size
                                   edgecolors=critical_edge_color, linewidths=1, ax=ax, alpha=0.5)


        # Draw edges
        regular_edges_list = [(u, v) for u, v in graph.edges() if (u,v) not in critical_edges_set]
        critical_edges_list = [(u,v) for u,v in critical_edges_set if graph.has_edge(u,v)] # Ensure edge exists

        nx.draw_networkx_edges(graph, valid_pos, edgelist=regular_edges_list,
                               edge_color=regular_edge_color, width=2.0, arrows=True, arrowsize=25,
                               arrowstyle='-|>', node_size=3800, ax=ax, connectionstyle="arc3,rad=0.05", alpha=0.7)
        if critical_edges_list:
            nx.draw_networkx_edges(graph, valid_pos, edgelist=critical_edges_list,
                                   edge_color=critical_edge_color, width=4.0, style='solid', 
                                   arrows=True, arrowsize=30, arrowstyle='-|>', 
                                   node_size=4000, ax=ax, connectionstyle="arc3,rad=0.05", alpha=1.0)
        
        # Node labels (ID, ES, LF)
        node_labels_custom = {}
        if st.session_state.analysis_results and 'node_times' in st.session_state.analysis_results:
            node_ET = st.session_state.analysis_results['node_times']['es']
            node_LT = st.session_state.analysis_results['node_times']['lf']
            for node in all_graph_nodes:
                et_val = node_ET.get(node, "N/A")
                lt_val = node_LT.get(node, "N/A")
                et_str = f"{et_val:.1f}" if isinstance(et_val, (int, float)) else str(et_val)
                lt_str = f"{lt_val:.1f}" if isinstance(lt_val, (int, float)) else str(lt_val)
                node_labels_custom[node] = f"{node}\nES: {et_str}\nLF: {lt_str}"
        else: # Fallback if no analysis yet
            node_labels_custom = {node: str(node) for node in all_graph_nodes}

        for node, label_text in node_labels_custom.items():
            if node not in valid_pos: continue # Skip if position is not determined
            x, y = valid_pos[node]
            is_critical_node = node in critical_nodes_set
            text_color = critical_border_color if is_critical_node else node_border_color
            font_weight = 'bold' if is_critical_node else 'normal'
            ax.text(x, y, label_text, ha='center', va='center', fontsize=10, fontweight=font_weight, color=text_color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='none', alpha=0.7))

        # Edge labels (Activity ID, Duration)
        edge_labels_custom = {}
        for u, v, data in graph.edges(data=True):
            activity_id = data.get('activity', '')
            duration = data.get('duration', 0)
            is_critical_edge = (u,v) in critical_edges_set
            star = " ⭐" if is_critical_edge else ""
            edge_labels_custom[(u,v)] = f"{activity_id} ({duration}d){star}"

        for (u,v), label_text in edge_labels_custom.items():
            if u not in valid_pos or v not in valid_pos : continue
            x1, y1 = valid_pos[u]
            x2, y2 = valid_pos[v]
            x_mid, y_mid = (x1 + x2) / 2, (y1 + y2) / 2
            # Offset label slightly to avoid overlap with thick edges
            angle = np.arctan2(y2 - y1, x2 - x1)
            offset_dx, offset_dy = -np.sin(angle) * 0.15, np.cos(angle) * 0.15 # Perpendicular offset
            
            is_critical_edge = (u,v) in critical_edges_set
            bbox_fc = '#fff0f0' if is_critical_edge else '#f5f5f5'
            text_col = critical_edge_color if is_critical_edge else regular_edge_color
            font_w = 'bold' if is_critical_edge else 'normal'
            
            ax.text(x_mid + offset_dx, y_mid + offset_dy, label_text, ha='center', va='center', fontsize=9, fontweight=font_w, color=text_col,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=bbox_fc, edgecolor='grey', alpha=0.9, lw=1.5 if is_critical_edge else 0.5))


        ax.set_title("Project Network Diagram & Critical Path", fontsize=22, fontweight='bold', pad=30, color='#333')
        
        if st.session_state.project_duration > 0:
            ax.text(0.5, 0.96, f"Total Project Duration: {st.session_state.project_duration} days",
                    transform=ax.transAxes, ha='center', fontsize=14, color=critical_edge_color, weight='bold')

        ax.axis('off')
        plt.margins(0.1) # Add some margin around the graph elements

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Regular Node (ES/LF)',
                   markerfacecolor=regular_node_color, markeredgecolor=node_border_color, markersize=15, mew=2.5),
            Line2D([0], [0], marker='o', color='w', label='Critical Node (ES/LF)',
                   markerfacecolor=critical_node_color, markeredgecolor=critical_border_color, markersize=15, mew=3),
            Line2D([0], [0], color=regular_edge_color, lw=2, label=f'Regular Activity (Name (Duration))'),
            Line2D([0], [0], color=critical_edge_color, lw=3.5, label=f'Critical Activity (Name (Duration) ⭐)'),
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=11, frameon=True, fancybox=True, shadow=True, title="Legend", title_fontsize=13)

        if st.session_state.critical_path_nodes:
            # Display critical nodes list at the bottom right or another suitable place
            critical_nodes_str = ", ".join(map(str, st.session_state.critical_path_nodes))
            if len(critical_nodes_str) > 80: # Truncate if too long for diagram display
                 critical_nodes_str = critical_nodes_str[:77] + "..."
            ax.text(0.98, 0.02, f"Critical Nodes: {critical_nodes_str}",
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='#ffebee', edgecolor=critical_border_color, alpha=0.85),
                    color=critical_border_color, weight='bold')

        plt.tight_layout(pad=1.5)
        return fig

    def add_dummy_data(self):
        """Add sample project data, clearing any existing data first."""
        self.clear_all_activities() # Ensure a clean slate

        dummy_activities = [
            ("PLAN", "S", "1", 2),
            ("DESIGN_A", "1", "2", 3),
            ("DESIGN_B", "1", "3", 4),
            ("CODE_A", "2", "4", 5),
            ("CODE_B", "3", "4", 6),
            ("CODE_C", "3", "5", 4),
            ("TEST_AB", "4", "6", 3),
            ("TEST_C", "5", "6", 2),
            ("DEPLOY", "6", "F", 1)
        ]
        
        # Use the internal add_activity logic but without resetting analysis results on each add for this bulk load
        temp_activities = st.session_state.activities
        temp_graph = st.session_state.graph
        st.session_state.activities = [] # Temp clear for direct append
        st.session_state.graph = nx.DiGraph()

        for activity_id, start, end, duration in dummy_activities:
            # Basic validation similar to add_activity, but simplified for dummy data
            if not all([activity_id, start, end]) or duration <= 0 or start == end:
                # Rollback if dummy data is malformed
                st.session_state.activities = temp_activities
                st.session_state.graph = temp_graph
                return False, f"Malformed dummy data for activity '{activity_id}'. Load aborted."

            st.session_state.graph.add_edge(start, end, duration=duration, activity=activity_id)
            st.session_state.activities.append((activity_id, start, end, duration))
        
        # Reset analysis states after modifying activities/graph directly
        st.session_state.critical_path = []
        st.session_state.critical_path_nodes = []
        st.session_state.project_duration = 0
        st.session_state.analysis_results = {}
        return True, "Sample project data loaded successfully. Previous data was cleared."


def main():
    app = NetworkDiagramApp()
    st.markdown("""
    <div class="main-header">
        <h1>📊 Network Diagram Simulator</h1>
        <p>A Tool for Critical Path Method (CPM) Analysis and Visualization</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 📝 Add New Activity")
        with st.form("activity_form", clear_on_submit=True):
            activity_id = st.text_input("Activity ID", placeholder="e.g., A or Task1", help="Unique identifier for the activity (e.g., A, B, DESIGN, etc.)")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                start_node = st.text_input("Start Node", placeholder="e.g., 1 or S", help="Identifier for the activity's starting node/event.")
            with col_s2:
                end_node = st.text_input("End Node", placeholder="e.g., 2 or P1", help="Identifier for the activity's ending node/event.")
            duration = st.number_input("Duration (days)", min_value=0.1, value=1.0, step=0.1, format="%.1f", help="Time required to complete the activity, in days.")
            submitted_add = st.form_submit_button("➕ Add Activity", type="primary", use_container_width=True)

        if submitted_add:
            success, message = app.add_activity(activity_id, start_node, end_node, duration)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

        st.markdown("---")
        st.markdown("### 🛠️ Project Actions")
        if st.button("🎲 Load Sample Data", use_container_width=True, help="Loads a predefined sample project. Clears any current data."):
            success, message = app.add_dummy_data()
            if success:
                st.success(message)
                st.rerun()
            else:
                st.warning(message) # Or st.error if it's a critical failure

        if st.button("🔍 Analyze Critical Path", use_container_width=True, type="primary", help="Performs CPM analysis on the current set of activities."):
            if not st.session_state.activities:
                 st.warning("Please add activities or load sample data before analyzing.")
            else:
                success, message = app.compute_critical_path()
                if success:
                    st.success(message)
                    st.rerun() 
                else:
                    st.error(message)
        
        st.markdown("---")
        st.markdown("### ⚙️ Edit Project")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            if st.button("↩️ Undo Last Add", use_container_width=True, help="Removes the most recently added activity."):
                success, message = app.remove_last_activity()
                if success:
                    st.info(message) # Use info for undo
                    st.rerun()
                else:
                    st.warning(message)
        with col_e2:
            if st.button("🗑️ Clear Project", use_container_width=True, help="Removes all activities and resets the analysis."):
                alert_placeholder = st.empty()
                # Confirmation can be added here if desired, e.g., using a modal or a checkbox
                success, message = app.clear_all_activities()
                st.success(message)
                st.rerun()

    if not st.session_state.activities:
        st.markdown("""
        <div class="info-box">
            <h3>🚀 Getting Started with Your Project Network</h3>
            <p>Welcome to the Network Diagram Simulator! Here's how to begin:</p>
            <ul>
                <li><strong>Add Activities:</strong> Use the sidebar form to define your project activities, their start/end nodes, and durations.</li>
                <li><strong>Load Sample:</strong> Alternatively, click "Load Sample Data" in the sidebar to see a pre-filled example.</li>
                <li><strong>Analyze:</strong> Once you have some activities, click "Analyze Critical Path" to perform the CPM calculations.</li>
                <li><strong>View Results:</strong> The network diagram, critical path details, and activity analysis table will be displayed below.</li>
            </ul>
            <p>💡 <strong>Tip:</strong> Node and Activity IDs can be alphanumeric (e.g., "A1", "TaskStart", "10"). Ensure Activity IDs are unique.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Main content area: Metrics, Diagram, Analysis Table
    if st.session_state.analysis_results and 'activities' in st.session_state.analysis_results:
        st.markdown("### 📈 Project Metrics Overview")
        total_activities_count = len(st.session_state.activities)
        critical_activities_list = st.session_state.analysis_results.get('critical_activities', [])
        critical_activities_count = len(critical_activities_list)

        cols_metrics = st.columns(4)
        with cols_metrics[0]:
            st.metric("Project Duration", f"{st.session_state.project_duration} days" if st.session_state.project_duration else "N/A")
        with cols_metrics[1]:
            st.metric("Total Activities", total_activities_count)
        with cols_metrics[2]:
            st.metric("Critical Activities", f"{critical_activities_count} / {total_activities_count}")
        
        analysis_df_for_metrics = pd.DataFrame(st.session_state.analysis_results['activities'])
        if not analysis_df_for_metrics.empty and 'Total_Float' in analysis_df_for_metrics.columns:
            avg_total_float = analysis_df_for_metrics['Total_Float'].mean()
            with cols_metrics[3]:
                 st.metric("Avg. Total Float", f"{avg_total_float:.1f} days")
        else:
            with cols_metrics[3]:
                 st.metric("Avg. Total Float", "N/A")


        if st.session_state.critical_path_nodes or critical_activities_list:
            nodes_str = ", ".join(map(str, st.session_state.critical_path_nodes))
            activities_str = ", ".join(map(str, critical_activities_list))
            st.markdown(f"""
            <div class="critical-path-display">
                🔑 Critical Nodes: {nodes_str if nodes_str else "None identified"}<br>
                🔑 Critical Activities: {activities_str if activities_str else "None identified"}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("### 🕸️ Network Diagram")
    fig = app.draw_network_diagram()
    if fig:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig) # Important to close the figure to free memory
    elif st.session_state.activities : # If activities exist but diagram fails
        st.warning("Diagram could not be generated. Please check activity data or try re-analyzing.")


    if st.session_state.analysis_results and 'activities' in st.session_state.analysis_results:
        st.markdown("### 📊 Activity Analysis Details")
        analysis_data = st.session_state.analysis_results['activities']
        df = pd.DataFrame(analysis_data)

        # Calculate Free Float
        node_ET = st.session_state.analysis_results['node_times']['es']
        def calculate_free_float(row):
            start_node_et = node_ET.get(row['Start_Node'], 0.0) # ET of activity's start node
            end_node_et = node_ET.get(row['End_Node'], 0.0)     # ET of activity's end node
            duration = row['Duration']
            # Free Float = ET(j) - (ET(i) + Duration(i,j))
            ff = end_node_et - (start_node_et + duration)
            return round(ff, 2) if abs(ff) > 1e-9 else 0.0


        if not df.empty:
            df['Free_Float'] = df.apply(calculate_free_float, axis=1)
            display_cols = ['Activity', 'Start_Node', 'End_Node', 'Duration', 'ES', 'EF', 'LS', 'LF', 'Total_Float', 'Free_Float']
            df_display = df[display_cols].copy()
            df_display.columns = ['Activity', 'Start Node', 'End Node', 'Duration', 
                                  'Early Start (ES)', 'Early Finish (EF)', 
                                  'Late Start (LS)', 'Late Finish (LF)', 
                                  'Total Float', 'Free Float']
        else:
            df_display = pd.DataFrame(columns=['Activity', 'Start Node', 'End Node', 'Duration', 'ES', 'EF', 'LS', 'LF', 'Total Float', 'Free Float'])


        def style_analysis_table(row):
            styles = [''] * len(row)
            total_float_val = row['Total Float']
            if abs(total_float_val) < 1e-9 : # Critical
                styles = ['background-color: #ffebee; font-weight: bold; color: #c62828'] * len(row)
            elif total_float_val <= 1.0: # Near-critical (example threshold)
                styles = ['background-color: #fff9c4; color: #af8600'] * len(row) # Light yellow
            else: # Non-critical
                styles = ['background-color: #e8f5e9; color: #2e7d32'] * len(row) # Light green
            return styles

        if not df_display.empty:
            styled_df = df_display.style.apply(style_analysis_table, axis=1).format({
                'Duration': '{:.1f}', 'Early Start (ES)': '{:.1f}', 'Early Finish (EF)': '{:.1f}',
                'Late Start (LS)': '{:.1f}', 'Late Finish (LF)': '{:.1f}',
                'Total Float': '{:.1f}', 'Free Float': '{:.1f}'
            })
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            st.markdown("#### 📈 Summary Statistics from Analysis")
            cols_summary = st.columns(4)
            with cols_summary[0]:
                critical_acts_summary = df_display[df_display['Total Float'] == 0.0]['Activity'].tolist()
                st.info(f"**Critical Activities:**\n{', '.join(critical_acts_summary) if critical_acts_summary else 'None'}")
            with cols_summary[1]:
                if not df_display.empty:
                    max_tf = df_display['Total Float'].max()
                    most_flexible = df_display[df_display['Total Float'] == max_tf]['Activity'].tolist()
                    st.info(f"**Most Flexible:**\n{', '.join(most_flexible) if most_flexible else 'N/A'}\n(Total Float: {max_tf:.1f}d)")
                else: st.info("**Most Flexible:** N/A")
            with cols_summary[2]:
                avg_dur = df_display['Duration'].mean() if not df_display.empty else 0
                st.info(f"**Avg. Activity Duration:**\n{avg_dur:.1f} days")
            with cols_summary[3]:
                total_work_days = df_display['Duration'].sum() if not df_display.empty else 0
                st.info(f"**Total Work (Sum of Durations):**\n{total_work_days:.0f} days")

            # Download Buttons
            st.markdown("#### 📥 Export Results")
            dl_cols = st.columns(2)
            with dl_cols[0]:
                if fig: # Check if figure was generated
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
                    img_buffer.seek(0)
                    st.download_button(label="💾 Download Diagram (PNG)", data=img_buffer,
                                       file_name="network_diagram.png", mime="image/png", use_container_width=True)
            with dl_cols[1]:
                csv_buffer = io.StringIO()
                df_display.to_csv(csv_buffer, index=False)
                st.download_button(label="📤 Export Analysis Data (CSV)", data=csv_buffer.getvalue(),
                                   file_name="project_analysis_data.csv", mime="text/csv", use_container_width=True)
        else:
            st.markdown("<p class='info-box'>No analysis data to display. Please run the analysis.</p>", unsafe_allow_html=True)


    elif st.session_state.activities: # If activities exist but no analysis results yet
        st.markdown("### 📋 Current Activities List")
        st.info("Project activities are loaded. Click 'Analyze Critical Path' in the sidebar to perform CPM analysis.")
        simple_df = pd.DataFrame(st.session_state.activities, columns=['Activity ID', 'Start Node', 'End Node', 'Duration (days)'])
        st.dataframe(simple_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 2rem 1rem; font-size: 0.9em;'>"
        "Network Diagram Simulator | Web Application <br>"
        "Developed by J. Inigo Papu Vinodhan, Asst. Prof., BBA Dept., St. Joseph's College, Trichy"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
```
