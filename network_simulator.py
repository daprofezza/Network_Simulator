import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Advanced CPM Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class Activity:
    """Activity data structure"""
    id: str
    start_node: str
    end_node: str
    duration: float
    es: float = 0.0  # Early Start
    ef: float = 0.0  # Early Finish
    ls: float = 0.0  # Late Start
    lf: float = 0.0  # Late Finish
    total_float: float = 0.0
    free_float: float = 0.0
    is_critical: bool = False

class CPMAnalyzer:
    """Optimized CPM Analysis Engine"""
    
    def __init__(self):
        self.activities: Dict[str, Activity] = {}
        self.graph = nx.DiGraph()
        self.node_times: Dict[str, Dict[str, float]] = {}
        self.critical_path: List[str] = []
        self.project_duration: float = 0.0
        
    def add_activity(self, activity_id: str, start: str, end: str, duration: float) -> Tuple[bool, str]:
        """Add activity with comprehensive validation"""
        # Input validation
        if not all([activity_id.strip(), start.strip(), end.strip()]):
            return False, "All fields required"
        
        if duration <= 0:
            return False, "Duration must be positive"
            
        if start == end:
            return False, "Start and end nodes must differ"
            
        activity_id = activity_id.strip().upper()
        start, end = start.strip(), end.strip()
        
        if activity_id in self.activities:
            return False, f"Activity {activity_id} already exists"
            
        # Cycle detection
        temp_graph = self.graph.copy()
        temp_graph.add_edge(start, end)
        if not nx.is_directed_acyclic_graph(temp_graph):
            return False, "Would create cycle"
            
        # Add activity
        activity = Activity(activity_id, start, end, duration)
        self.activities[activity_id] = activity
        self.graph.add_edge(start, end, activity=activity_id, duration=duration)
        
        return True, f"Activity {activity_id} added"
    
    def analyze_network(self) -> Tuple[bool, str]:
        """Complete CPM analysis with all calculations"""
        if not self.activities:
            return False, "No activities to analyze"
            
        try:
            # Reset previous analysis
            self._reset_analysis()
            
            # Get network topology
            nodes = list(self.graph.nodes())
            start_nodes = [n for n in nodes if self.graph.in_degree(n) == 0]
            end_nodes = [n for n in nodes if self.graph.out_degree(n) == 0]
            
            if not start_nodes or not end_nodes:
                return False, "Invalid network topology"
            
            # Forward pass - Early times
            self._forward_pass(start_nodes)
            
            # Calculate project duration
            self.project_duration = max(self.node_times[node]['ef'] for node in end_nodes)
            
            # Backward pass - Late times  
            self._backward_pass(end_nodes)
            
            # Calculate activity times and floats
            self._calculate_activity_times()
            
            # Find critical path
            self._find_critical_path(start_nodes, end_nodes)
            
            return True, f"Analysis complete. Duration: {self.project_duration:.1f}"
            
        except Exception as e:
            return False, f"Analysis error: {str(e)}"
    
    def _reset_analysis(self):
        """Reset all calculated values"""
        self.node_times = {node: {'es': 0, 'ef': 0, 'ls': 0, 'lf': 0} 
                          for node in self.graph.nodes()}
        for activity in self.activities.values():
            activity.es = activity.ef = activity.ls = activity.lf = 0
            activity.total_float = activity.free_float = 0
            activity.is_critical = False
    
    def _forward_pass(self, start_nodes: List[str]):
        """Calculate early start and finish times"""
        # Initialize start nodes
        for node in start_nodes:
            self.node_times[node]['es'] = 0
            self.node_times[node]['ef'] = 0
            
        # Topological sort for correct processing order
        for node in nx.topological_sort(self.graph):
            node_es = self.node_times[node]['es']
            
            # Process all outgoing activities
            for successor in self.graph.successors(node):
                duration = self.graph[node][successor]['duration']
                successor_es = node_es + duration
                
                # Update successor if this path is longer
                if successor_es > self.node_times[successor]['es']:
                    self.node_times[successor]['es'] = successor_es
                    
            # Node EF = Node ES (for event nodes)
            self.node_times[node]['ef'] = self.node_times[node]['es']
    
    def _backward_pass(self, end_nodes: List[str]):
        """Calculate late start and finish times"""
        # Initialize end nodes
        for node in end_nodes:
            self.node_times[node]['lf'] = self.node_times[node]['ef']
            self.node_times[node]['ls'] = self.node_times[node]['es']
            
        # Reverse topological order
        for node in reversed(list(nx.topological_sort(self.graph))):
            if self.graph.out_degree(node) == 0:
                continue
                
            # Find minimum late start of successors
            min_ls = float('inf')
            for successor in self.graph.successors(node):
                duration = self.graph[node][successor]['duration']
                successor_ls = self.node_times[successor]['ls'] - duration
                min_ls = min(min_ls, successor_ls)
                
            self.node_times[node]['ls'] = min_ls
            self.node_times[node]['lf'] = min_ls
    
    def _calculate_activity_times(self):
        """Calculate all activity times and floats"""
        for activity in self.activities.values():
            start_node = activity.start_node
            end_node = activity.end_node
            
            # Activity times
            activity.es = self.node_times[start_node]['es']
            activity.ef = activity.es + activity.duration
            activity.lf = self.node_times[end_node]['lf']
            activity.ls = activity.lf - activity.duration
            
            # Floats
            activity.total_float = activity.ls - activity.es
            
            # Free float = EF of activity - ES of end node
            activity.free_float = self.node_times[end_node]['es'] - activity.ef
            
            # Critical activity check
            activity.is_critical = abs(activity.total_float) < 0.001
    
    def _find_critical_path(self, start_nodes: List[str], end_nodes: List[str]):
        """Find the critical path through the network"""
        # Build critical edges
        critical_edges = []
        for activity in self.activities.values():
            if activity.is_critical:
                critical_edges.append((activity.start_node, activity.end_node))
        
        # Find longest critical path
        max_length = 0
        best_path = []
        
        for start in start_nodes:
            for end in end_nodes:
                path = self._find_path_through_critical_edges(start, end, critical_edges)
                if path and len(path) > max_length:
                    max_length = len(path)
                    best_path = path
                    
        self.critical_path = best_path
    
    def _find_path_through_critical_edges(self, start: str, end: str, 
                                        critical_edges: List[Tuple[str, str]]) -> List[str]:
        """Find path using only critical edges"""
        def dfs(current: str, target: str, path: List[str], visited: set) -> Optional[List[str]]:
            if current == target:
                return path
                
            for edge_start, edge_end in critical_edges:
                if edge_start == current and edge_end not in visited:
                    visited.add(edge_end)
                    result = dfs(edge_end, target, path + [edge_end], visited)
                    if result:
                        return result
                    visited.remove(edge_end)
            return None
        
        return dfs(start, end, [start], {start})
    
    def get_critical_activities(self) -> List[Activity]:
        """Get list of critical activities in path order"""
        if len(self.critical_path) < 2:
            return [a for a in self.activities.values() if a.is_critical]
            
        critical_activities = []
        for i in range(len(self.critical_path) - 1):
            start, end = self.critical_path[i], self.critical_path[i + 1]
            for activity in self.activities.values():
                if activity.start_node == start and activity.end_node == end:
                    critical_activities.append(activity)
                    break
        return critical_activities
    
    def get_analysis_dataframe(self) -> pd.DataFrame:
        """Generate comprehensive analysis dataframe"""
        data = []
        for activity in self.activities.values():
            data.append({
                'Activity': activity.id,
                'Start': activity.start_node,
                'End': activity.end_node,
                'Duration': activity.duration,
                'ES': activity.es,
                'EF': activity.ef,
                'LS': activity.ls,
                'LF': activity.lf,
                'Total Float': round(activity.total_float, 2),
                'Free Float': round(activity.free_float, 2),
                'Critical': '★' if activity.is_critical else ''
            })
        
        return pd.DataFrame(data).sort_values('ES')

class NetworkVisualizer:
    """Advanced network visualization using Plotly"""
    
    @staticmethod
    def create_interactive_network(analyzer: CPMAnalyzer) -> go.Figure:
        """Create interactive network diagram"""
        if not analyzer.activities:
            return go.Figure()
            
        # Calculate layout
        pos = nx.spring_layout(analyzer.graph, k=3, iterations=50, seed=42)
        
        # Prepare node data
        node_x, node_y, node_text, node_colors, node_sizes = [], [], [], [], []
        critical_nodes = set(analyzer.critical_path)
        
        for node in analyzer.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            es = analyzer.node_times.get(node, {}).get('es', 0)
            ls = analyzer.node_times.get(node, {}).get('ls', 0)
            
            node_text.append(f"Node {node}<br>ES: {es:.1f}<br>LS: {ls:.1f}")
            
            if node in critical_nodes:
                node_colors.append('#ff4444')
                node_sizes.append(25)
            else:
                node_colors.append('#4CAF50')
                node_sizes.append(20)
        
        # Prepare edge data
        edge_x, edge_y, edge_text = [], [], []
        
        for activity in analyzer.activities.values():
            start_pos = pos[activity.start_node]
            end_pos = pos[activity.end_node]
            
            edge_x.extend([start_pos[0], end_pos[0], None])
            edge_y.extend([start_pos[1], end_pos[1], None])
            
            # Edge annotation
            mid_x = (start_pos[0] + end_pos[0]) / 2
            mid_y = (start_pos[1] + end_pos[1]) / 2
            
            color = '#ff4444' if activity.is_critical else '#666666'
            symbol = '★' if activity.is_critical else ''
            
            edge_text.append({
                'x': mid_x, 'y': mid_y,
                'text': f"{activity.id}({activity.duration}){symbol}",
                'color': color
            })
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=2, color='#666666'),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color='white')),
            text=[node for node in analyzer.graph.nodes()],
            textposition="middle center",
            hovertext=node_text,
            hoverinfo='text',
            showlegend=False
        ))
        
        # Add edge labels
        for edge_info in edge_text:
            fig.add_annotation(
                x=edge_info['x'], y=edge_info['y'],
                text=edge_info['text'],
                showarrow=False,
                font=dict(color=edge_info['color'], size=10, family="Arial Black"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=edge_info['color'],
                borderwidth=1
            )
        
        fig.update_layout(
            title=dict(
                text=f"Critical Path Network (Duration: {analyzer.project_duration:.1f} days)",
                x=0.5, font=dict(size=18)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(text="Red = Critical Path | Green = Non-Critical", 
                     showarrow=False, xref="paper", yref="paper",
                     x=0.005, y=-0.002, xanchor='left', yanchor='bottom',
                     font=dict(size=12, color="#666666"))
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return fig

# Streamlit App
def main():
    # Custom CSS
    st.markdown("""
    <style>
    .metric-container { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; border-radius: 10px; color: white; text-align: center; margin: 0.5rem 0;
    }
    .critical-path {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white; padding: 1.5rem; border-radius: 10px; text-align: center; margin: 1rem 0;
    }
    .stDataFrame [data-testid="stTable"] { background-color: #f8f9fa; }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = CPMAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Header
    st.title("🚀 Advanced CPM Network Analyzer")
    st.markdown("*Professional Critical Path Method Analysis Tool*")
    
    # Sidebar - Input Panel
    with st.sidebar:
        st.header("📝 Project Management")
        
        # Activity input form
        with st.form("add_activity", clear_on_submit=True):
            st.subheader("Add Activity")
            activity_id = st.text_input("Activity ID", placeholder="A")
            col1, col2 = st.columns(2)
            with col1:
                start_node = st.text_input("From", placeholder="1")
            with col2:
                end_node = st.text_input("To", placeholder="2")
            duration = st.number_input("Duration", min_value=0.1, value=1.0, step=0.1)
            
            if st.form_submit_button("➕ Add Activity", type="primary"):
                success, message = analyzer.add_activity(activity_id, start_node, end_node, duration)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        
        st.divider()
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎲 Sample Data"):
                # Clear and add sample
                st.session_state.analyzer = CPMAnalyzer()
                analyzer = st.session_state.analyzer
                
                sample_activities = [
                    ("A", "1", "2", 3), ("B", "1", "3", 4), ("C", "2", "4", 2),
                    ("D", "3", "4", 5), ("E", "4", "5", 3), ("F", "2", "5", 6)
                ]
                
                for act_id, start, end, dur in sample_activities:
                    analyzer.add_activity(act_id, start, end, dur)
                
                st.success("Sample project loaded!")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear All"):
                st.session_state.analyzer = CPMAnalyzer()
                st.success("All data cleared!")
                st.rerun()
        
        if st.button("🔍 Analyze Network", type="primary", use_container_width=True):
            success, message = analyzer.analyze_network()
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
    
    # Main content
    if not analyzer.activities:
        st.info("👆 Add activities using the sidebar to get started, or load sample data!")
        return
    
    # Display current activities
    st.subheader("📋 Project Activities")
    activities_df = pd.DataFrame([
        {
            'ID': a.id, 'From': a.start_node, 'To': a.end_node, 
            'Duration': a.duration, 'Critical': '★' if a.is_critical else ''
        }
        for a in analyzer.activities.values()
    ])
    st.dataframe(activities_df, use_container_width=True, hide_index=True)
    
    # Analysis results
    if analyzer.project_duration > 0:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Project Duration", f"{analyzer.project_duration:.1f} days")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            critical_count = sum(1 for a in analyzer.activities.values() if a.is_critical)
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Critical Activities", critical_count)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Total Activities", len(analyzer.activities))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            avg_float = np.mean([a.total_float for a in analyzer.activities.values() if not a.is_critical])
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Avg Float", f"{avg_float:.1f}" if not np.isnan(avg_float) else "0")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Critical path display
        if analyzer.critical_path:
            path_str = " → ".join(analyzer.critical_path)
            critical_activities = analyzer.get_critical_activities()
            activities_str = " → ".join([f"{a.id}({a.duration})" for a in critical_activities])
            
            st.markdown(f"""
            <div class="critical-path">
                <h3>🔑 Critical Path Analysis</h3>
                <p><strong>Path:</strong> {path_str}</p>
                <p><strong>Activities:</strong> {activities_str}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed analysis table
        st.subheader("📊 Detailed Activity Analysis")
        analysis_df = analyzer.get_analysis_dataframe()
        
        # Style critical activities
        def highlight_critical(row):
            return ['background-color: #ffebee' if row['Critical'] == '★' else '' for _ in row]
        
        styled_df = analysis_df.style.apply(highlight_critical, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Interactive network diagram
        st.subheader("🕸️ Interactive Network Diagram")
        fig = NetworkVisualizer.create_interactive_network(analyzer)
        st.plotly_chart(fig, use_container_width=True)
        
        # Schedule insights
        st.subheader("📈 Project Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**⚠️ Risk Analysis:**")
            high_risk = [a for a in analyzer.activities.values() if a.is_critical]
            medium_risk = [a for a in analyzer.activities.values() if 0 < a.total_float <= 2]
            
            st.write(f"• {len(high_risk)} critical activities (zero float)")
            st.write(f"• {len(medium_risk)} near-critical activities (≤2 days float)")
            
        with col2:
            st.markdown("**💡 Recommendations:**")
            if len(high_risk) > 0:
                st.write("• Monitor critical activities closely")
                st.write("• Consider resource allocation to critical path")
            if len(medium_risk) > 0:
                st.write("• Watch near-critical activities")

if __name__ == "__main__":
    main()
