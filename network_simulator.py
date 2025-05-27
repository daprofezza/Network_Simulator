# In the CSS styling section, add this animation:
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
</style>
""", unsafe_allow_html=True)

# In the draw_network_diagram method, modify the edge labels section:
# Change from:
if is_critical:
    edge_labels[(u, v)] = f"⭐ {activity}\n({duration}d)"
else:
    edge_labels[(u, v)] = f"{activity}\n({duration}d)"

# To:
if is_critical:
    edge_labels[(u, v)] = f"{activity}\n({duration}d" + "⭐)"
else:
    edge_labels[(u, v)] = f"{activity}\n({duration}d)"

# And add the pulsing class to critical elements:
if critical_edges:
    # Outer glow with pulse effect
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
    
    # Main critical path with pulse effect
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

# In the Activity Analysis table section, modify the display_df:
# Change from:
display_df = df[['Activity', 'Start_Node', 'End_Node', 'Duration', 'ES', 'EF', 'LS', 'LF', 'Float', 'Free_Float', 'Critical']].copy()
display_df.columns = ['Activity', 'Start Node', 'End Node', 'Duration (days)', 'Early Start (ES)', 
                     'Early Finish (EF)', 'Late Start (LS)', 'Late Finish (LF)', 
                     'Total Float', 'Free Float', 'Critical?']

# To:
display_df = df[['Activity', 'Start_Node', 'End_Node', 'Duration', 'ES', 'EF', 'LS', 'LF', 'Float', 'Free_Float']].copy()
display_df.columns = ['Activity', 'Start Node', 'End Node', 'Duration (days)', 'Early Start (ES)', 
                     'Early Finish (EF)', 'Late Start (LS)', 'Late Finish (LF)', 
                     'Total Float', 'Free Float']

# And update the styling function:
def style_analysis_table(row):
    styles = [''] * len(row)
    float_val = row['Total Float']
    if float_val <= 0.001:  # Critical activities
        styles = ['background-color: #ffcdd2; font-weight: bold; color: #b71c1c'] * len(row)
    elif float_val <= 1:  # Near-critical
        styles = ['background-color: #fff9c4; color: #f57f17'] * len(row)
    else:  # Regular
        styles = ['background-color: #e8f5e8; color: #2e7d32'] * len(row)
    return styles
