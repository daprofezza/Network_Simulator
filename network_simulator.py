import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import ArrowStyle
import csv
import itertools
from matplotlib.animation import FuncAnimation
import platform
from tkinter.font import Font

# Tooltip replacement for idlelib.tooltip (EXE compatibility)
class Hovertip:
    def __init__(self, anchor_widget, text, hover_delay=1000):
        self.anchor_widget = anchor_widget
        self.text = text
        self.hover_delay = hover_delay
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        
        self.anchor_widget.bind('<Enter>', self.enter)
        self.anchor_widget.bind('<Leave>', self.leave)
        
    def enter(self, event=None):
        self.schedule()
        
    def leave(self, event=None):
        self.unschedule()
        self.hidetip()
        
    def schedule(self):
        self.unschedule()
        self.id = self.anchor_widget.after(self.hover_delay, self.showtip)
        
    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.anchor_widget.after_cancel(id)
            
    def showtip(self, event=None):
        try:
            x, y, cx, cy = self.anchor_widget.bbox("insert")
        except:
            x = y = 0
        x += self.anchor_widget.winfo_rootx() + 25
        y += self.anchor_widget.winfo_rooty() + 25
        
        self.tipwindow = tw = tk.Toplevel(self.anchor_widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                        background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                        font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)
        
    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

class ProjectNetworkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Network Diagram Simulator")
        self.graph = nx.DiGraph()
        self.activities = []
        self.critical_path = []
        self.project_duration = 0
        self.float_table = []
        self.anim = None
        self.text_animation = None

        # Improved font settings
        try:
            if platform.system() == 'Windows':
                font_family = 'Segoe UI'
            else:
                font_family = 'Arial'
        except:
            font_family = 'Arial'
            
        self.large_font = Font(family=font_family, size=16, weight='bold')
        self.medium_font = Font(family=font_family, size=14)
        self.small_font = Font(family=font_family, size=12)
        
        # Modern color scheme
        self.colors = {
            'primary': '#4a6baf',
            'secondary': '#6c757d',
            'success': '#28a745',
            'danger': '#dc3545',
            'warning': '#ffc107',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40',
            'background': '#f0f2f5',
            'text': '#212529'
        }
        
        self.root.configure(bg=self.colors['background'])
        self._setup_widgets()

    def _setup_widgets(self):
        # Input Frame with improved styling
        self.input_frame = tk.Frame(self.root, bg='#ffffff', padx=15, pady=15, 
                                   highlightbackground='#e0e0e0', highlightthickness=1)
        self.input_frame.pack(pady=10, fill='x', padx=10)

        labels = ["Activity", "Start Node", "End Node", "Duration"]
        tooltips = [
            "Unique identifier for each task\n\nExample: A, B, C, etc.",
            "Beginning node of the activity\n\nMust connect to other nodes\nExample: 1, 2",
            "Completion node of the activity\n\nMust differ from start node\nExample: 2, 3",
            "Time required to complete activity\n\nMust be positive integer (days)\nExample: 3, 5, 10"
        ]
        self.entries = []
        for i, (label, tip) in enumerate(zip(labels, tooltips)):
            lbl = tk.Label(self.input_frame, text=label+":", bg='#ffffff', 
                          font=self.medium_font, fg=self.colors['text'])
            lbl.grid(row=0, column=i*2, padx=(0,5))
            entry = tk.Entry(self.input_frame, font=self.medium_font, 
                            bd=1, relief=tk.SOLID, highlightbackground='#e0e0e0',
                            highlightcolor=self.colors['primary'], highlightthickness=1)
            entry.grid(row=0, column=i*2+1, padx=5)
            self.entries.append(entry)
            Hovertip(entry, tip, hover_delay=300)
        self.activity_entry, self.start_entry, self.end_entry, self.duration_entry = self.entries

        # Status message area
        self.error_label = tk.Label(self.root, text="", fg="#ffffff", bg=self.colors['primary'], 
                                   font=self.medium_font, padx=10, pady=8, anchor='w')
        self.error_label.pack(fill='x', padx=10)

        # Button Frame with icons and improved styling
        self.button_frame = tk.Frame(self.root, bg=self.colors['background'], pady=10)
        self.button_frame.pack(fill='x', padx=10)

        buttons = [
            ("📝 Add Activity", self.add_activity, self.colors['primary'], "Add the current activity to the network"),
            ("↩️ Undo Last", self.undo_last_activity, self.colors['warning'], "Remove the most recently added activity"),
            ("🗑️ Clear All", self.clear_all, self.colors['danger'], "Reset the entire application"),
            ("📊 Draw Network", self.draw_network, self.colors['success'], "Visualize the current network structure"),
            ("🔍 Find Critical Path", self.compute_critical_path, self.colors['info'], "Calculate the longest path in the network"),
            ("🎲 Dummy Data", self.add_dummy_data, self.colors['secondary'], "Load sample data for demonstration"),
            ("💾 Save Image", self.save_diagram, '#6f42c1', "Export the diagram as image file"),
            ("📤 Export CSV", self.export_to_csv, '#20c997', "Save activity data to spreadsheet")
        ]

        for i, (text, cmd, color, tip) in enumerate(buttons):
            btn = tk.Button(self.button_frame, text=text, command=cmd, bg=color, fg='white',
                          font=self.medium_font, relief='flat', padx=12, pady=8,
                          activebackground=color, activeforeground='white',
                          bd=0, highlightthickness=0, compound=tk.LEFT)
            btn.grid(row=0, column=i, padx=3)
            Hovertip(btn, tip, hover_delay=300)

        # Main content area
        self.body_frame = tk.Frame(self.root, bg=self.colors['background'])
        self.body_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        self.body_frame.columnconfigure(0, weight=1)
        self.body_frame.columnconfigure(1, weight=3)
        self.body_frame.rowconfigure(0, weight=1)

        # Table with single set of headers
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except:
            style.theme_use('default')
            
        style.configure("Treeview.Heading", font=self.medium_font, 
                       background=self.colors['primary'], foreground='white')
        style.configure("Treeview", font=self.medium_font, rowheight=30,
                      background='#ffffff', fieldbackground='#ffffff')
        style.map("Treeview.Heading", background=[('active', '#3a5a8f')])
        
        # Create the table
        self.table = ttk.Treeview(
            self.body_frame,
            columns=("Activity", "Start", "End", "Duration", "ES", "EF", "LS", "LF", "Float"),
            show='headings',
            style="Treeview"
        )
        
        # Configure column headers
        headers = {
            "Activity": "Task identifier\n(e.g., A, B, C)",
            "Start": "Starting node\n(where activity begins)",
            "End": "Ending node\n(where activity completes)",
            "Duration": "Time required\n(in days)",
            "ES": "Earliest Start time\n(calculated from project start)",
            "EF": "Earliest Finish time\n(ES + Duration)",
            "LS": "Latest Start time\n(without delaying project)",
            "LF": "Latest Finish time\n(without delaying project)",
            "Float": "Slack time available\n(zero = critical activity)"
        }
        
        for col in headers:
            self.table.heading(col, text=col)
            self.table.column(col, width=100, anchor='center')

        # Add scrollbars
        yscroll = ttk.Scrollbar(self.body_frame, orient='vertical', command=self.table.yview)
        xscroll = ttk.Scrollbar(self.body_frame, orient='horizontal', command=self.table.xview)
        self.table.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        
        # Grid the widgets
        self.table.grid(row=0, column=0, sticky='nsew', padx=(0,5))
        yscroll.grid(row=0, column=1, sticky='ns')
        xscroll.grid(row=1, column=0, sticky='ew')

        # Result banner with improved styling
        self.result_banner = tk.Frame(self.body_frame, bg=self.colors['light'], pady=12, padx=15, 
                                    highlightbackground='#e0e6ed', highlightthickness=1)
        self.result_banner.grid(row=1, column=0, sticky='ew', pady=(5,0))

        self.result_label = tk.Label(
            self.result_banner,
            text="",
            font=self.large_font,
            fg=self.colors['dark'],
            bg=self.colors['light'],
            justify=tk.LEFT,
            wraplength=450
        )
        self.result_label.pack(anchor='w')

        # Canvas frame with improved styling
        self.canvas_frame = tk.Frame(self.body_frame, bg='#ffffff', 
                                   highlightbackground='#e0e6ed', highlightthickness=1)
        self.canvas_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(5,0))

        # Configure matplotlib figure
        try:
            plt.style.use('ggplot')
        except:
            plt.style.use('default')
            
        self.figure, self.ax = plt.subplots(figsize=(7, 6), facecolor=self.colors['light'])
        self.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.canvas_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Watermark with improved styling
        self.watermark_label = tk.Label(
            self.root,
            text="© Network Diagram Simulator | Developed by J. Inigo Papu Vinodhan, Asst. Prof., BBA Dept., St. Joseph's College, Trichy",
            font=self.small_font, fg=self.colors['secondary'], bg=self.colors['background']
        )
        self.watermark_label.pack(side="bottom", pady=(0,10))

    def add_activity(self):
        try:
            activity = self.activity_entry.get().strip()
            start = self.start_entry.get().strip()
            end = self.end_entry.get().strip()
            duration = int(self.duration_entry.get().strip())

            if not all([activity, start, end]) or duration <= 0:
                raise ValueError("All fields must be filled with valid values.")

            if (start, end) in self.graph.edges:
                raise ValueError("This connection already exists.")

            self.graph.add_edge(start, end, duration=duration, activity=activity)
            self.activities.append((activity, start, end, duration))
            self.table.insert('', 'end', values=(activity, start, end, duration, '', '', '', '', ''))

            for entry in self.entries:
                entry.delete(0, tk.END)
            self.error_label.config(text="✓ Activity added successfully", bg=self.colors['success'])

        except ValueError as e:
            self.error_label.config(text=f"⚠ Error: {str(e)}", bg=self.colors['danger'])
        except Exception as e:
            self.error_label.config(text=f"⚠ Unexpected error: {str(e)}", bg=self.colors['danger'])

    def undo_last_activity(self):
        if self.activities:
            last = self.activities.pop()
            try:
                # Remove edge from graph
                self.graph.remove_edge(last[1], last[2])
                
                # Find and remove the corresponding row from table
                rows_to_delete = []
                for row in self.table.get_children():
                    row_values = self.table.item(row)['values']
                    # Compare the first 4 values (Activity, Start, End, Duration)
                    if (str(row_values[0]) == str(last[0]) and 
                        str(row_values[1]) == str(last[1]) and 
                        str(row_values[2]) == str(last[2]) and 
                        str(row_values[3]) == str(last[3])):
                        rows_to_delete.append(row)
                
                # Delete all matching rows (in case of duplicates)
                for row in rows_to_delete:
                    self.table.delete(row)
                
                # Clear any existing calculations
                self.critical_path.clear()
                self.float_table.clear()
                self.project_duration = 0
                self.result_label.config(text="")
                
                # Redraw network if there are still activities
                if self.activities:
                    self.draw_network()
                else:
                    self.ax.clear()
                    self.canvas.draw()
                
                self.error_label.config(text="✓ Last activity removed", bg=self.colors['success'])
                
            except Exception as e:
                # If there's an error, add the activity back
                self.activities.append(last)
                self.error_label.config(text=f"⚠ Error removing activity: {str(e)}", bg=self.colors['danger'])
        else:
            self.error_label.config(text="⚠ No activities to remove", bg=self.colors['warning'])

    def clear_all(self):
        if self.anim:
            self.anim.event_source.stop()
        if self.text_animation:
            self.root.after_cancel(self.text_animation)
        
        self.activities.clear()
        self.graph.clear()
        self.critical_path.clear()
        self.project_duration = 0
        self.float_table.clear()
        for row in self.table.get_children():
            self.table.delete(row)
        self.result_label.config(text="")
        self.ax.clear()
        self.canvas.draw()
        self.error_label.config(text="✓ All activities cleared", bg=self.colors['success'])

    def compute_critical_path(self):
        try:
            if not self.activities:
                raise ValueError("No activities to analyze")

            durations = nx.get_edge_attributes(self.graph, 'duration')
            topo_order = list(nx.topological_sort(self.graph))
            es, ef, ls, lf = {}, {}, {}, {}

            # Forward pass
            for node in topo_order:
                es[node] = max((ef[p] for p in self.graph.predecessors(node)), default=0)
                ef[node] = es[node] + max((durations.get((node, s), 0) for s in self.graph.successors(node)), default=0)

            project_duration = max(ef.values())
            
            # Backward pass
            for node in reversed(topo_order):
                lf[node] = min((ls.get(s, project_duration) - durations.get((node, s), 0) for s in self.graph.successors(node)), default=project_duration)
                ls[node] = lf[node] - max((durations.get((node, s), 0) for s in self.graph.successors(node)), default=0)

            # Calculate floats and identify critical path
            self.critical_path.clear()
            self.float_table.clear()
            for u, v, data in self.graph.edges(data=True):
                es_u = es[u]
                ef_u = es_u + data['duration']
                lf_v = lf[v]
                ls_u = lf_v - data['duration']
                fl = lf_v - ef_u
                self.float_table.append((data['activity'], es_u, ef_u, ls_u, lf_v, fl))
                if fl == 0:
                    self.critical_path.append((u, v))

            # Update table with calculated values
            for i, row in enumerate(self.table.get_children()):
                act = self.table.item(row)['values'][0]
                for r in self.float_table:
                    if r[0] == act:
                        self.table.item(row, values=(r[0], self.table.item(row)['values'][1], 
                                      self.table.item(row)['values'][2], self.table.item(row)['values'][3], 
                                      r[1], r[2], r[3], r[4], r[5]))
                        break

            self.project_duration = project_duration
            path_nodes = [u for u, v in self.critical_path] + [self.critical_path[-1][1]] if self.critical_path else []
            
            # Display results with animation
            self.result_label.config(text="")
            self.animate_text(f"🔑 Critical Path: {' → '.join(path_nodes)}\n⏱️ Project Duration: {self.project_duration} days")
            self.draw_network(es, lf, animate=True)

        except Exception as e:
            self.error_label.config(text=f"⚠ Error: {str(e)}", bg=self.colors['danger'])

    def animate_text(self, msg):
        if self.text_animation:
            self.root.after_cancel(self.text_animation)
        
        colors = [self.colors['primary'], self.colors['info'], 
                 self.colors['success'], self.colors['warning'],
                 self.colors['danger'], '#6f42c1']
        
        def update(i=0):
            color = colors[i % len(colors)]
            self.result_label.config(fg=color)
            self.text_animation = self.root.after(800, update, i+1)
        
        self.result_label.config(text=msg)
        update()

    def draw_network(self, es_map=None, lf_map=None, animate=False):
        if self.anim:
            self.anim.event_source.stop()
            
        self.ax.clear()
        pos = nx.spring_layout(self.graph, seed=42, k=0.8)
        
        # Color scheme
        base_node_color = '#e6f3ff'
        base_edge_color = '#a0aec0'
        critical_node_color = '#ff6b6b'
        critical_edge_color = '#ff3860'
        
        # Draw base network with properly sized arrowheads
        nx.draw_networkx_nodes(self.graph, pos, ax=self.ax, 
                             node_color=base_node_color, 
                             node_size=1200,
                             edgecolors=self.colors['primary'],
                             linewidths=2)
        
        # Draw edges with better looking arrows
        nx.draw_networkx_edges(self.graph, pos, ax=self.ax, 
                             edge_color=base_edge_color,
                             width=1.5,
                             arrows=True,
                             arrowsize=15,  # Reduced arrow size
                             arrowstyle='-|>',  # Standard arrow style
                             node_size=1200)
        
        nx.draw_networkx_labels(self.graph, pos, ax=self.ax, 
                              font_size=12,
                              font_weight='bold',
                              font_color=self.colors['dark'])

        # Edge labels
        edge_labels = {(u, v): f"{d['activity']}\n{d['duration']}" 
                       for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, 
                                   ax=self.ax, 
                                   font_size=12,
                                   bbox=dict(facecolor='white', 
                                            edgecolor='none',
                                            alpha=0.8))

        # Add ES/LF labels if provided
        if es_map and lf_map:
            for node in self.graph.nodes:
                x, y = pos[node]
                self.ax.text(x, y + 0.12, f"ES={es_map.get(node, '')}", 
                            fontsize=12, ha='center', color=self.colors['success'])
                self.ax.text(x, y - 0.12, f"LF={lf_map.get(node, '')}", 
                            fontsize=12, ha='center', color=self.colors['danger'])

        # Critical path animation with flowing arrows
        if animate and self.critical_path:
            cp_edges = set(self.critical_path)
            cp_nodes = set(u for u, v in cp_edges) | set(v for u, v in cp_edges)
            
            # Animation colors
            pulse_colors = ['#ff3860', '#ff6b6b', '#ff8a5b', '#ffb347', '#ffdd59', '#f6e58d']
            
            # Create animated arrows for critical path
            arrow_objects = []
            for u, v in cp_edges:
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                arrow = self.ax.annotate("",
                    xy=(x2, y2), xycoords='data',
                    xytext=(x1, y1), textcoords='data',
                    arrowprops=dict(arrowstyle='-|>',
                                  color=pulse_colors[0],
                                  lw=2,
                                  shrinkA=12, shrinkB=12))
                arrow_objects.append(arrow)
            
            def update(i):
                color = pulse_colors[i % len(pulse_colors)]
                
                # Update all arrows
                for arrow in arrow_objects:
                    arrow.arrow_patch.set_color(color)
                    arrow.arrow_patch.set_linewidth(2)
                
                # Update node colors
                nx.draw_networkx_nodes(self.graph, pos, nodelist=cp_nodes, 
                                     ax=self.ax, 
                                     node_color=color,
                                     node_size=1200,
                                     edgecolors=self.colors['danger'],
                                     linewidths=2)
                
                self.canvas.draw()

            self.anim = FuncAnimation(self.figure, update, interval=600, cache_frame_data=False)
        
        self.ax.set_facecolor(self.colors['light'])
        self.figure.set_facecolor(self.colors['light'])
        self.canvas.draw()

    def add_dummy_data(self):
        if not self.activities:
            dummy = [("A", "1", "2", 4), ("B", "1", "3", 3), 
                    ("C", "2", "4", 2), ("D", "3", "4", 5), 
                    ("E", "4", "5", 1)]
            for act, s, e, d in dummy:
                self.graph.add_edge(s, e, duration=d, activity=act)
                self.activities.append((act, s, e, d))
                self.table.insert('', 'end', values=(act, s, e, d, '', '', '', '', ''))
            self.error_label.config(text="✓ Sample data loaded successfully", bg=self.colors['success'])
        else:
            self.error_label.config(text="⚠ Clear existing data before loading sample", bg=self.colors['warning'])

    def save_diagram(self):
        try:
            file = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), 
                          ("JPEG Image", "*.jpg"), 
                          ("PDF Document", "*.pdf"),
                          ("SVG Vector", "*.svg")],
                title="Save Network Diagram"
            )
            if file:
                self.figure.savefig(file, dpi=300, bbox_inches='tight', facecolor=self.figure.get_facecolor())
                self.error_label.config(text=f"✓ Diagram saved to {file}", bg=self.colors['success'])
        except Exception as e:
            self.error_label.config(text=f"⚠ Error saving file: {str(e)}", bg=self.colors['danger'])

    def export_to_csv(self):
        try:
            file = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), 
                          ("Excel Files", "*.xlsx")],
                title="Export Activity Data"
            )
            if file:
                with open(file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Activity", "Start Node", "End Node", "Duration"])
                    writer.writerows(self.activities)
                self.error_label.config(text=f"✓ Data exported to {file}", bg=self.colors['success'])
        except Exception as e:
            self.error_label.config(text=f"⚠ Error exporting file: {str(e)}", bg=self.colors['danger'])

if __name__ == '__main__':
    try:
        root = tk.Tk()
        root.geometry("1280x800")
        root.minsize(1024, 600)
        app = ProjectNetworkApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Application Error", f"Failed to start application: {str(e)}")
