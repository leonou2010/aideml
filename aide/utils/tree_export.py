"""Export journal to HTML visualization of tree + code."""

import json
import textwrap
from pathlib import Path

import numpy as np
from igraph import Graph
from ..journal import Journal


def get_edges(journal: Journal):
    for node in journal:
        for c in node.children:
            yield (node.step, c.step)


def generate_layout(n_nodes, edges, layout_type="rt"):
    """Generate visual layout of graph"""
    layout = Graph(
        n_nodes,
        edges=edges,
        directed=True,
    ).layout(layout_type)
    y_max = max(layout[k][1] for k in range(n_nodes))
    layout_coords = []
    for n in range(n_nodes):
        layout_coords.append((layout[n][0], 2 * y_max - layout[n][1]))
    return np.array(layout_coords)


def normalize_layout(layout: np.ndarray):
    """Normalize layout to [0, 1]"""
    layout = (layout - layout.min(axis=0)) / (layout.max(axis=0) - layout.min(axis=0))
    layout[:, 1] = 1 - layout[:, 1]
    layout[:, 1] = np.nan_to_num(layout[:, 1], nan=0)
    layout[:, 0] = np.nan_to_num(layout[:, 0], nan=0.5)
    return layout


def strip_code_markers(code: str) -> str:
    """Remove markdown code block markers if present."""
    code = code.strip()
    if code.startswith("```"):
        # Remove opening backticks and optional language identifier
        first_newline = code.find("\n")
        if first_newline != -1:
            code = code[first_newline:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return code


def cfg_to_tree_struct(cfg, jou: Journal, selected_nodes=None):
    edges = list(get_edges(jou))
    layout = normalize_layout(generate_layout(len(jou), edges))

    # Get actual metrics from nodes
    metrics = []
    metric_values = []
    is_buggy = []
    seen_nodes_per_node = []  # Track which nodes each node "saw"
    
    for n in jou:
        if hasattr(n, 'metric') and n.metric:
            if hasattr(n.metric, 'value') and n.metric.value is not None:
                metrics.append(n.metric.value)
                metric_values.append(f"{n.metric.value:.4f}")
            else:
                metrics.append(0)
                metric_values.append("N/A")
        else:
            metrics.append(0)
            metric_values.append("N/A")
        is_buggy.append(getattr(n, 'is_buggy', False))
        
        # NEW: Get the nodes this node "saw" when it was created
        seen_nodes = getattr(n, 'seen_nodes', [])
        seen_nodes_per_node.append(seen_nodes)

    # Mark which nodes were selected for summary
    node_selected = [False for n in jou.nodes]
    if selected_nodes:
        for node in selected_nodes:
            if node.step < len(node_selected):
                node_selected[node.step] = True

    return dict(
        edges=edges,
        layout=layout.tolist(),
        plan=[textwrap.fill(n.plan, width=80) for n in jou.nodes],
        code=[strip_code_markers(n.code) for n in jou],
        term_out=[n.term_out for n in jou],
        analysis=[n.analysis for n in jou],
        exp_name=cfg.exp_name,
        metrics=metrics,
        metric_values=metric_values,
        is_buggy=is_buggy,
        selected_for_summary=node_selected,
        seen_nodes_per_node=seen_nodes_per_node,  # This line exists but wasn't populated
    )

def generate_html(tree_graph_str: str):
    template_dir = Path(__file__).parent / "viz_templates"

    with open(template_dir / "template.js") as f:
        js = f.read()
        js = js.replace("/*<placeholder>*/ {}", tree_graph_str)

    with open(template_dir / "template.html") as f:
        html = f.read()
        html = html.replace("<!-- placeholder -->", js)

        return html


def generate(cfg, jou: Journal, out_path: Path, selected_nodes=None):
    tree_graph_str = json.dumps(cfg_to_tree_struct(cfg, jou, selected_nodes))
    html = generate_html(tree_graph_str)
    with open(out_path, "w") as f:
        f.write(html)
