import ssl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import community as community_louvain
from nilearn import datasets, surface
import streamlit as st

# ============================================================
# FIX SSL
# ============================================================
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# ============================================================
# Cached fsaverage surfaces
# ============================================================
@st.cache_resource
def load_fsaverage():
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    l_verts, l_faces = surface.load_surf_data(fsaverage.pial_left)
    r_verts, r_faces = surface.load_surf_data(fsaverage.pial_right)
    return l_verts, l_faces, r_verts, r_faces


# ============================================================
# Build connectome figure
# ============================================================
def build_figure(atlas, edges_ids, edges_names,
                 shrink, brain_opacity,
                 node_color_palette,
                 highlight_node,
                 highlight_node_color,
                 highlight_edge_color):

    # Reset index
    atlas = atlas.reset_index(drop=True)
    n_nodes = len(atlas)

    # Build adjacency matrix
    adj_matrix = np.zeros((n_nodes, n_nodes))
    id_to_index = {row.roi_id: idx for idx, row in atlas.iterrows()}

    for _, row in edges_ids.iterrows():
        i = id_to_index.get(row["source_id"])
        j = id_to_index.get(row["target_id"])
        if i is not None and j is not None:
            adj_matrix[i, j] = row["weight"]
            adj_matrix[j, i] = row["weight"]

    # Louvain community detection
    G = nx.from_numpy_array(adj_matrix)
    partition = community_louvain.best_partition(G, weight="weight")
    atlas["community"] = atlas.index.map(partition)
    n_comm = len(atlas["community"].unique())

    # Degree for extra info
    atlas["degree"] = np.sum(adj_matrix > 0, axis=1)

    # Load brain surfaces
    l_verts, l_faces, r_verts, r_faces = load_fsaverage()

    # Normalization + recentering + shrink
    brain_min = np.min(np.vstack([l_verts, r_verts]), axis=0)
    brain_max = np.max(np.vstack([l_verts, r_verts]), axis=0)
    nodes_min = atlas[["x", "y", "z"]].min().values
    nodes_max = atlas[["x", "y", "z"]].max().values

    scale = (brain_max - brain_min) / (nodes_max - nodes_min)
    atlas[["x", "y", "z"]] = (atlas[["x", "y", "z"]] - nodes_min) * scale + brain_min

    mesh_center = (brain_min + brain_max) / 2
    nodes_center = atlas[["x", "y", "z"]].mean().values

    atlas[["x", "y", "z"]] = atlas[["x", "y", "z"]] - nodes_center + mesh_center
    atlas[["x", "y", "z"]] = mesh_center + (atlas[["x", "y", "z"]] - mesh_center) * shrink

    coords = atlas[["x", "y", "z"]].values
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # -------------------------------
    # Brain meshes
    # -------------------------------
    brain_lh = go.Mesh3d(
        x=l_verts[:, 0], y=l_verts[:, 1], z=l_verts[:, 2],
        i=l_faces[:, 0], j=l_faces[:, 1], k=l_faces[:, 2],
        color='lightgray',
        opacity=brain_opacity
    )
    brain_rh = go.Mesh3d(
        x=r_verts[:, 0], y=r_verts[:, 1], z=r_verts[:, 2],
        i=r_faces[:, 0], j=r_faces[:, 1], k=r_faces[:, 2],
        color='lightgray',
        opacity=brain_opacity
    )

    # -------------------------------
    # Edges
    # -------------------------------
    edge_x, edge_y, edge_z, edge_color = [], [], [], []

    highlight_edges_set = set()
    highlight_idx = None

    if highlight_node is not None:
        match = atlas[atlas.roi_name.str.contains(highlight_node, case=False, na=False)]
        if len(match) > 0:
            highlight_idx = match.index[0]

    for idx, row in edges_ids.iterrows():
        i = id_to_index[row["source_id"]]
        j = id_to_index[row["target_id"]]

        edge_x += [x[i], x[j], None]
        edge_y += [y[i], y[j], None]
        edge_z += [z[i], z[j], None]

        if highlight_idx is not None and (i == highlight_idx or j == highlight_idx):
            edge_color.append(highlight_edge_color)
            edge_color.append(highlight_edge_color)
            edge_color.append(None)
        else:
            edge_color.append("cyan")
            edge_color.append("cyan")
            edge_color.append(None)

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(width=2, color=edge_color),
        hoverinfo="none",
        showlegend=False
    )

    # -------------------------------
    # Node colors
    # -------------------------------
    from random import shuffle
    palette = px.colors.qualitative.__dict__[node_color_palette]
    shuffle(palette)

    fig = go.Figure(data=[brain_lh, brain_rh, edge_trace])

    for comm_id in sorted(atlas["community"].unique()):
        comm_df = atlas[atlas.community == comm_id]

        # Default visual properties
        size = 7
        color = palette[comm_id % len(palette)]

        # Highlight node (bigger & colored)
        if highlight_idx is not None:
            if comm_id == atlas.iloc[highlight_idx]["community"]:
                pass

        fig.add_trace(go.Scatter3d(
            x=comm_df.x,
            y=comm_df.y,
            z=comm_df.z,
            mode="markers+text" if highlight_idx else "markers",
            marker=dict(
                size=[15 if i == highlight_idx else size for i in comm_df.index],
                color=[
                    highlight_node_color if i == highlight_idx else color
                    for i in comm_df.index
                ],
                opacity=0.85
            ),
            text=[atlas.iloc[highlight_idx].roi_name] if highlight_idx else None,
            name=f"Community {comm_id}"
        ))

    # -------------------------------
    # Layout
    # -------------------------------
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        )
    )

    return fig, highlight_idx


# ============================================================
# STREAMLIT INTERFACE
# ============================================================
st.set_page_config(page_title="Brain Connectome 3D", layout="wide")

st.title("🧠 Brain Connectome 3D — Advanced Viewer")

with st.sidebar:
    st.header("⚙️ Parametri")

    atlas_file = st.file_uploader("📄 Atlas (CSV)", type="csv")
    edges_ids_file = st.file_uploader("📄 Edge list (ID)", type="csv")
    edges_names_file = st.file_uploader("📄 Edge list (Names)", type="csv")

    st.markdown("---")
    shrink = st.slider("Shrink nodi", 0.5, 1.0, 0.75)
    brain_opacity = st.slider("Trasparenza cervello", 0.1, 1.0, 0.6)

    node_color_palette = st.selectbox(
        "🎨 Palette colori",
        ["Dark24", "Set3", "Bold", "Pastel1", "Vivid"]
    )

    highlight_node = st.text_input("🔍 Cerca ROI (nome)")

    highlight_node_color = st.color_picker("🎯 Colore nodo evidenziato", "#FF0000")
    highlight_edge_color = st.color_picker("🔗 Colore connessioni evidenziate", "#FFFF00")

    run_button = st.button("🚀 Genera connectome")


# ============================================================
# MAIN OUTPUT
# ============================================================
if run_button:
    if atlas_file is None or edges_ids_file is None:
        st.error("Carica almeno Atlas e Edge List (ID).")
    else:
        atlas = pd.read_csv(atlas_file)
        edges_ids = pd.read_csv(edges_ids_file)
        edges_names = pd.read_csv(edges_names_file) if edges_names_file else None

        fig, idx = build_figure(
            atlas,
            edges_ids,
            edges_names,
            shrink,
            brain_opacity,
            node_color_palette,
            highlight_node,
            highlight_node_color,
            highlight_edge_color
        )

        st.plotly_chart(fig, use_container_width=True)

        if idx is not None:
            st.success(f"Nodo trovato: **{atlas.iloc[idx].roi_name}**")
            st.write("📌 **Degree:**", atlas.iloc[idx].degree)
            st.write("📌 **ID:**", atlas.iloc[idx].roi_id)
else:
    st.info("⬅️ Carica i file e clicca *Genera connectome*.")
