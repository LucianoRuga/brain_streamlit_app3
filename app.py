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
# FIX SSL (nilearn)
# ============================================================
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# ============================================================
# Load fsaverage once (cached)
# ============================================================
@st.cache_resource
def load_fsaverage():
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    l_verts, l_faces = surface.load_surf_data(fsaverage.pial_left)
    r_verts, r_faces = surface.load_surf_data(fsaverage.pial_right)
    return l_verts, l_faces, r_verts, r_faces


# ============================================================
# Main function to build figure
# ============================================================
def build_figure(atlas, edges_ids, edges_names,
                 shrink, brain_opacity,
                 node_palette,
                 highlight_query,
                 highlight_color,
                 highlight_edge_color):

    atlas = atlas.reset_index(drop=True)
    n_nodes = len(atlas)

    # --------------------------------------------------------
    # Build adjacency matrix
    # --------------------------------------------------------
    adj_matrix = np.zeros((n_nodes, n_nodes))
    id_to_index = {row.roi_id: idx for idx, row in atlas.iterrows()}

    for _, row in edges_ids.iterrows():
        i = id_to_index.get(row["source_id"])
        j = id_to_index.get(row["target_id"])
        if i is not None and j is not None:
            adj_matrix[i, j] = row["weight"]
            adj_matrix[j, i] = row["weight"]

    # Degree info
    atlas["degree"] = np.sum(adj_matrix > 0, axis=1)

    # --------------------------------------------------------
    # Louvain community
    # --------------------------------------------------------
    G = nx.from_numpy_array(adj_matrix)
    partition = community_louvain.best_partition(G, weight="weight")
    atlas["community"] = atlas.index.map(partition)

    # --------------------------------------------------------
    # Determine highlight node
    # --------------------------------------------------------
    highlight_idx = None
    if highlight_query:
        match = atlas[atlas.roi_name.str.contains(highlight_query, case=False, na=False)]
        if len(match) > 0:
            highlight_idx = match.index[0]

    # --------------------------------------------------------
    # Load fsaverage
    # --------------------------------------------------------
    l_verts, l_faces, r_verts, r_faces = load_fsaverage()

    # --------------------------------------------------------
    # Normalize atlas coordinates to brain surface
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Brain meshes
    # --------------------------------------------------------
    brain_lh = go.Mesh3d(
        x=l_verts[:, 0], y=l_verts[:, 1], z=l_verts[:, 2],
        i=l_faces[:, 0], j=l_faces[:, 1], k=l_faces[:, 2],
        color="lightgray", opacity=brain_opacity
    )
    brain_rh = go.Mesh3d(
        x=r_verts[:, 0], y=r_verts[:, 1], z=r_verts[:, 2],
        i=r_faces[:, 0], j=r_faces[:, 1], k=r_faces[:, 2],
        color="lightgray", opacity=brain_opacity
    )

    # ===========================================================
    # EDGES — FIX: normal trace + highlight trace (NO LIST COLORS)
    # ===========================================================
    normal_x, normal_y, normal_z = [], [], []
    high_x, high_y, high_z = [], [], []

    for _, row in edges_ids.iterrows():
        i = id_to_index[row["source_id"]]
        j = id_to_index[row["target_id"]]

        seg_x = [x[i], x[j], None]
        seg_y = [y[i], y[j], None]
        seg_z = [z[i], z[j], None]

        if highlight_idx is not None and (i == highlight_idx or j == highlight_idx):
            high_x += seg_x
            high_y += seg_y
            high_z += seg_z
        else:
            normal_x += seg_x
            normal_y += seg_y
            normal_z += seg_z

    edge_normal = go.Scatter3d(
        x=normal_x, y=normal_y, z=normal_z,
        mode="lines",
        line=dict(color="cyan", width=2),
        hoverinfo="none",
        name="Edges"
    )

    edge_high = go.Scatter3d(
        x=high_x, y=high_y, z=high_z,
        mode="lines",
        line=dict(color=highlight_edge_color, width=6),
        hoverinfo="none",
        name="Highlighted Edges"
    )

    # ===========================================================
    # NODES
    # ===========================================================
    palette = px.colors.qualitative.__dict__[node_palette]

    fig = go.Figure(data=[brain_lh, brain_rh, edge_normal, edge_high])

    for comm in sorted(atlas.community.unique()):
        comm_df = atlas[atlas.community == comm]

        fig.add_trace(go.Scatter3d(
            x=comm_df.x,
            y=comm_df.y,
            z=comm_df.z,
            mode="markers",
            marker=dict(
                size=[15 if i == highlight_idx else 7 for i in comm_df.index],
                color=[
                    highlight_color if i == highlight_idx else palette[comm % len(palette)]
                    for i in comm_df.index
                ],
                opacity=0.85
            ),
            name=f"Community {comm}"
        ))

    # Layout
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
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Brain Connectome 3D", layout="wide")

st.title("🧠 Brain Connectome 3D — Advanced Viewer")

# Sidebar
with st.sidebar:
    st.header("📁 File Input")

    atlas_file = st.file_uploader("Atlas CSV", type="csv")
    edges_ids_file = st.file_uploader("Edges CSV (IDs)", type="csv")
    edges_names_file = st.file_uploader("Edges CSV (Names) [optional]", type="csv")

    st.header("🎨 Visual Settings")
    shrink = st.slider("Shrink nodes", 0.5, 1.0, 0.75)
    brain_opacity = st.slider("Brain opacity", 0.1, 1.0, 0.6)
    palette = st.selectbox("Node palette", ["Dark24", "Bold", "Pastel1", "Set3", "Vivid"])

    st.header("🔍 Highlight Node")
    query = st.text_input("Search ROI name")
    highlight_color = st.color_picker("Node highlight color", "#FF0000")
    highlight_edge_color = st.color_picker("Edge highlight color", "#FFFF00")

    run = st.button("🚀 Generate Connectome")


if run:
    if atlas_file is None or edges_ids_file is None:
        st.error("⚠️ Carica Atlas e Edge List (ID)")
    else:
        atlas = pd.read_csv(atlas_file)
        edges_ids = pd.read_csv(edges_ids_file)
        edges_names = pd.read_csv(edges_names_file) if edges_names_file else None

        with st.spinner("Elaborazione in corso..."):
            fig, idx = build_figure(
                atlas, edges_ids, edges_names,
                shrink, brain_opacity,
                palette,
                query,
                highlight_color,
                highlight_edge_color
            )

        st.plotly_chart(fig, use_container_width=True)

        if idx is not None:
            st.success(f"Nodo trovato: **{atlas.iloc[idx].roi_name}**")
            st.write("📌 Degree:", atlas.iloc[idx].degree)
            st.write("📌 ID:", atlas.iloc[idx].roi_id)
else:
    st.info("⬅️ Carica i dati nella sidebar e premi *Generate Connectome*.")
