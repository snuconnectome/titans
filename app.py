import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import h5py

st.set_page_config(layout="wide", page_title="Titan-Neuro Dashboard")

st.title("🧠 Titan-Neuro Experiment Dashboard")
st.markdown("Real-time monitoring of Titans vs SOTA baselines on NVIDIA AI Workbench.")

# Sidebar
st.sidebar.header("Configuration")
data_path = st.sidebar.text_input("Data Path", "/home/juke/git/ds003020")
refresh = st.sidebar.button("Refresh Data")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Leaderboard", "📉 Training Curves", "🧠 Brain Visualizer"])

# --- Tab 1: Leaderboard ---
with tab1:
    st.header("Benchmark Leaderboard")
    
    # Load Encoding Results
    if os.path.exists("encoding_benchmark_results.json"):
        df = pd.read_json("encoding_benchmark_results.json")
        st.subheader("Task 2: Brain Encoding (Audio -> Brain)")
        
        # Display Table
        st.dataframe(df.style.highlight_min(subset=["Final Loss"], color='lightgreen'))
        
        # Bar Chart
        fig = px.bar(df, x="Model", y="Final Loss", color="Model", title="Final Loss Comparison (Lower is Better)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Speed Comparison
        fig2 = px.bar(df, x="Model", y="Time (s)", color="Model", title="Training Time Comparison (Lower is Better)")
        st.plotly_chart(fig2, use_container_width=True)
        
    else:
        st.info("No encoding benchmark results found yet. Run `train_encoding.py`.")

# --- Tab 2: Training Curves ---
with tab2:
    st.header("Training Dynamics")
    
    if os.path.exists("encoding_benchmark_results.json"):
        df = pd.read_json("encoding_benchmark_results.json")
        
        fig = go.Figure()
        
        for idx, row in df.iterrows():
            history = row["History"]
            if history:
                fig.add_trace(go.Scatter(y=history, mode='lines+markers', name=row["Model"]))
                
        fig.update_layout(title="Loss Curve per Epoch", xaxis_title="Epoch", yaxis_title="MSE Loss")
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Brain Visualizer ---
with tab3:
    st.header("fMRI Slice Viewer")
    
    subject = st.selectbox("Select Subject", ["UTS01", "UTS02", "UTS03"])
    
    # Try to load a real brain file
    hf5_path = os.path.join(data_path, "derivative", "preprocessed_data", subject, f"{subject}_task-adollshouse.hf5")
    # Actually filenames might vary, let's search
    subj_dir = os.path.join(data_path, "derivative", "preprocessed_data", subject)
    
    if os.path.exists(subj_dir):
        files = [f for f in os.listdir(subj_dir) if f.endswith(".hf5")]
        if files:
            selected_file = st.selectbox("Select Run", files)
            file_path = os.path.join(subj_dir, selected_file)
            
            if st.button("Load Brain Volume"):
                try:
                    with h5py.File(file_path, 'r') as f:
                        keys = list(f.keys())
                        data = f[keys[0]][:] # (Time, Voxels)
                        
                    st.success(f"Loaded {data.shape} data.")
                    
                    # Visualization (Mock 3D mapping if flat)
                    # Real ds003020 data is often flat (fsaverage). 
                    # For visualization, we might just show a heatmap of the vector or mapped to a grid.
                    
                    st.subheader("Voxel Activity Map (Flattened)")
                    time_idx = st.slider("Timepoint (TR)", 0, data.shape[0]-1, 0)
                    
                    # Reshape to square for basic viz
                    voxels = data[time_idx]
                    side = int(np.ceil(np.sqrt(voxels.shape[0])))
                    padded = np.zeros(side*side)
                    padded[:voxels.shape[0]] = voxels
                    img = padded.reshape(side, side)
                    
                    fig_brain = px.imshow(img, color_continuous_scale='RdBu_r', title=f"Brain Activity at TR {time_idx}")
                    st.plotly_chart(fig_brain, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error loading hf5: {e}")
        else:
            st.warning("No .hf5 files found in subject dir.")
    else:
        st.warning(f"Subject directory not found: {subj_dir}")

st.sidebar.markdown("---")
st.sidebar.info("Powered by **Titans** & **NVIDIA AI Workbench**")

