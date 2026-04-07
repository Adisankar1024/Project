import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

# Custom modules
from data_generation import generate_truck_data
from model_training import train_fuel_model
from route_optimization import (
    create_road_network, 
    update_edge_costs_and_predict, 
    find_optimal_routes, 
    plot_network_plotly
)

st.set_page_config(page_title="AI Fuel-Efficient Route Optimization", layout="wide", page_icon="🚛")

# ------------- Constants & Caching -------------
@st.cache_data
def load_data():
    return generate_truck_data(3000, seed=123)

@st.cache_resource
def load_model(df):
    model, metrics, importances = train_fuel_model(df)
    return model, metrics, importances

@st.cache_resource
def load_graph():
    return create_road_network(num_nodes=30, seed=42)

# Load resources
st.title("🚛 AI-Driven Fuel-Efficient Route Optimization System")
st.markdown("A demonstration of optimizing heavy-duty truck routes to minimize **fuel consumption** rather than just distance, by avoiding excessive accelerations, stops, and poor traffic.")

df = load_data()
model, metrics, importances = load_model(df)
base_graph = load_graph()

# ------------- Sidebar -------------
st.sidebar.header("🔄 Real-Time Traffic Conditions")
st.sidebar.markdown("Simulate changes across the network to see how the optimal route shifts.")

traffic_mult = st.sidebar.slider("Traffic Density Multiplier", 0.5, 3.0, 1.0, 0.1)
accel_mult = st.sidebar.slider("Acceleration Events Multiplier", 0.1, 5.0, 1.0, 0.1)
stops_mult = st.sidebar.slider("Stops Multiplier", 0.1, 3.0, 1.0, 0.1)
speed_mult = st.sidebar.slider("Avg Speed Multiplier", 0.5, 1.5, 1.0, 0.1)

user_scalars = {
    'traffic_density_mult': traffic_mult,
    'accel_mult': accel_mult,
    'stops_mult': stops_mult,
    'speed_mult': speed_mult
}

# ------------- Layout -------------
tab1, tab2, tab3 = st.tabs(["🛣️ Route Simulation", "📊 Dataset & ML Model", "ℹ️ About the System"])

with tab2:
    st.header("Dataset Preview")
    st.dataframe(df.head(10))
    st.caption(f"Showing 10 rows from {len(df)} synthetic samples generated.")
    
    st.divider()
    
    st.header("Model Performance")
    colA, colB = st.columns(2)
    colA.metric(label="Mean Absolute Error (L/100km)", value=metrics['MAE'])
    colB.metric(label="Root Mean Squared Error (L/100km)", value=metrics['RMSE'])
    
    st.subheader("RandomForest Feature Importance")
    fig = px.bar(importances, x='Importance', y='Feature', orientation='h', 
                 title="Impact of Road Dynamics on Fuel Consumption")
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)


with tab1:
    st.header("Interactive Route Maps")
    
    c1, c2 = st.columns([1, 1])
    # For selection
    nodes_list = list(base_graph.nodes())
    with c1: start_node = st.selectbox("Start Location (Node)", nodes_list, index=0)
    with c2: end_node = st.selectbox("Destination (Node)", nodes_list, index=len(nodes_list)-1)
    
    if start_node == end_node:
        st.warning("Please select distinct Start and End locations.")
    else:
        # Clone graph to apply live predictions safely
        live_graph = base_graph.copy()
        live_graph = update_edge_costs_and_predict(live_graph, model, user_scalars)
        
        # Calculate Paths
        routes = find_optimal_routes(live_graph, start_node, end_node)
        
        if routes is None:
            st.error("No path found between these nodes!")
        else:
            sr = routes['shortest']
            fr = routes['fuel_efficient']
            
            # --- Visualization ---
            fig_map = plot_network_plotly(
                live_graph, 
                sr['path'], 
                fr['path']
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
            # --- Comparison Metrics ---
            st.subheader("Optimization Results")
            
            # Helper to calculate absolute fuel (Liters)
            # The algorithm outputs the fuel consumed directly.
            
            # Calculate metrics
            dist_short = sr['distance']
            fuel_short = sr['fuel']
            
            dist_eff = fr['distance']
            fuel_eff = fr['fuel']
            
            # Carbon emission conversion (~2.68 kg CO2 per liter of diesel)
            co2_short = fuel_short * 2.68
            co2_eff = fuel_eff * 2.68
            
            fuel_savings_pct = max(0, ((fuel_short - fuel_eff) / fuel_short) * 100)
            
            rc1, rc2, rc3 = st.columns(3)
            
            with rc1:
                st.markdown("### 🗺️ Traditional Path")
                st.markdown("_Optimizing for shortest Distance_")
                st.metric("Total Distance", f"{dist_short:.2f} km")
                st.metric("Total Fuel Consumed", f"{fuel_short:.2f} L")
                st.metric("Est. CO2 Emissions", f"{co2_short:.2f} kg")
                
            with rc2:
                st.markdown("### 🍃 Fuel-Efficient Path")
                st.markdown("_Optimizing via Machine Learning_")
                
                dist_delta = dist_eff - dist_short
                fuel_delta = fuel_short - fuel_eff # Inverse for green
                co2_delta = co2_short - co2_eff
                
                st.metric("Total Distance", f"{dist_eff:.2f} km", f"{dist_delta:+.2f} km", delta_color="inverse")
                st.metric("Total Fuel Consumed", f"{fuel_eff:.2f} L", f"{fuel_delta:+.2f} L savings", delta_color="normal")
                st.metric("Est. CO2 Emissions", f"{co2_eff:.2f} kg", f"{co2_delta:+.2f} kg savings", delta_color="normal")
                
            with rc3:
                st.markdown("### 💹 Final Impact")
                st.markdown(f"**Fuel Savings:** `<span style='color:green;font-size:32px'>{fuel_savings_pct:.1f}%</span>`", unsafe_allow_html=True)
                
                if fuel_savings_pct > 0:
                    st.success(f"By driving an extra {dist_delta:.2f} km to avoid harsh accelerations and excessive stops, you save {fuel_delta:.2f} liters of fuel!")
                elif fuel_savings_pct == 0:
                    st.info("The shortest path is already the most fuel-efficient route under the current conditions.")

with tab3:
    st.markdown("""
    ### About the System
    This prototype simulates a complete AI routing engine:
    1. **Data**: Generates synthetic truck data, mapping high penalties to harsh events like accelerations or stop-and-go traffic.
    2. **Machine Learning Model**: Uses a `RandomForestRegressor` to map telemetry features onto raw consumption.
    3. **Graph Theory**: A simulated road network where every edge runs through the ML inference pipeline so the absolute minimum total energy cost footprint path can be calculated via Dijkstra's algorithm.
    """)
