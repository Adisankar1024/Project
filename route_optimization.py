import networkx as nx
import numpy as np
import pandas as pd
import random
import plotly.graph_objects as go

def create_road_network(num_nodes=30, seed=42):
    """
    Creates a synthetic road network graph and populates edges with attributes.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate random layout and graph
    # 30 nodes, probability of edge 0.15 to ensure several paths but not fully connected
    G = nx.erdos_renyi_graph(n=num_nodes, p=0.2, seed=seed)
    
    # Ensure graph is connected, if not add edges between components
    components = list(nx.connected_components(G))
    if len(components) > 1:
        for i in range(len(components)-1):
            u = list(components[i])[0]
            v = list(components[i+1])[0]
            G.add_edge(u, v)
    
    # Generate coordinates for mapping
    pos = nx.spring_layout(G, seed=seed)
    for node in G.nodes():
        G.nodes[node]['pos'] = pos[node]
        
    # Assign attributes to edges
    for u, v, data in G.edges(data=True):
        data['distance'] = round(random.uniform(2.0, 15.0), 2)  # km
        data['avg_speed'] = round(random.uniform(40, 90), 1)
        data['acceleration_events_per_km'] = round(random.uniform(0.5, 4.0), 1)
        data['stops_per_km'] = round(random.uniform(0.1, 2.0), 1)
        data['traffic_density'] = random.choice([1, 2, 3])
        data['road_type'] = random.choice([1, 2, 3])
        data['elevation_gain'] = round(random.uniform(-30, 80), 1)
        
    return G

def update_edge_costs_and_predict(G, model, user_scalars=None):
    """
    Apply any user-overridden global attributes, and predict the fuel cost of each edge.
    user_scalars: dict mapping from feature name -> scaler multiplier or replacement
    """
    features = ['distance', 'avg_speed', 'acceleration_events_per_km', 
                'stops_per_km', 'traffic_density', 'road_type', 'elevation_gain']
    
    # Extract data to dataframe for bulk prediction to be efficient
    edge_list = list(G.edges(data=True))
    
    # Build a rapid structured array/dict
    edge_data = []
    keys = []
    
    for u, v, d in edge_list:
        keys.append((u, v))
        row = [d[feat] for feat in features]
        edge_data.append(row)
        
    df = pd.DataFrame(edge_data, columns=features)
    
    # Apply user sliders (multiply base by slider factor)
    if user_scalars:
        if 'traffic_density_mult' in user_scalars:
            df['traffic_density'] = np.clip(np.round(df['traffic_density'] * user_scalars['traffic_density_mult']), 1, 3)
        if 'stops_mult' in user_scalars:
            df['stops_per_km'] = df['stops_per_km'] * user_scalars['stops_mult']
        if 'accel_mult' in user_scalars:
            df['acceleration_events_per_km'] = df['acceleration_events_per_km'] * user_scalars['accel_mult']
        if 'speed_mult' in user_scalars:
            df['avg_speed'] = np.clip(df['avg_speed'] * user_scalars['speed_mult'], 15, 120)
            
    # Model prediction: returns L/100km
    preds = model.predict(df[features])
    
    # Assign back and compute absolute route cost
    for index, (u, v) in enumerate(keys):
        predicted_l_100km = preds[index]
        dist = df.iloc[index]['distance']
        abs_fuel_used = (predicted_l_100km / 100) * dist
        
        G[u][v]['predicted_l_100km'] = predicted_l_100km
        G[u][v]['total_predicted_fuel'] = abs_fuel_used
        
    return G

def find_optimal_routes(G, start_node, end_node):
    """
    Returns shortest distance path and lowest fuel path.
    """
    try:
        # Distance-based
        shortest_path = nx.shortest_path(G, source=start_node, target=end_node, weight='distance')
        shortest_dist = nx.path_weight(G, shortest_path, weight='distance')
        shortest_fuel = nx.path_weight(G, shortest_path, weight='total_predicted_fuel')
        
        # Fuel-based
        fuel_path = nx.shortest_path(G, source=start_node, target=end_node, weight='total_predicted_fuel')
        fuel_dist = nx.path_weight(G, fuel_path, weight='distance')
        fuel_cost = nx.path_weight(G, fuel_path, weight='total_predicted_fuel')
        
        return {
            'shortest': {'path': shortest_path, 'distance': shortest_dist, 'fuel': shortest_fuel},
            'fuel_efficient': {'path': fuel_path, 'distance': fuel_dist, 'fuel': fuel_cost}
        }
    except nx.NetworkXNoPath:
        return None

def plot_network_plotly(G, shortest_path=None, fuel_path=None):
    """
    Create a beautiful interactive plot of the network and the two routes.
    """
    pos = nx.get_node_attributes(G, 'pos')
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition="bottom center",
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='lightblue',
            size=15,
            line_width=2))
            
    # Trace for paths
    traces = [edge_trace, node_trace]
    
    if shortest_path:
        sp_x, sp_y = [], []
        for i in range(len(shortest_path)-1):
            x0, y0 = pos[shortest_path[i]]
            x1, y1 = pos[shortest_path[i+1]]
            sp_x.extend([x0, x1, None])
            sp_y.extend([y0, y1, None])
            
        traces.append(go.Scatter(
            x=sp_x, y=sp_y, mode='lines', 
            name='Shortest Distance', line=dict(color='red', width=4)
        ))
        
    if fuel_path:
        fp_x, fp_y = [], []
        for i in range(len(fuel_path)-1):
            x0, y0 = pos[fuel_path[i]]
            x1, y1 = pos[fuel_path[i+1]]
            fp_x.extend([x0, x1, None])
            fp_y.extend([y0, y1, None])
            
        traces.append(go.Scatter(
            x=fp_x, y=fp_y, mode='lines', 
            name='Fuel Efficient', line=dict(color='green', width=4, dash='dot')
        ))

    fig = go.Figure(data=traces,
             layout=go.Layout(
                title=dict(text='Road Network and Routes', font=dict(size=16)),
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
             )
    return fig
