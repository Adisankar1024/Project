import pandas as pd
import numpy as np

def generate_truck_data(num_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data for heavy-duty truck fuel consumption based on driving dynamics.
    
    Features:
    - road_segment_id: Unique identifier for a pseudo road segment
    - distance: km, range[0.5, 20.0]
    - avg_speed: km/h, range[15, 90]
    - acceleration_events_per_km: Highly penalty-associated event, range[0, 8]
    - stops_per_km: Penalty, range[0, 5]
    - traffic_density: 1(Low), 2(Medium), 3(High)
    - road_type: 1(Highway), 2(City), 3(Rural)
    - elevation_gain: meters, range[-50, 150]
    
    Target:
    - fuel_consumption: L/100km
    """
    np.random.seed(seed)
    
    data = {
        'road_segment_id': np.arange(1, num_samples + 1),
        'distance': np.round(np.random.uniform(0.5, 20.0, num_samples), 2),
        'avg_speed': np.round(np.random.uniform(15, 90, num_samples), 1),
        'acceleration_events_per_km': np.round(np.random.uniform(0, 8, num_samples), 1),
        'stops_per_km': np.round(np.random.uniform(0, 5, num_samples), 1),
        'traffic_density': np.random.choice([1, 2, 3], num_samples),
        'road_type': np.random.choice([1, 2, 3], num_samples),
        'elevation_gain': np.round(np.random.uniform(-50, 150, num_samples), 1)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate target: fuel_consumption (base around 32 L/100km for a heavy truck)
    # We heavily penalize acceleration and stops, and slightly penalize traffic and elevation
    
    base_consumption = 32.0 
    
    # Traffic penalty
    df['fuel_consumption'] = base_consumption + (df['traffic_density'] * 1.5)
    
    # Speed effect: sweet spot is typically around 65-75 km/h for trucks. 
    # High speed and very low speed consume more.
    df['fuel_consumption'] += np.abs(df['avg_speed'] - 70) * 0.05
    
    # Dynamics (the core focus: acceleration and stops)
    df['fuel_consumption'] += (df['acceleration_events_per_km'] * 1.8)
    df['fuel_consumption'] += (df['stops_per_km'] * 2.5)
    
    # Elevation effect (uphill takes fuel, downhill saves some but breaks consume energy)
    df['fuel_consumption'] += (df['elevation_gain'] * 0.05)
    
    # Road type effect (city=2 consumes more base fuel than highway=1)
    road_penalty = {1: -1.0, 2: 3.0, 3: 0.5}
    df['fuel_consumption'] += df['road_type'].map(road_penalty)
    
    # Add random real-world noise (sigma=1.5 L/100km)
    df['fuel_consumption'] += np.random.normal(0, 1.5, num_samples)
    
    # Ensure minimum reasonable value ~20
    df['fuel_consumption'] = np.maximum(df['fuel_consumption'], 20.0)
    df['fuel_consumption'] = np.round(df['fuel_consumption'], 2)
    
    return df

if __name__ == "__main__":
    df = generate_truck_data(10)
    print("Generated 10 samples for preview:")
    print(df.head())
