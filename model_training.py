import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_fuel_model(df: pd.DataFrame):
    """
    Train a RandomForestRegressor to predict fuel consumption on a generated dataset.
    Returns the trained model, performance metrics, and feature importances.
    """
    # Exclude non-predictive features
    features = ['distance', 'avg_speed', 'acceleration_events_per_km', 
                'stops_per_km', 'traffic_density', 'road_type', 'elevation_gain']
    target = 'fuel_consumption'
    
    X = df[features]
    y = df[target]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Feature Importances
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    metrics = {
        'MAE': round(mae, 3),
        'RMSE': round(rmse, 3)
    }
    
    return model, metrics, importances

if __name__ == "__main__":
    from data_generation import generate_truck_data
    df = generate_truck_data(2000)
    model, metrics, importances = train_fuel_model(df)
    print("Model Metrics:", metrics)
    print("\nFeature Importances:\n", importances)
