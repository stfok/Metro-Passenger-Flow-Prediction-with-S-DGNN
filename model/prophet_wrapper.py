import numpy as np
import pandas as pd
from fbprophet import Prophet
from config import Config

class ProphetWrapper:
    def __init__(self, growth='linear', daily_seasonality=True, weekly_seasonality=True):
        """
        Initialize the Prophet wrapper for correlation prediction
        
        Args:
            growth: Trend growth type ('linear' or 'logistic')
            daily_seasonality: Whether to include daily seasonality
            weekly_seasonality: Whether to include weekly seasonality
        """
        self.models = {}
        self.cfg = Config()
        self.growth = growth
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
    
    def fit(self, edge_histories):
        """
        Train Prophet models for each edge in the transfer-centric graph
        
        Args:
            edge_histories: Dictionary of historical correlations for each edge
                            Format: {(source, target): [corr1, corr2, ...]}
        """
        self.models = {}
        
        for (src, tgt), correlations in edge_histories.items():
            # Create time series dataframe for Prophet
            dates = pd.date_range(start='2023-01-01', periods=len(correlations), freq='15T')
            df = pd.DataFrame({
                'ds': dates,
                'y': correlations
            })
            
            # Initialize and fit Prophet model
            model = Prophet(
                growth=self.growth,
                daily_seasonality=self.daily_seasonality,
                weekly_seasonality=self.weekly_seasonality
            )
            model.fit(df)
            
            self.models[(src, tgt)] = model
    
    def predict(self, timestamps):
        """
        Predict correlations for specified timestamps
        
        Args:
            timestamps: List of datetime objects to predict for
            
        Returns:
            Dictionary of predicted correlations for each edge
            Format: {(source, target): [pred1, pred2, ...]}
        """
        predictions = {}
        
        for edge, model in self.models.items():
            # Create prediction dataframe
            future = pd.DataFrame({'ds': timestamps})
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Clip predictions to [0, 1] range
            preds = forecast['yhat'].clip(0, 1).values
            predictions[edge] = preds
        
        return predictions

    def predict_correlation_matrix(self, timestamps, adj_matrix):
        """
        Predict the full time-varying correlation matrix
        
        Args:
            timestamps: List of datetime objects to predict for
            adj_matrix: Adjacency matrix of the transfer-centric graph
            
        Returns:
            List of correlation matrices for each timestamp
        """
        num_nodes = adj_matrix.shape[0]
        num_timesteps = len(timestamps)
        
        # Initialize empty correlation matrices
        correlation_matrices = np.zeros((num_timesteps, num_nodes, num_nodes))
        
        # Get predictions for all edges
        edge_predictions = self.predict(timestamps)
        
        # Fill correlation matrices
        for (src, tgt), preds in edge_predictions.items():
            for t in range(num_timesteps):
                correlation_matrices[t, src, tgt] = preds[t]
        
        # For transfer stations, set self-connection to 1
        transfer_ids = list(range(num_nodes - self.cfg.num_transfer, num_nodes))
        for t in range(num_timesteps):
            for tid in transfer_ids:
                correlation_matrices[t, tid, tid] = 1.0
        
        return correlation_matrices

def generate_simulated_edge_histories(num_nodes, num_edges, time_steps):
    """
    Generate simulated edge histories for testing
    
    Args:
        num_nodes: Number of nodes in graph
        num_edges: Number of edges to simulate
        time_steps: Number of time steps to simulate
        
    Returns:
        Dictionary of simulated edge histories
    """
    edge_histories = {}
    
    # Create random edges
    edges = []
    for _ in range(num_edges):
        src = np.random.randint(0, num_nodes)
        tgt = np.random.randint(0, num_nodes)
        # Avoid self-loops except for transfer stations
        while src == tgt and tgt < num_nodes - 4:
            tgt = np.random.randint(0, num_nodes)
        edges.append((src, tgt))
    
    # Generate time series for each edge
    for edge in edges:
        # Base pattern (daily seasonality)
        base = np.sin(np.linspace(0, 4 * np.pi, time_steps)) * 0.4 + 0.5
        
        # Add weekly seasonality
        weekly = np.sin(np.linspace(0, 2 * np.pi, time_steps // 7)) * 0.2
        weekly = np.tile(weekly, 7)[:time_steps]
        
        # Add random noise
        noise = np.random.normal(0, 0.05, time_steps)
        
        # Combine components
        series = base + weekly + noise
        edge_histories[edge] = series.clip(0, 1)
    
    return edge_histories

if __name__ == "__main__":
    # Test the ProphetWrapper with simulated data
    cfg = Config()
    
    # Generate simulated edge histories
    num_time_steps = 1000
    edge_histories = generate_simulated_edge_histories(
        cfg.num_nodes, 
        num_edges=200,
        time_steps=num_time_steps
    )
    
    # Initialize and train wrapper
    wrapper = ProphetWrapper()
    wrapper.fit(edge_histories)
    
    # Create timestamps for prediction
    last_date = pd.Timestamp('2023-01-01') + pd.Timedelta(minutes=15*num_time_steps)
    timestamps = pd.date_range(start='2023-01-01', end=last_date, freq='15T')
    
    # Predict next 24 hours (96 time steps)
    future_timestamps = pd.date_range(
        start=timestamps[-1] + pd.Timedelta(minutes=15),
        periods=96,
        freq='15T'
    )
    
    # Generate predictions
    predictions = wrapper.predict(future_timestamps)
    
    print("ProphetWrapper test completed successfully!")
    print(f"Trained models: {len(wrapper.models)}")
    print(f"Predictions generated for {len(future_timestamps)} timestamps")