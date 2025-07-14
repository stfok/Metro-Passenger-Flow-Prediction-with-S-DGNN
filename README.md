Metro Passenger Flow Prediction with S-DGNN
Project Overview
This project implements the Sequence Directed Graph Neural Network (S-DGNN) model for predicting passenger flow at metro transfer stations, based on the research paper:

Prediction and Warning Method for Large Passenger Flow in Metro Transfer Stations Based on Spatial and Temporal Characteristics of Personnel Trajectories
Dawei Cui, Zewei Zhang, Tongfeng Sun
Chinese Institute of Coal Science, China University of Mining and Technology

The model leverages spatial-temporal characteristics of passenger trajectories to predict transfer station flows with over 90% accuracy, enabling early warning systems for large passenger flow risks.

Key Features
🚇 Transfer-Centric Topology: Novel graph structure focusing on transfer stations

⏳ Time Delay Modeling: Accounts for travel time between stations

📈 S-DGNN Architecture: Combines Directed Graph Neural Networks with Gated Recurrent Units

📊 High Accuracy: Achieves >90% prediction accuracy on transfer station flows

🏙️ Scenario-Specific Models: Separate training for weekday and weekend patterns

⚡ Real-time Capable: Inference in 0.12 seconds per prediction

Model Architecture
https://docs/s_dgnn_architecture.png

The S-DGNN model consists of two main components:

Spatial Module (DGNN):

Processes metro network as directed graph

Incorporates time delay calculations

Uses transfer-centric topology

Edge attributes: correlation weights, distances, departure intervals

Temporal Module (GRU):

Captures time series dependencies

Processes sequential passenger flow data

100 hidden units for optimal performance

Getting Started
Prerequisites
Python 3.8+

PyTorch 2.0+

pandas, numpy, matplotlib

scikit-learn

fbprophet (for correlation prediction)




Train the S-DGNN model:


python train.py
Evaluation
Test model performance:


python test.py
Visualization
Generate topology and flow pattern visualizations:


python visualize.py
Results
Performance Metrics
Dataset	MAE	RMSE	MAPE (%)	Accuracy
Weekday	11.86	19.13	22.38	92.76%
Weekend	13.02	22.45	25.17	86.59%
Extreme	14.87	25.31	29.84	83.84%

Key Implementation Details
Metro Topology
84 general stations + 4 transfer stations

4 metro lines with interconnections:

Line 1 ↔ Line 2 at T1

Line 2 ↔ Line 3 at T2

Line 3 ↔ Line 4 at T3

Line 4 ↔ Line 1 at T4

Graph Structure
python
# Transfer-centric topology connections
adj[general_i][transfer_j] = 1.0  # If same line
adj[transfer_i][transfer_j] = 1.0 # If same line (bidirectional)
Time Delay Calculation
python
time_delay = distances / (metro_speed / 60) + departure_interval / 2
Model Input/Output
Input: Sequence of station entry flows (4 × 15-min intervals)

Output: Predicted passenger flow at 4 transfer stations (next 15-min)

Directory Structure
text
metro-flow-prediction/
├── data/                   # Data files
│   ├── metro_network.json  # Metro topology
│   ├── afc_data.csv        # AFC data
│   └── processed/          # Preprocessed data
├── model/                  # Model implementations
│   ├── dgnn.py             # Directed Graph Neural Network
│   ├── s_dgnn.py           # S-DGNN main model
│   └── prophet_wrapper.py  # Prophet correlation prediction
├── utils/                  # Utility modules
│   ├── graph_builder.py    # Graph construction
│   ├── data_loader.py      # Data loading
│   └── metrics.py          # Evaluation metrics
├── config.py               # Configuration
├── train.py                # Model training
├── test.py                 # Model testing
└── visualize.py            # Visualization
References
Cui D., Zhang Z., Sun T. (2024). Prediction and Warning Method for Large Passenger Flow in Metro Transfer Stations Based on Spatial and Temporal Characteristics of Personnel Trajectories

Li Y., et al. (2017). Diffusion Convolutional Recurrent Neural Network: Data-driven Traffic Forecasting. arXiv:1707.01926

Guo S., et al. (2019). Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting. AAAI

Zhou R., et al. (2021). Research on Traffic Situation Analysis for Urban Road Network Through Spatiotemporal Data Mining. IEEE Access

License
This project is licensed under the MIT License - see the LICENSE file for details.



