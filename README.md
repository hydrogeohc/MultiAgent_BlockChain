# Blockchain Fraud Detection System

This project implements a multi-agent system for detecting fraud in blockchain transactions using crewAI. The system utilizes various agents to mine contract data, detect potential fraud, ensure ethical considerations, and monitor performance.

## Overview
The Blockchain Fraud Detection System consists of the following key components:

1. ContractMiner Agent: Extracts contract data from blockchain explorers and creates balanced datasets.
2. Investigative Agents: Detect potential fraud using different algorithms.
3. Ethics Agent: Ensures fair and unbiased fraud detection.
4. PerformanceMonitor Agent: Tracks and analyzes the performance of Investigative Agents.

## Project Structure

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/hydrogeohc/Multiagent_blockchain.git
   ```
#### Components

#### ContractMiner Agent

 - Mines fraud and normal contracts from blockchain explorers
 - Creates balanced datasets with controllable fraud-to-normal contract ratios

#### Investigative Agents

 - Utilize various detection algorithms to identify potential fraud
 - Adapt dynamically to changing fraud-to-normal contract ratios

#### Ethics Agent

 - Oversees the fraud detection process
 - Ensures fairness and reduces bias in detection methods

#### PerformanceMonitor Agent

 - Monitors the performance of Investigative Agents
 - Provides feedback for improvements
 - Saves performance data for future analysis

#### Data

The system uses a dataset (address_data_features_combined.csv) containing blockchain transaction data, including features such as:

 - Transaction patterns
 - Balance information
 - Time-based metrics

## Contributing

Contributions are welcome! If you find any issues or have ideas for improvements, feel free to open a pull request.

## License

This project is licensed under the MIT License - see the MIT LICENSE file for details.
