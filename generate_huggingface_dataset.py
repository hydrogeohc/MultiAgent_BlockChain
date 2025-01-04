import pandas as pd
from datasets import load_dataset
from web3 import Web3

# Step 1: Load the Forta Dataset
print("Loading the Forta dataset...")
forta_dataset = load_dataset("forta/malicious-smart-contract-dataset", split="train")
forta_df = forta_dataset.to_pandas()

# Step 2: Add a `FLAG` Column to Label Hacked Addresses
# Use 'malicious' column as an indicator (1 = hacked, 0 = normal)
if 'malicious' in forta_df.columns:
    forta_df['FLAG'] = forta_df['malicious'].apply(lambda x: 1 if x else 0)
else:
    raise KeyError("The Forta dataset does not contain a `malicious` column.")

# Step 3: Define Functions for Transaction-Based Features

def calculate_transaction_features(tx_df, address):
    """
    Calculate transaction-based statistics for an Ethereum address.
    """
    tx_df['eth_value'] = tx_df['value'].apply(lambda x: Web3.from_wei(int(x), 'ether'))
    tx_df['txn_type'] = tx_df['from'].apply(lambda x: 'sent' if x == address else 'received')

    # Sent Transactions
    sent_txs = tx_df[tx_df['txn_type'] == 'sent']
    received_txs = tx_df[tx_df['txn_type'] == 'received']

    stats = {
        'Sent Transactions': len(sent_txs),
        'Received Transactions': len(received_txs),
        'Unique Sent To Addresses': sent_txs['to'].nunique(),
        'Unique Received From Addresses': received_txs['from'].nunique(),
        'Total Ether Sent': sent_txs['eth_value'].sum(),
        'Total Ether Received': received_txs['eth_value'].sum(),
        'Ether Balance': received_txs['eth_value'].sum() - sent_txs['eth_value'].sum(),
    }

    return stats

# Step 4: Apply Feature Extraction to Each Address
print("Calculating transaction features for each address...")
feature_list = []
for index, row in forta_df.iterrows():
    address = row['contract_address']
    # Assume `transactions` is a placeholder for transaction data (mock logic)
    tx_data = pd.DataFrame([])  # Replace with actual transaction data if available
    if not tx_data.empty:
        features = calculate_transaction_features(tx_data, address)
    else:
        features = {key: 0 for key in [
            'Sent Transactions', 'Received Transactions',
            'Unique Sent To Addresses', 'Unique Received From Addresses',
            'Total Ether Sent', 'Total Ether Received', 'Ether Balance'
        ]}
    features['Address'] = address
    features['FLAG'] = row['FLAG']
    feature_list.append(features)

# Convert to DataFrame
final_df = pd.DataFrame(feature_list)

# Step 5: Save the Resulting Dataset
output_path = "forta_hacked_address_features.csv"
final_df.to_csv(output_path, index=False)
print(f"Processed dataset saved to {output_path}")
