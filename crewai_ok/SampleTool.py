import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from web3 import Web3
from tqdm import tqdm
import os
from crewai import tavily_tool

class SamplingTool:
    name = "SamplingTool"

    def __init__(self):
        if not os.environ.get("ETHERSCAN_API_KEY"):
            ETHERSCAN_API_KEY=os.environ["ETHERSCAN_API_KEY"] 

    def sample_by_flag_percentage(self, data, flag_column, flag_percentage, n_samples):
        if not set(data[flag_column].unique()).issubset({0, 1}):
            raise ValueError(f"The {flag_column} column should contain only 0s and 1s.")

        flag_1_samples = int(flag_percentage * n_samples)
        flag_0_samples = n_samples - flag_1_samples

        data_flag_1 = data[data[flag_column] == 1].sample(flag_1_samples, replace=True)
        data_flag_0 = data[data[flag_column] == 0].sample(flag_0_samples, replace=True)

        sampled_data = pd.concat([data_flag_1, data_flag_0]).sample(frac=1).reset_index(drop=True)
        return sampled_data

    def formQueryString(self, address, pgNo, offset):
        return f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&page={pgNo}&offset={offset}&sort=asc&apikey={os.environ.get('ETHERSCAN_API_KEY')}"

    def get_address_stats_normal_tnx(self, address, hacked):
        response = requests.get(self.formQueryString(address, 1, 0))
        sample_df = pd.DataFrame(response.json()['result'])
        
        if sample_df.empty:
            return self.get_empty_details_for_address(address, hacked)

        sample_df['eth value'] = sample_df['value'].apply(lambda x: Web3.from_wei(int(x), 'ether'))
        sample_df['txn type'] = sample_df['from'].apply(lambda x: 'sent' if x == address else 'received')

        sample_df_sent = sample_df[sample_df['txn type'] == 'sent']
        sample_df_received = sample_df[sample_df['txn type'] == 'received']
        sample_df_sent_contracts = sample_df[sample_df['contractAddress'] != '']

        compiled_normal_tnx_result = {
            'Address': address, 'FLAG': hacked,
            'Sent tnx': len(sample_df_sent),
            'Received Tnx': len(sample_df_received),
            'Number of Created Contracts': len(sample_df_sent_contracts),
            'Unique Received From Addresses': len(sample_df_received['from'].unique()),
            'Unique Sent To Addresses': len(sample_df_sent['to'].unique()),
            'min value received': sample_df_received['eth value'].min(),
            'max value received': sample_df_received['eth value'].max(),
            'avg val received': sample_df_received['eth value'].mean(),
            'min val sent': sample_df_sent['eth value'].min(),
            'max val sent': sample_df_sent['eth value'].max(),
            'avg val sent': sample_df_sent['eth value'].mean(),
            'total transactions': len(sample_df),
            'total Ether sent': sample_df_sent['eth value'].sum(),
            'total ether received': sample_df_received['eth value'].sum(),
            'total ether balance': sample_df_received['eth value'].sum() - sample_df_sent['eth value'].sum()
        }
        return pd.DataFrame([compiled_normal_tnx_result])

    def get_empty_details_for_address(self, address, hacked):
        return pd.DataFrame([{
            'Address': address, 'FLAG': hacked,
            'Sent tnx': 0, 'Received Tnx': 0,
            'Number of Created Contracts': 0,
            'Unique Received From Addresses': 0,
            'Unique Sent To Addresses': 0,
            'min value received': 0,
            'max value received': 0,
            'avg val received': 0,
            'min val sent': 0,
            'max val sent': 0,
            'avg val sent': 0,
            'total transactions': 0,
            'total Ether sent': 0,
            'total ether received': 0,
            'total ether balance': 0
        }])

    def collect_ethereum_data(self, addresses, hacked_flag):
        base_df = pd.DataFrame()
        total_transactions = 0
        
        for i, address in enumerate(tqdm(addresses)):
            try:
                df = self.get_address_stats_normal_tnx(address, hacked_flag)
                if i == 0:
                    base_df = df
                else:
                    base_df = pd.concat([base_df, df])
                
                txns = df.loc[0, 'total transactions']
                total_transactions += txns
                print(f"Address number {i}: {address} mined! {txns} retrieved. {total_transactions} total transactions.")
            except:
                df = self.get_empty_details_for_address(address, hacked_flag)
                base_df = pd.concat([base_df, df])
                print(f"Address number {i}: {address} mined! 0 txns retrieved. {total_transactions} total transactions.")
        
        return base_df.reset_index(drop=True)

    @tavily_tool
    def ethereum_sampling_tool(self, hacked_addresses, non_hacked_addresses, flag_percentage, n_samples):
        """
        Collect Ethereum address data and generate a sample based on specified parameters.
        """
        hacked_data = self.collect_ethereum_data(hacked_addresses, 1)
        non_hacked_data = self.collect_ethereum_data(non_hacked_addresses, 0)
        
        combined_data = pd.concat([hacked_data, non_hacked_data], axis=0, ignore_index=True)
        sampled_data = self.sample_by_flag_percentage(combined_data, 'FLAG', flag_percentage, n_samples)
        
        return sampled_data