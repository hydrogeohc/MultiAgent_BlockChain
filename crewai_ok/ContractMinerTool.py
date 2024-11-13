import pandas as pd
import numpy as np

class ContractMinerTool:
    name = "ContractMinerTool"
    def __init__(self, data):
        self.data = data

    def mine_contracts(self, fraud_percentage=0.2):
        fraud_contracts = self.data[self.data['FLAG'] == 1]
        normal_contracts = self.data[self.data['FLAG'] == 0]

        fraud_sample_size = int(len(self.data) * fraud_percentage)
        normal_sample_size = len(self.data) - fraud_sample_size

        fraud_sample = fraud_contracts.sample(n=fraud_sample_size, replace=True)
        normal_sample = normal_contracts.sample(n=normal_sample_size, replace=True)

        balanced_dataset = pd.concat([fraud_sample, normal_sample]).sample(frac=1).reset_index(drop=True)
        return balanced_dataset