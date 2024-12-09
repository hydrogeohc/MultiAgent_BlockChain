import pandas as pd
import numpy as np

class ContractMinerTool:
    name = "ContractMinerTool"

    def __init__(self, data):
        self.data = data

    def mine_contracts(self, fraud_percentage_range=(0, 1), step=0.1):
        """
        Mine contracts from the dataset with a specified range of fraudulent contract percentages.

        Args:
            fraud_percentage_range (tuple): A tuple containing the start and end percentages (0 to 1).
            step (float): The step size between percentages.

        Returns:
            dict: A dictionary of balanced datasets, keyed by fraud percentage.
        """
        fraud_contracts = self.data[self.data['FLAG'] == 1]
        normal_contracts = self.data[self.data['FLAG'] == 0]

        balanced_datasets = {}

        for fraud_percentage in np.arange(fraud_percentage_range[0], fraud_percentage_range[1] + step, step):
            fraud_percentage = round(fraud_percentage, 2)
            total_sample_size = len(self.data)
            fraud_sample_size = int(total_sample_size * fraud_percentage)
            normal_sample_size = total_sample_size - fraud_sample_size

            # Ensure we don't sample more than available
            fraud_sample_size = min(fraud_sample_size, len(fraud_contracts))
            normal_sample_size = min(normal_sample_size, len(normal_contracts))

            fraud_sample = fraud_contracts.sample(n=fraud_sample_size, replace=True)
            normal_sample = normal_contracts.sample(n=normal_sample_size, replace=True)

            balanced_dataset = pd.concat([fraud_sample, normal_sample]).sample(frac=1).reset_index(drop=True)
            balanced_datasets[fraud_percentage] = balanced_dataset

        return balanced_datasets