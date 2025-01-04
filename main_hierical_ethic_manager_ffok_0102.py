import sys
import json
import os
import pandas as pd
#from datasets import load_dataset
from crewai import Agent, Task, Crew, Process
# from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Any
from crewai.tools import BaseTool
from dotenv import load_dotenv

# Append current dir for local imports (if needed)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ------------------------------------------------------------------------------
# 1. Load the Forta Malicious Smart Contract Dataset (up to 100M samples)
# ------------------------------------------------------------------------------

# Load and preprocess the data
data = pd.read_csv('./forta_hacked_address_features.csv')

# ------------------------------------------------------------------------------
# 2. Initialize Your LLM
# ------------------------------------------------------------------------------

load_dotenv()  # Ensure .env is loaded if using OpenAI keys
chat_model = ChatOpenAI(
    model_name="gpt-4o-2024-08-06",  # Custom or future model name
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.1
)

# ------------------------------------------------------------------------------
# 3. Define Tools
# ------------------------------------------------------------------------------

class ContractMinerTool(BaseTool):
    name: str = "ContractMinerTool"
    description: str = "Mine contracts and create balanced datasets"
    data: Any = Field(default=None, exclude=True)

    def __init__(self, data):
        super().__init__()
        self.data = data

    def _run(self, num_contracts: int, fraud_percentage: float) -> str:
        return f"Mined {num_contracts} contracts with {fraud_percentage}% fraud contracts"

class FraudDetectionTool(BaseTool):
    name: str = "FraudDetectionTool"
    description: str = "Detect fraud contracts using specified algorithm"
    algorithm: str = Field(..., description="The algorithm to use for fraud detection")
    data: Any = Field(default=None, exclude=True)

    def __init__(self, algorithm: str, data):
        super().__init__(algorithm=algorithm)
        self.data = data

    def _run(self) -> str:
        """
        Revised to use 'FLAG' as the label column.
        FLAG:
         - 1 indicates malicious/fraud
         - 0 indicates normal
        """
        if "FLAG" in self.data.columns:
            malicious_count = (self.data["FLAG"] == 1).sum()
            normal_count = (self.data["FLAG"] == 0).sum()
            return (
                f"Detected fraud contracts using algorithm {self.algorithm}. "
                f"Malicious count = {malicious_count}, Normal count = {normal_count}"
            )
        else:
            return (
                f"Detected fraud contracts using algorithm {self.algorithm} - "
                "No 'FLAG' info found in dataset."
            )

class EthicsCheckerTool(BaseTool):
    name: str = "EthicsCheckerTool"
    description: str = "Evaluate fairness and bias of fraud detection results"
    data: Any = Field(default=None)

    def __init__(self, data: Any = None, **kwargs):
        super().__init__(**kwargs)
        self.data = data

    def _run(self, results: str) -> str:
        return f"Evaluated fairness and bias of results: {results}"

class PerformanceMonitorTool(BaseTool):
    name: str = "PerformanceMonitorTool"
    description: str = "Monitor and improve performance of Investigative Agents"

    def _run(self, agent_results: str) -> str:
        return f"Monitored performance of agents based on results: {agent_results}"

# ------------------------------------------------------------------------------
# 4. Define Agents
# ------------------------------------------------------------------------------

contract_miner = Agent(
    name="ContractMiner",
    role="Contract Mining Specialist",
    goal="Mine fraud and normal contracts from blockchain explorers and create balanced datasets",
    backstory="An expert in blockchain data extraction and dataset creation",
    tools=[ContractMinerTool(data)],
    llm=chat_model,
    verbose=True
)

investigative_agent1 = Agent(
    name="InvestigativeAgent1",
    role="Fraud Detection Specialist A",
    goal="Detect fraud contracts accurately using Algorithm A",
    backstory="A fraud detection specialist using advanced machine learning techniques",
    tools=[FraudDetectionTool(algorithm="A", data=data)],
    llm=chat_model,
    verbose=True
)

investigative_agent2 = Agent(
    name="InvestigativeAgent2",
    role="Fraud Detection Specialist B",
    goal="Detect fraud contracts accurately using Algorithm B",
    backstory="A fraud detection specialist using statistical analysis",
    tools=[FraudDetectionTool(algorithm="B", data=data)],
    llm=chat_model,
    verbose=True
)

investigative_agent3 = Agent(
    name="InvestigativeAgent3",
    role="Fraud Detection Specialist C",
    goal="Detect fraud contracts accurately using Algorithm C",
    backstory="A fraud detection specialist using statistical analysis",
    tools=[FraudDetectionTool(algorithm="C", data=data)],
    llm=chat_model,
    verbose=True
)

cleaned_data = data.copy()
ethics_agent = Agent(
    name="EthicsAgent",
    role="AI Ethics Expert",
    goal="Ensure fair and unbiased fraud detection",
    backstory="An AI ethics expert overseeing the fraud detection process",
    tools=[EthicsCheckerTool(data=cleaned_data)],
    llm=chat_model,
    verbose=True
)

performance_monitor = Agent(
    name="PerformanceMonitor",
    role="Performance Analyst",
    goal="Monitor and improve the performance of Investigative Agents",
    backstory="A data analyst specializing in performance optimization",
    tools=[PerformanceMonitorTool()],
    llm=chat_model,
    verbose=True
)

# ------------------------------------------------------------------------------
# 5. Define Tasks
# ------------------------------------------------------------------------------

task1 = Task(
    description="Mine contracts and create a dataset with fraud contracts",
    agent=contract_miner,
    expected_output="A dataset containing mined contracts with fraud contracts"
)

task2 = Task(
    description="Detect fraud contracts in the dataset using Algorithm A",
    agent=investigative_agent1,
    expected_output="A list of detected fraud contracts using Algorithm A"
)

task3 = Task(
    description="Detect fraud contracts in the dataset using Algorithm B",
    agent=investigative_agent2,
    expected_output="A list of detected fraud contracts using Algorithm B"
)

task4 = Task(
    description="Detect fraud contracts in the dataset using Algorithm C",
    agent=investigative_agent3,
    expected_output="A list of detected fraud contracts using Algorithm C"
)

results_data = {
    "predictions": [1, 0, 1, 1, 0],  # Example predictions
    "feature_importances": [0.2, 0.3, 0.5, 0.1, 0.4]  # Example feature importances
}
results_json = json.dumps(results_data)

task5 = Task(
    description="Evaluate the fairness and bias of the fraud detection results",
    agent=ethics_agent,
    expected_output="An evaluation report on the fairness and bias of the fraud detection results",
    context=[
        {
            "description": "Provide predictions and feature importances for fairness evaluation",
            "expected_output": "Fairness evaluation results based on the input predictions",
            "results": results_json
        }
    ]
)

task6 = Task(
    description="Monitor the performance of Investigative Agents and provide feedback",
    agent=performance_monitor,
    expected_output="A performance report and feedback on the Investigative Agents",
    context=[
        {
            "description": "Performance data on agents' contract processing",
            "expected_output": "Performance analysis results and actionable feedback",
            "data": {
                "agent_results": [
                    {"agent": "InvestigativeAgent1", "identified_contracts": [2, 5]},
                    {"agent": "InvestigativeAgent2", "identified_contracts": [3, 7]},
                    {"agent": "InvestigativeAgent3", "identified_contracts": [10]}
                ]
            }
        }
    ]
)

# ------------------------------------------------------------------------------
# 6. Manager Agent & Crew Setup
# ------------------------------------------------------------------------------

manager_agent = Agent(
    name="ManagerAgent",
    role="Manager",
    goal="Oversee and manage the hierarchical process",
    backstory="An AI agent solely responsible for managing the workflow",
    tools=[],  # No tools for the manager agent
    llm=chat_model,
    verbose=True
)

crew = Crew(
    agents=[
        contract_miner,
        investigative_agent1,
        investigative_agent2,
        investigative_agent3,
        ethics_agent,
        performance_monitor
    ],
    tasks=[task1, task2, task3, task4, task5, task6],
    process=Process.hierarchical,
    manager_llm=chat_model,
    respect_context_window=True,
    memory=True,
    manager_agent=manager_agent,
    planning=True
)

# ------------------------------------------------------------------------------
# 7. Kick Off the Process
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    result = crew.kickoff()
    print(result)
