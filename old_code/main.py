import sys
import os
import pandas as pd
from crewai import Agent, Task, Crew, Process
from llama_index.llms.openrouter import OpenRouter
from pydantic import BaseModel, Field
from typing import Optional
from crewai.tools import BaseTool
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crewai_ok import ContractMinerTool, FraudDetectionTool, PerformanceMonitorTool, EthicsCheckerTool
#import openrouter
#from openrouter import OpenRouter
#from langchain.llms import OpenRouterLLM
#from tools import ContractMinerTool, FraudDetectionTool, PerformanceMonitorTool, EthicsCheckerTool
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load and preprocess the data
data = pd.read_csv('./address_data_features_combined.csv')

# Initialize OpenRouter client
""" openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("Please set the OPENROUTER_API_KEY environment variable")

client = OpenRouter(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://your-app-url.com",  # Replace with your app's URL
        "X-Title": "Blockchain Fraud Detection"  # Replace with your app's name
    }
) """
# Initialize OpenRouter LLM
llm = OpenRouter(
    api_key="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",
    max_tokens=256,
    context_window=4096,
    model="meta-llama/llama-3.1-8b-instruct",
)

""" llm = OpenRouterLLM(
    model="openai/gpt-3.5-turbo",  # You can change this to any model supported by OpenRouter
    openrouter_api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    headers={
        "HTTP-Referer": "https://your-app-url.com",  # Replace with your app's URL
        "X-Title": "Blockchain Fraud Detection"  # Replace with your app's name
    }
) """

# Define tool schemas
class ContractMinerToolSchema(BaseModel):
    num_contracts: int = Field(..., description="Number of contracts to mine")
    fraud_percentage: float = Field(..., description="Percentage of fraud contracts to include")

class FraudDetectionToolSchema(BaseModel):
    algorithm: str = Field(..., description="Algorithm to use for fraud detection")

class EthicsCheckerToolSchema(BaseModel):
    results: str = Field(..., description="Fraud detection results to evaluate")

class PerformanceMonitorToolSchema(BaseModel):
    agent_results: str = Field(..., description="Results from investigative agents to monitor")

# Define tools
class ContractMinerTool(BaseTool):
    name: str = "ContractMinerTool"
    description: str = "Mine contracts and create balanced datasets"
    args_schema: type[BaseModel] = ContractMinerToolSchema

    def __init__(self, data):
        super().__init__()
        self.data = data

    def _run(self, num_contracts: int, fraud_percentage: float) -> str:
        # Implement contract mining logic here
        return f"Mined {num_contracts} contracts with {fraud_percentage}% fraud contracts"

class FraudDetectionTool(BaseTool):
    name: str = "FraudDetectionTool"
    description: str = "Detect fraud contracts using specified algorithm"
    args_schema: type[BaseModel] = FraudDetectionToolSchema

    def __init__(self, data):
        super().__init__()
        self.data = data

    def _run(self, algorithm: str) -> str:
        # Implement fraud detection logic here
        return f"Detected fraud contracts using algorithm {algorithm}"

class EthicsCheckerTool(BaseTool):
    name: str = "EthicsCheckerTool"
    description: str = "Evaluate fairness and bias of fraud detection results"
    args_schema: type[BaseModel] = EthicsCheckerToolSchema

    def __init__(self, data):
        super().__init__()
        self.data = data

    def _run(self, results: str) -> str:
        # Implement ethics checking logic here
        return f"Evaluated fairness and bias of results: {results}"

class PerformanceMonitorTool(BaseTool):
    name: str = "PerformanceMonitorTool"
    description: str = "Monitor and improve performance of Investigative Agents"
    args_schema: type[BaseModel] = PerformanceMonitorToolSchema

    def _run(self, agent_results: str) -> str:
        # Implement performance monitoring logic here
        return f"Monitored performance of agents based on results: {agent_results}"

# Initialize tools
contract_miner_tool = ContractMinerTool(data)
fraud_detection_tool_a = FraudDetectionTool(data)
fraud_detection_tool_b = FraudDetectionTool(data)
ethics_checker_tool = EthicsCheckerTool(data)
performance_monitor_tool = PerformanceMonitorTool()

# Define agents
contract_miner = Agent(
    name="ContractMiner",
    role="Contract Mining Specialist",
    goal="Mine fraud and normal contracts from blockchain explorers and create balanced datasets",
    backstory="An expert in blockchain data extraction and dataset creation",
    tools=[contract_miner_tool],
    llm=llm,
    verbose=True
)

investigative_agent1 = Agent(
    name="InvestigativeAgent1",
    goal="Detect fraud contracts accurately using Algorithm A",
    backstory="A fraud detection specialist using advanced machine learning techniques",
    tools=[fraud_detection_tool_a],
    llm=llm,
    verbose=True
)

investigative_agent2 = Agent(
    name="InvestigativeAgent2",
    goal="Detect fraud contracts accurately using Algorithm B",
    backstory="A fraud detection specialist using statistical analysis",
    tools=[fraud_detection_tool_b],
    llm=llm,
    verbose=True
)

ethics_agent = Agent(
    name="EthicsAgent",
    goal="Ensure fair and unbiased fraud detection",
    backstory="An AI ethics expert overseeing the fraud detection process",
    tools=[ethics_checker_tool],
    llm=llm,
    verbose=True
)

performance_monitor = Agent(
    name="PerformanceMonitor",
    goal="Monitor and improve the performance of Investigative Agents",
    backstory="A data analyst specializing in performance optimization",
    tools=[performance_monitor_tool],
    llm=llm,
    verbose=True
)

# Define tasks
task1 = Task(
    description="Mine contracts and create a dataset with 20% fraud contracts",
    agent=contract_miner
)

task2 = Task(
    description="Detect fraud contracts in the dataset using Algorithm A",
    agent=investigative_agent1
)

task3 = Task(
    description="Detect fraud contracts in the dataset using Algorithm B",
    agent=investigative_agent2
)

task4 = Task(
    description="Evaluate the fairness and bias of the fraud detection results",
    agent=ethics_agent
)

task5 = Task(
    description="Monitor the performance of Investigative Agents and provide feedback",
    agent=performance_monitor
)

# Create the crew
crew = Crew(
    agents=[contract_miner, investigative_agent1, investigative_agent2, ethics_agent, performance_monitor],
    tasks=[task1, task2, task3, task4, task5],
    process=Process.sequential
)

# Start the crew's work
result = crew.kickoff()
print(result)