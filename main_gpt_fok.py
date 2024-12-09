import sys
import os
import pandas as pd
import random
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Any
from crewai_ok.SampleTool import SampleTool
from crewai.tools import BaseTool
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

load_dotenv()

# Initialize ChatOpenAI
chat_model = ChatOpenAI(
    model_name="gpt-4",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.1
)

class SampleTool(BaseTool):
    name: str = "SampleTool"
    description: str = "Sample data with specified flag percentage"

    def _run(self, hacked_addresses, non_hacked_addresses, flag_percentage, n_samples):
        return f"Sampled {n_samples} addresses with {flag_percentage:.2f}% flagged as hacked"

class SamplingAgent(Agent):
    flag_percentage_min: float = Field(default=0.1)
    flag_percentage_max: float = Field(default=0.9)
    hacked_addresses: list = Field(default_factory=list)
    non_hacked_addresses: list = Field(default_factory=list)

    def __init__(self, flag_percentage_min=0.1, flag_percentage_max=0.9, hacked_addresses=None, non_hacked_addresses=None):
        sampling_tool = SampleTool()
        super().__init__(
            name="SamplingAgent",
            role="Data Sampler",
            goal="Sample data with a specified flag percentage range",
            backstory="An expert in data sampling and preparation",
            llm=chat_model,
            verbose=True
        )

        
        self.flag_percentage_min = flag_percentage_min
        self.flag_percentage_max = flag_percentage_max
        self.hacked_addresses = hacked_addresses or []
        self.non_hacked_addresses = non_hacked_addresses or []
        self.tools = [sampling_tool]

    async def act(self, n_samples=1000):
        flag_percentage = random.uniform(self.flag_percentage_min, self.flag_percentage_max)
        return await self.use_tool(
            "SampleTool",
            hacked_addresses=self.hacked_addresses,
            non_hacked_addresses=self.non_hacked_addresses,
            flag_percentage=flag_percentage,
            n_samples=n_samples
        )

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
        return f"Detected fraud contracts using algorithm {self.algorithm}"

class EthicsCheckerTool(BaseTool):
    name: str = "EthicsCheckerTool"
    description: str = "Evaluate fairness and bias of fraud detection results"
    data: Any = Field(default=None, exclude=True)

    def __init__(self, data):
        super().__init__()
        self.data = data

    def _run(self, results: str) -> str:
        return f"Evaluated fairness and bias of results: {results}"

class PerformanceMonitorTool(BaseTool):
    name: str = "PerformanceMonitorTool"
    description: str = "Monitor and improve performance of Investigative Agents"

    def _run(self, agent_results: str) -> str:
        return f"Monitored performance of agents based on results: {agent_results}"

hacked_addresses = ["0x123...", "0x456..."]  # Add your actual hacked addresses here
non_hacked_addresses = ["0x789...", "0xabc..."]  
# Define agents
sampling_agent = SamplingAgent(
    flag_percentage_min=0.1,
    flag_percentage_max=0.9,
    hacked_addresses=hacked_addresses,  # Add your hacked addresses here
    non_hacked_addresses=non_hacked_addresses  # Add your non-hacked addresses here
)
contract_miner = Agent(
    name="ContractMiner",
    role="Contract Mining Specialist",
    goal="Mine fraud and normal contracts from blockchain explorers and create balanced datasets",
    backstory="An expert in blockchain data extraction and dataset creation",
    tools=[ContractMinerTool(None)],
    llm=chat_model,
    verbose=True
)

investigative_agent1 = Agent(
    name="InvestigativeAgent1",
    role="Fraud Detection Specialist A",
    goal="Detect fraud contracts accurately using Algorithm A",
    backstory="A fraud detection specialist using advanced machine learning techniques",
    tools=[FraudDetectionTool(algorithm="A", data=None)],
    llm=chat_model,
    verbose=True
)

investigative_agent2 = Agent(
    name="InvestigativeAgent2",
    role="Fraud Detection Specialist B",
    goal="Detect fraud contracts accurately using Algorithm B",
    backstory="A fraud detection specialist using statistical analysis",
    tools=[FraudDetectionTool(algorithm="B", data=None)],
    llm=chat_model,
    verbose=True
)

ethics_agent = Agent(
    name="EthicsAgent",
    role="AI Ethics Expert",
    goal="Ensure fair and unbiased fraud detection",
    backstory="An AI ethics expert overseeing the fraud detection process",
    tools=[EthicsCheckerTool(None)],
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

# Define tasks
task0 = Task(
    description="Sample the dataset for further analysis with 10-90% flagged data",
    agent=sampling_agent,
    expected_output="A sampled dataset for analysis"
)
task1 = Task(
    description="Mine contracts and create a dataset with 20% fraud contracts",
    agent=contract_miner,
    expected_output="A dataset containing mined contracts with 20% fraud contracts"
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
    description="Evaluate the fairness and bias of the fraud detection results",
    agent=ethics_agent,
    expected_output="An evaluation report on the fairness and bias of the fraud detection results",
    tool_input="Fraud detection results identified contract 2 and contract 5 as fraudulent."
)

task5 = Task(
    description="Monitor the performance of Investigative Agents and provide feedback",
    agent=performance_monitor,
    expected_output="A performance report and feedback on the Investigative Agents",
    tool_input="Investigative agents' performance data showing contract 2 and contract 5 as identified frauds."
)


# Create the crew
crew = Crew(
    agents=[sampling_agent, contract_miner, investigative_agent1, investigative_agent2, ethics_agent, performance_monitor],
    hacked_addresses=hacked_addresses,
    non_hacked_addresses=non_hacked_addresses,
    tasks=[task0, task1, task2, task3, task4, task5],
    process=Process.sequential
    
)

# Start the crew's work
if __name__ == "__main__":
    result = crew.kickoff()
    print(result)