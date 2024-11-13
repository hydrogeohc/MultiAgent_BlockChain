from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
import os
import pandas as pd
import litellm

# Load environment variables
load_dotenv()

# Verify API key
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    raise ValueError("API key not found. Ensure that 'ANTHROPIC_API_KEY' is set in the environment.")

# Set verbose logging for detailed debugging output
os.environ['LITELLM_LOG'] = 'DEBUG'

# Set LiteLLM to modify parameters automatically for Anthropic's requirements
litellm.modify_params = True

# Define a function to make LLM calls directly
def call_llm(prompt):
    try:
        response = litellm.completion(
            model='claude-3-haiku-20240307',
            prompt=prompt,
            anthropic_api_key=anthropic_api_key,
            api_base='https://api.anthropic.com'  # Use if needed
        )
        return response
    except litellm.BadRequestError as e:
        print("Error during LLM call:", str(e))
        return None

# Test the function to verify setup
test_response = call_llm("Test prompt to check connection")
if test_response:
    print("LLM Response:", test_response)
else:
    print("LLM call failed.")

# Define the callable LLM for agents
llm_instance = call_llm

# Load the dataset
data = pd.read_csv('address_data_features_combined.csv')

# Define the ContractMiner Agent with LLM instance
contract_miner = Agent(
    name="ContractMiner",
    role="Blockchain Data Extractor",
    goal="Mine fraud and normal contracts from blockchain explorers and create balanced datasets",
    backstory="An expert in blockchain data extraction and dataset creation",
    tools=[],  # Add appropriate tools here
    llm=llm_instance,
    verbose=True
)

# Define multiple Investigative Agents with different detection algorithms
investigative_agent1 = Agent(
    name="InvestigativeAgent1",
    role="Fraud Detection Specialist A",
    goal="Detect fraud contracts accurately using Algorithm A",
    backstory="A fraud detection specialist using advanced machine learning techniques",
    tools=[],  # Add appropriate tools here
    llm=llm_instance,
    verbose=True
)

investigative_agent2 = Agent(
    name="InvestigativeAgent2",
    role="Fraud Detection Specialist B",
    goal="Detect fraud contracts accurately using Algorithm B",
    backstory="A fraud detection specialist using statistical analysis",
    tools=[],  # Add appropriate tools here
    llm=llm_instance,
    verbose=True
)

# Define the Ethics Agent
ethics_agent = Agent(
    name="EthicsAgent",
    role="AI Ethics Expert",
    goal="Ensure fair and unbiased fraud detection",
    backstory="An AI ethics expert overseeing the fraud detection process",
    tools=[],  # Add appropriate tools here
    llm=llm_instance,
    verbose=True
)

# Define the PerformanceMonitor Agent
performance_monitor = Agent(
    name="PerformanceMonitor",
    role="Performance Optimization Specialist",
    goal="Monitor and improve the performance of Investigative Agents",
    backstory="A data analyst specializing in performance optimization",
    tools=[],  # Add appropriate tools here
    llm=llm_instance,
    verbose=True
)

# Define tasks for each agent
task1 = Task(
    description="Mine contracts and create a dataset with 20% fraud contracts",
    agent=contract_miner,
    expected_output="A balanced dataset containing both fraud and normal contracts"
)

task2 = Task(
    description="Detect fraud contracts in the dataset using Algorithm A",
    agent=investigative_agent1,
    expected_output="A list of potentially fraudulent contracts identified by Algorithm A"
)

task3 = Task(
    description="Detect fraud contracts in the dataset using Algorithm B",
    agent=investigative_agent2,
    expected_output="A list of potentially fraudulent contracts identified by Algorithm B"
)

task4 = Task(
    description="Evaluate the fairness and bias of the fraud detection results",
    agent=ethics_agent,
    expected_output="An assessment report on the fairness and potential biases in the fraud detection process"
)

task5 = Task(
    description="Monitor the performance of Investigative Agents and provide feedback",
    agent=performance_monitor,
    expected_output="A performance report and recommendations for improving the fraud detection process"
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
