import logging
from typing import Any, Dict
from crewai import Agent, LLM # Import LLM here for type hinting
from .tools import ProductionContractMinerTool, ProductionFraudDetectionTool, ProductionEthicsCheckerTool, ProductionPerformanceMonitorTool # Import the production tools

# Get a logger for this module
logger = logging.getLogger(__name__)

def create_agents(tools: Dict[str, BaseTool], llm: LLM) -> Dict[str, Agent]:
    """
    Creates and returns a dictionary of CrewAI Agents.

    Args:
        tools (Dict[str, BaseTool]): A dictionary of initialized tool instances.
        llm (LLM): The initialized CrewAI LLM instance.

    Returns:
        Dict[str, Agent]: A dictionary of initialized Agent instances, keyed by a logical name.
    """
    logger.info("Creating CrewAI Agents.")
    if llm is None:
        logger.critical("LLM is None, cannot create agents.")
        raise ValueError("LLM must be initialized to create agents.")
    if not tools:
         logger.warning("No tools provided to create agents that require tools.")

    agents = {}

    # Ensure required tools are present
    miner_tool = tools.get("ContractMinerTool")
    if not miner_tool: logger.error("ContractMinerTool not found in provided tools.")

    detection_tool_a = tools.get("FraudDetectionTool_A")
    if not detection_tool_a: logger.error("FraudDetectionTool_A not found in provided tools.")
    detection_tool_b = tools.get("FraudDetectionTool_B")
    if not detection_tool_b: logger.error("FraudDetectionTool_B not found in provided tools.")
    detection_tool_c = tools.get("FraudDetectionTool_C")
    if not detection_tool_c: logger.error("FraudDetectionTool_C not found in provided tools.")

    ethics_tool = tools.get("EthicsCheckerTool")
    if not ethics_tool: logger.error("EthicsCheckerTool not found in provided tools.")

    performance_tool = tools.get("PerformanceMonitorTool")
    if not performance_tool: logger.error("PerformanceMonitorTool not found in provided tools.")


    # --- Define Agents ---

    agents["contract_miner"] = Agent(
        name="ContractMiningSpecialist",
        role="Contract Mining Specialist",
        goal="Mine and prepare balanced contract datasets with specified fraud percentages, focusing on creating datasets suitable for training and evaluation of fraud detection models.",
        backstory="An expert in blockchain data extraction and dataset creation, with a focus on generating representative and balanced datasets for analysis.",
        tools=[miner_tool] if miner_tool else [], # Assign tool if available
        llm=llm,
        verbose=True,
        # Add memory and other production-relevant params if needed
        # memory=True
    )

    agents["investigative_agent_a"] = Agent(
        name="FraudDetectionSpecialistA",
        role="Fraud Detection Specialist (Algorithm A)",
        goal="Apply Algorithm A to the provided contract dataset to accurately detect fraud contracts and report the results, including predictions and feature importances, in a structured format.",
        backstory="A fraud detection specialist skilled in using advanced machine learning algorithm A to identify malicious contract activity.",
        tools=[detection_tool_a] if detection_tool_a else [],
        llm=llm,
        verbose=True,
        # memory=True
    )

    agents["investigative_agent_b"] = Agent(
        name="FraudDetectionSpecialistB",
        role="Fraud Detection Specialist (Algorithm B)",
        goal="Apply Algorithm B to the provided contract dataset to accurately detect fraud contracts and report the results, including predictions and feature importances, in a structured format.",
        backstory="A fraud detection specialist skilled in using statistical algorithm B to identify malicious contract activity.",
        tools=[detection_tool_b] if detection_tool_b else [],
        llm=llm,
        verbose=True,
        # memory=True
    )

    agents["investigative_agent_c"] = Agent(
        name="FraudDetectionSpecialistC",
        role="Fraud Detection Specialist (Algorithm C)",
        goal="Apply Algorithm C to the provided contract dataset to accurately detect fraud contracts and report the results, including predictions and feature importances, in a structured format.",
        backstory="A fraud detection specialist skilled in using algorithm C to identify malicious contract activity.",
        tools=[detection_tool_c] if detection_tool_c else [],
        llm=llm,
        verbose=True,
        # memory=True
    )

    agents["ethics_agent"] = Agent(
        name="AIEthicsExpert",
        role="AI Ethics Expert",
        goal="Evaluate the fairness, bias, and performance of fraud detection results from multiple algorithms and generate a comprehensive, structured report.",
        backstory="An AI ethics expert with deep understanding of fairness metrics, bias detection, and model performance evaluation, dedicated to ensuring responsible AI deployment.",
        tools=[ethics_tool] if ethics_tool else [],
        llm=llm,
        verbose=True,
        # memory=True
    )

    agents["performance_monitor"] = Agent(
        name="PerformanceAnalyst",
        role="Performance Analyst",
        goal="Monitor and analyze the performance metrics and evaluation results from fraud detection and ethics evaluation tasks to provide insights and actionable feedback for workflow improvement.",
        backstory="A data analyst specializing in evaluating and improving the efficiency and effectiveness of automated workflows and agent performance.",
        tools=[performance_tool] if performance_tool else [],
        llm=llm,
        verbose=True,
        # memory=True
    )

    # Manager Agent (needed for hierarchical process)
    agents["manager"] = Agent(
        name="WorkflowManager",
        role="Workflow Manager",
        goal="Efficiently manage and coordinate the hierarchical workflow for contract fraud detection and ethics evaluation, ensuring tasks are completed in order and specialists collaborate effectively.",
        backstory="An AI agent responsible for overseeing the entire fraud detection and ethics evaluation process, coordinating specialist agents and reporting on overall progress.",
        tools=[],  # Manager agent typically doesn't use tools directly
        llm=llm, # Manager needs an LLM
        verbose=True,
        # memory=True
    )

    logger.info(f"Created {len(agents)} agents.")
    return agents