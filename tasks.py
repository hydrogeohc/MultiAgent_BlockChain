import logging
from typing import Dict, Any, List
from crewai import Task, Agent # Import Agent here for type hinting

# Get a logger for this module
logger = logging.getLogger(__name__)

def create_tasks(agents: Dict[str, Agent]) -> List[Task]:
    """
    Creates and returns a list of CrewAI Tasks.

    Args:
        agents (Dict[str, Agent]): A dictionary of initialized agent instances.

    Returns:
        List[Task]: A list of initialized Task instances.
    """
    logger.info("Creating CrewAI Tasks.")
    tasks = []

    # Ensure required agents are present
    miner_agent = agents.get("contract_miner")
    if not miner_agent: logger.error("Contract miner agent not found.")
    investigative_agent_a = agents.get("investigative_agent_a")
    if not investigative_agent_a: logger.error("Investigative agent A not found.")
    investigative_agent_b = agents.get("investigative_agent_b")
    if not investigative_agent_b: logger.error("Investigative agent B not found.")
    investigative_agent_c = agents.get("investigative_agent_c")
    if not investigative_agent_c: logger.error("Investigative agent C not found.")
    ethics_agent = agents.get("ethics_agent")
    if not ethics_agent: logger.error("Ethics agent not found.")
    performance_monitor = agents.get("performance_monitor")
    if not performance_monitor: logger.error("Performance monitor agent not found.")


    # --- Define Tasks ---
    # Ensure task descriptions guide the agents on tool usage and expected output format (JSON)

    # Task 1: Mine a specific dataset
    # Input for the tool needs to be a JSON string
    mine_task = Task(
        description=(
            "Use the ContractMinerTool to create a balanced dataset from the loaded data. "
            "The dataset should have approximately 50% fraudulent contracts. "
            "Call the tool's _run method with the JSON input: `{{\"fraud_percentage\": 0.5}}`. "
            "Your output must be the JSON string returned by the tool, containing dataset metadata and the path to the saved data."
        ),
        agent=miner_agent,
        expected_output='A JSON string containing metadata about the mined dataset (including actual fraud percentage, size, and the path to the saved dataset file). Example: {"status": "success", "actual_fraud_percentage": 0.49, "dataset_size": 1000, "dataset_path": "/path/to/data.csv", ...}',
        # tool_use=True # Explicitly allow tool use if needed
    )
    tasks.append(mine_task)


    # Task 2-4: Detect fraud using different algorithms on the mined dataset
    # These tasks need the output (dataset_path) from the mine_task.
    # The agent needs to extract 'dataset_path' from the mine_task output JSON string.
    # The input for the detection tool needs to be a JSON string like '{"algorithm": "A", "data_path": "/path/to/data.csv"}'

    detect_task_a = Task(
        description=(
            "Analyze the output of the 'ContractMiningSpecialist' task (preceding task). "
            "Extract the value of the 'dataset_path' key from the JSON output string. "
            "Use the FraudDetectionTool with Algorithm 'A'. "
            "Call the tool's _run method with the JSON input: `{{\"algorithm\": \"A\", \"data_path\": \"[extracted_dataset_path]\"}}`. "
            "Replace `[extracted_dataset_path]` with the actual path you extracted. "
            "Your output must be the JSON string returned by the tool, including predictions and feature importances."
        ),
        agent=investigative_agent_a,
        expected_output='A JSON string detailing the fraud detection results from Algorithm A, including "predictions" (List[int]) and "feature_importances" (List[Tuple[float, str]]). Example: {"status": "success", "algorithm": "A", "predictions": [0, 1, ...], "feature_importances": [[0.5, "feat1"], ...], ...}',
        context=[mine_task], # Connects the output of mine_task to detect_task_a's context
        # tool_use=True
    )
    tasks.append(detect_task_a)


    detect_task_b = Task(
        description=(
            "Analyze the output of the 'ContractMiningSpecialist' task (preceding task). "
            "Extract the value of the 'dataset_path' key from the JSON output string. "
            "Use the FraudDetectionTool with Algorithm 'B'. "
            "Call the tool's _run method with the JSON input: `{{\"algorithm\": \"B\", \"data_path\": \"[extracted_dataset_path]\"}}`. "
            "Replace `[extracted_dataset_path]` with the actual path you extracted. "
            "Your output must be the JSON string returned by the tool, including predictions and feature importances."
        ),
        agent=investigative_agent_b,
        expected_output='A JSON string detailing the fraud detection results from Algorithm B, including "predictions" (List[int]) and "feature_importances" (List[Tuple[float, str]]). Example: {"status": "success", "algorithm": "B", "predictions": [0, 1, ...], "feature_importances": [[0.5, "feat1"], ...], ...}',
        context=[mine_task], # Connects to mine_task output
        # tool_use=True
    )
    tasks.append(detect_task_b)


    detect_task_c = Task(
        description=(
            "Analyze the output of the 'ContractMiningSpecialist' task (preceding task). "
            "Extract the value of the 'dataset_path' key from the JSON output string. "
            "Use the FraudDetectionTool with Algorithm 'C'. "
            "Call the tool's _run method with the JSON input: `{{\"algorithm\": \"C\", \"data_path\": \"[extracted_dataset_path]\"}}`. "
            "Replace `[extracted_dataset_path]` with the actual path you extracted. "
            "Your output must be the JSON string returned by the tool, including predictions and feature importances."
        ),
        agent=investigative_agent_c,
        expected_output='A JSON string detailing the fraud detection results from Algorithm C, including "predictions" (List[int]) and "feature_importances" (List[Tuple[float, str]]). Example: {"status": "success", "algorithm": "C", "predictions": [0, 1, ...], "feature_importances": [[0.5, "feat1"], ...], ...}',
        context=[mine_task], # Connects to mine_task output
        # tool_use=True
    )
    tasks.append(detect_task_c)


    # Task 5: Evaluate Ethics and Performance using results from detection tasks
    # This task needs JSON outputs from detect_task_a, _b, _c.
    # The agent needs to parse these, format input JSON for EthicsCheckerTool, call it for each algorithm, and aggregate results.

    evaluate_ethics_task = Task(
        description=(
            "Analyze the JSON outputs from the 'FraudDetectionSpecialistA', 'B', and 'C' tasks (preceding tasks). "
            "For each algorithm's output JSON string, parse it and extract the 'predictions' and 'feature_importances' data. "
            "For EACH algorithm's extracted data, format it into a new JSON string like: `{{\"predictions\": [...], \"feature_importances\": [...]}}`. "
            "Use the EthicsCheckerTool to evaluate the results FOR EACH ALGORITHM. Call the tool's _run method with the formatted JSON string for each algorithm's results. "
            "Collect the JSON output string from each EthicsCheckerTool call. "
            "Your final output should be a single JSON string containing a dictionary where keys are algorithm identifiers (e.g., 'Algorithm_A') and values are the JSON evaluation reports from the EthicsCheckerTool for that algorithm."
        ),
        agent=ethics_agent,
        expected_output='A JSON string containing a dictionary. The keys are algorithm identifiers (e.g., "Algorithm_A", "Algorithm_B") and the values are the JSON evaluation reports from the EthicsCheckerTool for that algorithm. Example: {"Algorithm_A": {"status": "success", "fairness_metrics": {...}}, "Algorithm_B": {...}}',
        context=[detect_task_a, detect_task_b, detect_task_c],  # Connects to detection tasks
        # tool_use=True
        # Note: This requires the agent's LLM capabilities to parse JSON, format JSON, and iterate.
    )
    tasks.append(evaluate_ethics_task)


    # Task 6: Monitor Performance (needs outputs from detection and ethics tasks)
    # This task needs the JSON outputs from detect_task_a, _b, _c AND evaluate_ethics_task.
    # The agent needs to parse these, aggregate relevant info into input JSON for PerformanceMonitorTool, and call the tool.

    monitor_performance_task = Task(
        description=(
            "Analyze the JSON outputs from the Fraud Detection tasks (Algorithms A, B, C) and the Ethics Evaluation task (preceding tasks). "
            "Extract key information such as detection counts, reported ethics metrics (fairness, performance scores) from the JSON outputs. "
            "Structure this information into a single JSON string suitable as input for the PerformanceMonitorTool's _run method. Include metrics for each algorithm and the overall ethics evaluation summary. "
            "Call the PerformanceMonitorTool's _run method with this combined JSON input string. "
            "Your final output must be the JSON string returned by the PerformanceMonitorTool, summarizing the performance analysis and feedback."
        ),
        agent=performance_monitor,
        expected_output='A JSON string containing the performance analysis report and feedback from the PerformanceMonitorTool. Example: {"status": "success", "summary": "Analysis results", "feedback": "Feedback"}',
        context=[detect_task_a, detect_task_b, detect_task_c, evaluate_ethics_task],  # Connects to detection and ethics and tasks
        # tool_use=True
         # Note: This also requires the agent's ability to parse and structure JSON from multiple sources.
    )
    tasks.append(monitor_performance_task)


    logger.info(f"Created {len(tasks)} tasks.")
    return tasks