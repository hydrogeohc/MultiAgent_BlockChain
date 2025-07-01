import sys
import os
import logging

# Add the root directory of your project to the system path
# This assumes your project structure looks something like:
# /your_project_root/
#   ├── main.py
#   ├── config.py
#   ├── logging_setup.py
#   ├── data_loader.py
#   ├── llm_setup.py
#   ├── tools.py
#   ├── agents.py
#   ├── tasks.py
#   ├── crew_manager.py
#   └── temp_datasets/ # Created by ContractMinerTool
#   └── ... (your actual tool logic files if separate)

# Ensure your project root is added to sys.path if needed for imports like `from . import config`
# This approach assumes all other files are siblings of main.py or within subdirectories correctly imported.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Import Modules ---
try:
    from config import APP_CONFIG, validate_config
    from logging_setup import setup_logging
    from MAS.data_loader import load_dataset
    from MAS.llm_setup import initialize_llm
    from MAS.tools import ProductionContractMinerTool, ProductionFraudDetectionTool, ProductionEthicsCheckerTool, ProductionPerformanceMonitorTool # Import the tool classes
    from MAS.agents import create_agents
    from MAS.tasks import create_tasks
    from MAS.crew_manager import setup_crew
except ImportError as e:
    print(f"Error importing necessary modules. Ensure all files are in the correct path and dependencies are installed. Details: {e}", file=sys.stderr)
    sys.exit("Module import failed.")


# --- Main Execution Function ---
def run_workflow():
    """Orchestrates the CrewAI workflow."""
    # 1. Load and Validate Configuration
    try:
        validate_config(APP_CONFIG)
        # Set up logging after loading config, potentially using a log file path from config
        # setup_logging(level=logging.INFO, log_file=APP_CONFIG.get("log_file_path")) # Example with log file
        setup_logging(level=logging.INFO) # Basic console logging
        logger = logging.getLogger(__name__) # Get logger after setup
        logger.info("Configuration loaded and validated.")
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(f"Configuration error: {e}")


    # 2. Load Dataset
    # Pass the required flag column name from config to the data loader
    required_data_columns = [APP_CONFIG["ethics_checker_config"]["flag_column"]] # Add other essential columns here if any
    dataset = load_dataset(APP_CONFIG["data_file_path"], required_data_columns)

    if dataset is None:
        logger.critical("Dataset loading failed. Exiting workflow.")
        sys.exit("Dataset loading failed.")
    if dataset.empty:
        logger.warning("Dataset loaded is empty. Workflow may not produce meaningful results.")
        # Decide if you want to exit or proceed with an empty dataset


    # 3. Initialize LLM
    llm = initialize_llm(APP_CONFIG)

    if llm is None:
        logger.critical("LLM initialization failed. Exiting workflow.")
        sys.exit("LLM initialization failed.")


    # 4. Initialize Tools
    # Pass necessary data and configuration to tool instances
    logger.info("Initializing tools.")
    try:
        tools = {
            "ContractMinerTool": ProductionContractMinerTool(data=dataset, config=APP_CONFIG["contract_miner_config"]),
            # Note: Fraud Detection Tool is stateless w.r.t. data in _run, but might need config
            "FraudDetectionTool_A": ProductionFraudDetectionTool(algorithm="A"), # Pass config if needed
            "FraudDetectionTool_B": ProductionFraudDetectionTool(algorithm="B"), # Pass config if needed
            "FraudDetectionTool_C": ProductionFraudDetectionTool(algorithm="C"), # Pass config if needed
            "EthicsCheckerTool": ProductionEthicsCheckerTool(data=dataset, config=APP_CONFIG["ethics_checker_config"]), # Pass data and config
            "PerformanceMonitorTool": ProductionPerformanceMonitorTool(), # Pass config if needed
        }
        logger.info(f"Initialized {len(tools)} tool instances.")

    except Exception as e:
        logger.critical(f"Error initializing tools: {e}", exc_info=True)
        sys.exit(f"Error initializing tools: {e}")


    # 5. Create Agents
    # Pass the initialized tools and LLM to the agent creation function
    try:
        agents = create_agents(tools, llm)
        if not agents:
             logger.critical("Agent creation failed. Exiting workflow.")
             sys.exit("Agent creation failed.")

    except Exception as e:
        logger.critical(f"Error creating agents: {e}", exc_info=True)
        sys.exit(f"Error creating agents: {e}")


    # 6. Create Tasks
    # Pass the initialized agents to the task creation function
    try:
        tasks = create_tasks(agents)
        if not tasks:
            logger.critical("Task creation failed. Exiting workflow.")
            sys.exit("Task creation failed.")
    except Exception as e:
        logger.critical(f"Error creating tasks: {e}", exc_info=True)
        sys.exit(f"Error creating tasks: {e}")


    # 7. Setup Crew
    # Pass agents, tasks, config, and manager LLM to setup the crew
    try:
        # Get the manager LLM instance for the hierarchical process
        manager_llm_instance = initialize_llm(APP_CONFIG) # Re-initialize or pass the main llm if it's suitable for manager
        if APP_CONFIG.get("crew_process", "hierarchical").lower() == "hierarchical" and manager_llm_instance is None:
             logger.critical("Failed to initialize manager LLM for hierarchical process. Exiting.")
             sys.exit("Manager LLM initialization failed.")

        crew = setup_crew(agents, tasks, APP_CONFIG, manager_llm_instance) # Pass manager_llm_instance
        if crew is None:
             logger.critical("Crew setup failed. Exiting workflow.")
             sys.exit("Crew setup failed.")

    except Exception as e:
        logger.critical(f"Error setting up crew: {e}", exc_info=True)
        sys.exit(f"Error setting up crew: {e}")


    # 8. Kick Off the Process
    logger.info("--- Kicking off CrewAI workflow ---")
    try:
        # The kickoff method returns the final output of the last task by default
        # In a production system, you might want to capture outputs of intermediate tasks
        # by inspecting the Crew object after execution or by having tasks save their results.
        final_result = crew.kickoff()
        logger.info("--- CrewAI workflow finished ---")
        logger.info("\nFinal Workflow Result:")
        print(final_result)  # Print the final result

    except Exception as e:
        logger.critical(f"An error occurred during CrewAI workflow execution: {e}", exc_info=True)
        sys.exit(f"Workflow execution failed: {e}")


# --- Main Entry Point ---
if __name__ == "__main__":
    run_workflow()