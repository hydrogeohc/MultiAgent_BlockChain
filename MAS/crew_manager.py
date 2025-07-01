import logging
from typing import Dict, Any, List
from crewai import Crew, Agent, Task, Process, LLM # Import necessary CrewAI components

# Get a logger for this module
logger = logging.getLogger(__name__)

def setup_crew(agents: Dict[str, Agent], tasks: List[Task], config: Dict[str, Any], manager_llm: LLM) -> Crew:
    """
    Sets up and returns the CrewAI Crew.

    Args:
        agents (Dict[str, Agent]): A dictionary of initialized agent instances.
        tasks (List[Task]): A list of initialized task instances in workflow order.
        config (Dict[str, Any]): Application configuration dictionary.
        manager_llm (LLM): The LLM instance to use for the manager agent
                           (required for hierarchical process).

    Returns:
        Crew: The configured CrewAI Crew instance.

    Raises:
        ValueError: If the specified crew process is invalid or required components are missing.
    """
    logger.info("Setting up CrewAI Crew.")

    crew_agents = list(agents.values()) # Get agents as a list

    # Ensure the manager agent is included in the agents list for hierarchical process
    manager_agent = agents.get("manager")
    crew_process_str = config.get("crew_process", "hierarchical").lower()

    if crew_process_str == "hierarchical":
        if not manager_agent:
            logger.critical("Manager agent is required for hierarchical process but not found.")
            raise ValueError("Manager agent is required for hierarchical process.")
        if manager_llm is None:
             logger.critical("Manager LLM is required for hierarchical process but not provided.")
             raise ValueError("Manager LLM is required for hierarchical process.")
        process_type = Process.hierarchical
    elif crew_process_str == "sequential":
        process_type = Process.sequential
        if manager_agent:
             logger.warning("Manager agent is not used in sequential process.")
        if manager_llm:
             logger.warning("Manager LLM is not used in sequential process.")
    else:
        logger.critical(f"Invalid crew process '{crew_process_str}' specified in config.")
        raise ValueError(f"Invalid crew process: {crew_process_str}")

    try:
        crew = Crew(
            agents=crew_agents,
            tasks=tasks,
            process=process_type,
            manager_llm=manager_llm if process_type == Process.hierarchical else None,
            # Adjust memory and context window handling based on complexity and LLM limits
            respect_context_window=config.get("respect_context_window", True), # Example configurable
            memory=config.get("crew_memory", True), # Example configurable
            manager_agent=manager_agent if process_type == Process.hierarchical else None,
            # planning=config.get("crew_planning", True), # Example configurable
        )
        logger.info(f"Crew setup complete with process '{process_type}'.")
        return crew

    except Exception as e:
        logger.critical(f"Error setting up CrewAI Crew: {e}", exc_info=True)
        raise