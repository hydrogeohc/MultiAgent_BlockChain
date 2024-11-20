from crewai import Agent
from Crewai_v2.tools.sampling_tool import sampling_tool 

class SamplingAgent(Agent):
    def __init__(self):
        super().__init__()
        self.tools = [sampling_tool]  # Register the sampling tool

    async def act(self, data, flag_column='flag', flag_percentage=0.6, n_samples=20):
        return self.use_tool("sampling_tool", data=data, flag_column=flag_column, flag_percentage=flag_percentage, n_samples=n_samples)
