import json

class PerformanceMonitorTool:
    name = "PerformanceMonitorTool"
    def __init__(self):
        self.performance_data = {}

    def update_performance(self, agent_name, metrics):
        self.performance_data[agent_name] = metrics

    def get_performance_report(self):
        report = "Performance Report:\n"
        for agent, metrics in self.performance_data.items():
            report += f"{agent}:\n"
            for metric, value in metrics.items():
                report += f"  {metric}: {value:.4f}\n"
        return report

    def save_performance_data(self, filename="performance_data.json"):
        with open(filename, "w") as f:
            json.dump(self.performance_data, f)

    def load_performance_data(self, filename="performance_data.json"):
        try:
            with open(filename, "r") as f:
                self.performance_data = json.load(f)
        except FileNotFoundError:
            print("No previous performance data found.")