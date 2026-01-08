from collections import deque


class ExperimentLogger:
    def __init__(self, log_window_size):
        self.log_window_size = log_window_size
        self.raw_history = {}
        self.all_logged_episodes = []  # Collects episodes where data was logged (every 16th)
        self.all_avg_values = {}  # Collects the 100-step rolling average
        self.tensors = {}
    def get_rolling_average(self, data_deque):
        """Computes the simple average of the values currently in the deque."""
        if not data_deque:
            return 0.0
        return sum(data_deque) / len(data_deque)
    def get_metric_list(self):
        return list(self.raw_history.keys())
    
    def append_tensor(self, episode, data):
        if data:
            for key, value in data.items():
                if key not in self.tensors:
                    self.tensors[key] = []
                self.tensors[key].append({"episode": episode, "value": value})
                
    def append(self, episode, data):
        if data:
            self.all_logged_episodes.append(episode)
            # Process each metric found in the data
            for metric, value in data.items():
                # Initialize deque for new metrics
                if metric not in self.raw_history:
                    self.raw_history[metric] = deque(maxlen=self.log_window_size)
                    self.all_avg_values[metric] = []
                
                # 1. Update the raw data deques
                new_value = float(value)
                self.raw_history[metric].append(new_value)
                
                # 2. Compute and store the rolling average for final plot
                avg_value = self.get_rolling_average(self.raw_history[metric])
                self.all_avg_values[metric].append(avg_value)
