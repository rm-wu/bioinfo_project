from collections import defaultdict


class MetricMonitor:
    """
    Class used to monitor the evolution of different metrics for the model
    """
    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric['val'] += val
        metric['count'] += 1
        metric['avg'] = metric['val'] / metric['count']

    def __str__(self):
        return " | ".join(
            [
                f"{metric_name}: {metric['avg']:.{self.float_precision}f}"
                for (metric_name, metric) in self.metrics.items()
            ]
        )
