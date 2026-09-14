from .base import Metric, MetricResult
from .success_rate import SuccessRate
from .completion_time import CompletionTime
from .trajectory_deviation import TrajectoryDeviation
from .collision import CollisionCount
from .action_difference import ActionDifference
from .reward_difference import RewardDifference

DEFAULT_METRICS = [
    SuccessRate(),
    CompletionTime(),
    TrajectoryDeviation(),
    CollisionCount(),
    ActionDifference(),
    RewardDifference(),
]
