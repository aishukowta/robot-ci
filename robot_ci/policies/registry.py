from .base import PolicyInterface
from .baseline import BaselinePolicy
from .candidate import CandidatePolicy

_REGISTRY = {"baseline": BaselinePolicy, "candidate": CandidatePolicy}

def get_policy(name: str, **kwargs) -> PolicyInterface:
    """Instantiate and load a policy by name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown policy: {name}. Available: {list(_REGISTRY.keys())}")
    policy = _REGISTRY[name](**kwargs)
    policy.load()
    return policy

def register_policy(name: str, policy_class) -> None:
    _REGISTRY[name] = policy_class
