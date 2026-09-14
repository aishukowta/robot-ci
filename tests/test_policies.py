import numpy as np
from robot_ci.policies.registry import get_policy
from robot_ci.policies.baseline import BaselinePolicy
from robot_ci.policies.candidate import CandidatePolicy

def test_registry():
    baseline = get_policy("baseline")
    candidate = get_policy("candidate")
    assert isinstance(baseline, BaselinePolicy)
    assert isinstance(candidate, CandidatePolicy)
    
def test_baseline_policy():
    policy = get_policy("baseline")
    assert policy.version == "v1.0-baseline"
    
    obs = np.zeros(10)
    action = policy.act(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == (4,)

def test_candidate_policy():
    policy = get_policy("candidate")
    assert policy.version == "v2.0-candidate"
    
    obs = np.zeros(10)
    action = policy.act(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == (4,)
