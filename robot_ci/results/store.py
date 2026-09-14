import json
from pathlib import Path
from typing import List, Optional
from .models import TestRunResult

class ResultStore:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, result: TestRunResult) -> str:
        filepath = self.results_dir / f"run_{result.run_id}.json"
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        return str(filepath)
    
    def load(self, filepath: str) -> TestRunResult:
        with open(filepath) as f:
            data = json.load(f)
        return TestRunResult.from_dict(data)
    
    def load_latest(self) -> Optional[TestRunResult]:
        results = self.load_all()
        return results[0] if results else None
    
    def load_all(self) -> List[TestRunResult]:
        files = sorted(self.results_dir.glob("run_*.json"), reverse=True)
        results = []
        for f in files:
            try:
                results.append(self.load(str(f)))
            except Exception:
                continue
        results.sort(key=lambda r: r.timestamp, reverse=True)
        return results
