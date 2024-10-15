"""
Candidate class to be used during evolution.
"""
from pathlib import Path

import torch

class Prescriptor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(12, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        nn_outputs = self.model(x)
        return (nn_outputs > 0.5).float()


class Candidate():
    """
    Candidate class that holds the model and stores evaluation and sorting information for evolution.
    Model can be persisted to disk.
    """
    def __init__(self,
                 cand_id: str,
                 parents: list[str],
                 model_params: dict,
                 outcomes: dict[str, bool]):
        self.cand_id = cand_id
        self.outcomes = outcomes
        self.metrics = {}

        self.parents = parents
        self.rank = None
        self.distance = None

        # Model
        self.model_params = model_params
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Prescriptor().to(self.device)
        self.model.eval()

    @classmethod
    def from_seed(cls, path: Path, model_params: dict, outcomes: dict[str, bool]):
        """
        Loads PyTorch seed from disk.
        """
        cand_id = path.stem
        parents = []
        candidate = cls(cand_id, parents, model_params, outcomes)
        candidate.model.load_state_dict(torch.load(path, weights_only=True))
        return candidate

    def save(self, path: Path):
        """
        Saves PyTorch state dict to disk.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def prescribe(self, x: torch.Tensor) -> int:
        """
        Returns the highest probability action.
        """
        with torch.no_grad():
            return self.model.forward(x)

    def record_state(self):
        """
        Records metrics as well as seed and parents for reconstruction.
        """
        state = {
            "cand_id": self.cand_id,
            "parents": self.parents,
            "rank": self.rank,
            "distance": self.distance,
        }
        for outcome in self.outcomes:
            state[outcome] = self.metrics[outcome]
        return state

    def __str__(self):
        return f"Candidate({self.cand_id})"

    def __repr__(self):
        return f"Candidate({self.cand_id})"
