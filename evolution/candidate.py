"""
Candidate class to be used during evolution.
"""
from pathlib import Path

import torch

class Candidate():
    """
    Candidate class that holds the model and stores evaluation and sorting information for evolution.
    Model can be persisted to disk.
    """
    def __init__(self,
                 cand_id: str,
                 parents: list[str],
                 model_params: dict,
                 actions: list[str],
                 outcomes: dict[str, bool]):
        self.cand_id = cand_id
        self.actions = actions
        self.outcomes = outcomes
        self.metrics = {}

        self.parents = parents
        self.rank = None
        self.distance = None

        # Model
        self.model_params = model_params
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.model = NNPrescriptor(**model_params).to(self.device)
        self.model.eval()

    @classmethod
    def from_seed(cls, path: Path, model_params: dict, actions: list[str], outcomes: dict[str, bool]):
        """
        Loads PyTorch seed from disk.
        """
        cand_id = path.stem
        parents = []
        candidate = cls(cand_id, parents, model_params, actions, outcomes)
        candidate.model.load_state_dict(torch.load(path))
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
            nn_outputs = self.model.forward(x)
            action = torch.argmax(nn_outputs).item()

        return action

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


class NNPrescriptor(torch.nn.Module):
    """
    Torch neural network that the candidate wraps around.
    """
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, out_size),
            torch.nn.Sigmoid()
        )

        # Orthogonal initialization
        for layer in self.nn:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight)
                layer.bias.data.fill_(0.01)

    def forward(self, x):
        """
        Forward pass of neural network.
        Returns a tensor of shape (batch_size, num_actions).
        Values are scaled between 0 and 1.
        """
        nn_output = self.nn(x)
        return nn_output
