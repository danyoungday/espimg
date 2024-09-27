"""
Single objective sorter - just sorts the candidates by their single objective.
"""
from abc import ABC, abstractmethod

from evolution.candidate import Candidate


class SOSorter(ABC):
    """
    Single objective sorter that checks if the candidates all have one objective then sorts by it.
    """
    def sort_candidates(self, candidates: list[Candidate]) -> list[Candidate]:
        """
        Sorts candidates based on some criteria.
        """
        for candidate in candidates:
            assert len(candidate.outcomes) == 1

        candidates.sort(key=lambda cand: cand.metrics[cand.outcomes[0]])
        return candidates
