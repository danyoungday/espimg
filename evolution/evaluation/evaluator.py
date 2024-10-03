"""
Evaluates candidates in order for them to be sorted.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evolution.candidate import Candidate
from vae.vae import VAE


class Evaluator:
    """
    Evaluates candidates by generating the actions and running the enroads model on them.
    Generates and stores context data based on config using ContextDataset.
    """
    def __init__(self, data: np.ndarray, vae: VAE, model_config: dict):
        self.vae = vae


    def evaluate_candidate(self, candidate: Candidate):
        """
        Evaluates a single candidate by running all the context through it and receiving all the batches of actions.
        Then evaluates all the actions and returns the average outcome.
        """
        pass

    def evaluate_candidates(self, candidates: list[Candidate]):
        """
        Evaluates all candidates. Doesn't unnecessarily evaluate candidates that have already been evaluated.
        """
        for candidate in tqdm(candidates, leave=False):
            if len(candidate.metrics) == 0:
                self.evaluate_candidate(candidate)
