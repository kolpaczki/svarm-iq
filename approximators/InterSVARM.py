import copy
import itertools
import math
import random
import numpy as np
from scipy.special import binom

from approximators import BaseShapleyInteractions
from approximators.base import determine_complete_subsets

class InterSVARM(BaseShapleyInteractions):
    """ Estimates the SI (for SII, STI) using the Stratified SVARM sampling approach """

    def __init__(self, N, order, interaction_type="SII", top_order: bool = True):
        min_order = order if top_order else 1
        super().__init__(N, order, min_order)
        self.interaction_type = interaction_type
        self.orders = list(range(min_order, order + 1))
        self.consumed_budget = 0
        self.strata_estimates = {}
        self.strata_counts = {}

        self.strata_weights = {}
        if self.interaction_type == "SII":
            for k in self.orders:
                self.strata_weights[k] = [1 / ((self.n-k+1) * math.comb(self.n - k, l)) for l in range(0, self.n - k + 1)]
        elif self.interaction_type == "STI":
            for k in self.orders:
                self.strata_weights[k] = [k / (self.n * math.comb(self.n-1, l)) for l in range(0, self.n - k + 1)]
        elif self.interaction_type == "FSI":
            for k in self.orders:
                self.strata_weights[k] = [(math.factorial(2*k-1) / math.pow(math.factorial(k-1), 2)) * (math.factorial(self.n - l - 1) * math.factorial(l + k - 1) / math.factorial(self.n + k - 1)) for l in range(0, self.n - k + 1)]
        elif self.interaction_type == "BHI":
            for k in self.orders:
                self.strata_weights[k] = [math.pow(2, self.n - k) for l in range(0, self.n - k + 1)]

    def _init_sampling_weights(self):
        weight_vector = np.zeros(shape=self.n - 1)
        for subset_size in range(1, self.n):
            weight_vector[subset_size - 1] = (self.n - 1) / (subset_size * (self.n - subset_size))
        sampling_weight = (np.asarray([0] + [*weight_vector] + [0])) / sum(weight_vector)
        return sampling_weight

    def approximate_with_budget(self, game, budget):
        self.initialize()

        # initializes probability distribution for sampled coalition sizes
        # 0,1,n-1, and n are excluded due to the exact calculation
        weight_vector = self._init_sampling_weights()

        complete_subsets, incomplete_subsets, remaining_budget = determine_complete_subsets(s=1, n=self.n, budget=budget, q=weight_vector)
        complete_subsets = [0] + complete_subsets + [self.n]

        self.exact_calculation(game, complete_subsets)
        # self.warmup(game, incomplete_subsets, budget)

        probs = [1.0 for _ in range(0, len(incomplete_subsets))]
        probs = [0 for _ in range(0, int(len(complete_subsets) / 2))] + probs + [0 for _ in range(0, int(len(complete_subsets) / 2))]
        probs = np.asarray(probs) / sum(probs)

        self.consumed_budget += budget - remaining_budget - 2
        while budget > self.consumed_budget and sum(probs) > 0:
            self.sample_and_update(game, probs)
            self.consumed_budget += 1

        estimates = self.getEstimates()
        results = self._turn_estimates_into_results(estimates)

        return copy.deepcopy(results)


    # initializes the estimates and the counters for each strata
    # one estimate and counter for each (S,l,W)
    # S: subset of the player set having size of the given orders
    # l: between 0 and n-k
    # W: a subset of S
    def initialize(self):
        self.consumed_budget = 0
        self.strata_estimates = {}
        self.strata_counts = {}
        for k in self.orders:
            subsets = list(itertools.combinations(self.N, k))
            for subset in subsets:
                self.strata_estimates[tuple(subset)] = [{} for _ in range(0, self.n - k + 1)]
                self.strata_counts[tuple(subset)] = [{} for _ in range(0, self.n - k + 1)]
                for l in range(0, self.n - k + 1):
                    for w in range(0, k + 1):
                        subsets_W = list(itertools.combinations(subset, w))
                        for subset_W in subsets_W:
                            self.strata_estimates[tuple(subset)][l][tuple(subset_W)] = 0
                            self.strata_counts[tuple(subset)][l][tuple(subset_W)] = 0


    # calculates the border strata exactly by evaluating all coalitions with size 0,1,n-1,n
    def exact_calculation(self, game, complete_subsets):
        # iterate over all sizes l that are to be sampled exhaustively
        for l in complete_subsets:
            all_samples = list(itertools.combinations(self.N, l))
            # iterate over coalitions of particular size l
            for sample in all_samples:
                # access the value function only once for each such coalition
                val = game(sample)
                # iterate over all interaction orders k
                for k in self.orders:
                    subsets = list(itertools.combinations(self.N, k))
                    # iterate over all estimates for order k
                    for subset in subsets:
                        subset_W = set(sample).intersection(set(subset))
                        subset_W = tuple(sorted(subset_W))
                        w = len(subset_W)
                        self.strata_estimates[tuple(subset)][l - w][subset_W] += val * (1 / math.comb(self.n-k, l - w))


    # sample a coalition for each stratum, afterwards all counters are > 0
    def warmup(self, game, budget):
        for k in self.orders:
            subsets = list(itertools.combinations(self.N, k))
            for subset in subsets:
                for w in range(0, k + 1):
                    subsets_W = list(itertools.combinations(subset, w))
                    for subset_W in subsets_W:
                        for l in range(max(2 - w, 0), min(self.n - 2 - w, self.n - k) + 1):
                            if budget > self.consumed_budget:
                                available_players = list(set(self.N).copy().difference(subset))
                                coalition = set(np.random.choice(available_players, l, replace=False))
                                self.strata_estimates[tuple(subset)][l][tuple(subset_W)] = game(coalition)
                                self.strata_counts[tuple(subset)][l][tuple(subset_W)] = 1
                                self.consumed_budget += 1


    # sample a set and updates the strata accordingly
    def sample_and_update(self, game, probs):
        size = int(np.random.choice(range(0, self.n + 1), 1, p=probs))
        coalition = set(np.random.choice(list(self.N), size, replace=False))
        val = game(coalition)
        for k in self.orders:
            subsets = list(itertools.combinations(self.N, k))
            for subset in subsets:
                subset_W = coalition.intersection(set(subset))
                subset_W = tuple(sorted(subset_W))
                w = len(subset_W)
                avg_old = self.strata_estimates[tuple(subset)][size - w][subset_W]
                count_old = self.strata_counts[tuple(subset)][size - w][subset_W]
                self.strata_estimates[tuple(subset)][size - w][subset_W] = (avg_old * count_old + val) / (count_old + 1)
                self.strata_counts[tuple(subset)][size - w][subset_W] += 1


    # aggregates all strata estimates to estimates, one for each considered coalition, and returns them
    def getEstimates(self):
        estimates = {}
        for k in self.orders:
            subsets = list(itertools.combinations(self.N, k))
            for subset in subsets:
                estimates[tuple(subset)] = 0
                for l in range(0, self.n - k + 1):
                    factor = math.comb(self.n-k, l) * self.strata_weights[k][l]
                    for w in range(0, k + 1):
                        subsets_W = list(itertools.combinations(subset, w))
                        for subset_W in subsets_W:
                            strata_estimate = self.strata_estimates[tuple(subset)][l][tuple(subset_W)]
                            if (k - w) % 2 == 0:
                                estimates[tuple(subset)] += factor * strata_estimate
                            else:
                                estimates[tuple(subset)] -= factor * strata_estimate
        return estimates


    def _turn_estimates_into_results(self, estimates: dict[tuple, float]) -> dict[int, np.ndarray]:
        results: dict[int, np.ndarray] = self.init_results()
        for coalition, estimate in estimates.items():
            order = len(coalition)
            results[order][coalition] = estimate
        return results
