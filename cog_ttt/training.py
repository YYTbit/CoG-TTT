import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Preference:
    owner: str
    graph: str
    task_state: str
    history: object
    first: object
    second: object
    delta: float
    delta_radius: float
    interaction: float = 0.0
    interaction_radius: float = 0.0

    def __post_init__(self):
        values = [self.delta, self.delta_radius, self.interaction, self.interaction_radius]
        if not np.isfinite(values).all() or min(self.delta_radius, self.interaction_radius) < 0:
            raise ValueError("Preference estimates are finite with nonnegative uncertainty radii")
        if not self.owner or not self.graph or not self.task_state or not self.first or not self.second:
            raise ValueError("Preferences retain their owner, graph, checkpoint, and two node outputs")

    @property
    def sign(self):
        return int(np.sign(self.delta))

    def weight(self, graph, kappa=1.0, maximum=1.0, confidence=True):
        if not math.isfinite(kappa) or not math.isfinite(maximum) or kappa < 0 or maximum <= 0:
            raise ValueError("Training weights use nonnegative amplification and a positive cap")
        if graph != self.graph:
            return 0.0
        radius = self.delta_radius if confidence else 0.0
        interaction_radius = self.interaction_radius if confidence else 0.0
        strength = max(abs(self.delta) - radius, 0.0)
        amplification = 1 + kappa * max(self.sign * self.interaction - interaction_radius, 0.0)
        return min(maximum, strength * amplification)


@dataclass(frozen=True)
class VerifiedTarget:
    owner: str
    graph: str
    task_state: str
    history: object
    requirement: str
    target: str
    verification: dict

    def __post_init__(self):
        if (self.verification.get("passed") is not True or not self.verification.get("tool_result_id")
                or self.verification.get("task_state") != self.task_state
                or self.verification.get("source") not in {"tool", "execution_check", "api_validation"}):
            raise ValueError("Verified targets require an authorized tool check at the same task state")
        if not self.target or not self.owner or not self.graph:
            raise ValueError("A verified training target records its graph and parameter owner")


def contrast_preference(owner, graph, task_state, history, first, second, estimates, delta):
    from .confidence import credit_mean_radius
    if len(estimates) not in (2, 4) or any(not estimate.scores for estimate in estimates):
        raise ValueError("Continuation credit uses two observed legal outputs and an optional legal alternative graph")
    means = [estimate.mean for estimate in estimates]
    radii = [credit_mean_radius(delta, estimate.identifier, len(estimate.scores)) for estimate in estimates]
    difference = means[0] - means[1]
    interaction = difference - (means[2] - means[3]) if len(estimates) == 4 else 0.0
    return Preference(owner, graph, task_state, history, first, second, difference, sum(radii[:2]),
                      interaction, sum(radii) if len(estimates) == 4 else 0.0)


def preference_loss(first, second, signs, weights, beta=1.0, epsilon=0.001):
    import torch
    import torch.nn.functional as functional
    if first.ndim != 1 or first.shape != second.shape or first.shape != signs.shape or first.shape != weights.shape:
        raise ValueError("Preference likelihoods, signs, and weights have matching vector dimensions")
    if beta <= 0 or epsilon <= 0 or not np.isfinite([beta, epsilon]).all():
        raise ValueError("Preference scaling and normalization floor are positive")
    if not torch.isfinite(first).all() or not torch.isfinite(second).all() or not torch.isfinite(weights).all():
        raise ValueError("Preference losses use finite sequence log probabilities and weights")
    if torch.any(weights < 0) or torch.any((signs != -1) & (signs != 0) & (signs != 1)):
        raise ValueError("Preference weights are nonnegative and signs are -1, 0, or 1")
    return (weights * functional.softplus(-beta * signs * (first - second))).sum() / weights.sum().clamp_min(epsilon)


def sequence_kl_sample(current, reference):
    difference = current - reference.detach()
    return difference.detach() + difference.detach() * (current - current.detach())


def train_program(owner, graph, task_state, preferences, verified, steps, learning_rate, settings, seed=0):
    import torch
    if isinstance(steps, bool) or int(steps) != steps or steps < 1 or learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("Candidate programs declare positive gradient steps and learning rate")
    if any(not math.isfinite(settings.get(key, 0.1)) or settings.get(key, 0.1) < 0
           for key in ("lambda_verified", "lambda_kl")):
        raise ValueError("Verified supervision and KL penalties are finite and nonnegative")
    preferences = [row for row in preferences if row.owner == owner.name and row.task_state == task_state]
    verified = [row for row in verified if row.owner == owner.name and row.graph == graph and row.task_state == task_state]
    if settings.get("lambda_verified", 0.1) == 0:
        verified = []
    weights = [row.weight(graph, settings.get("kappa", 1.0), settings.get("maximum_weight", 1.0),
                          settings.get("confidence_weights", True)) for row in preferences]
    retained = [(row, weight) for row, weight in zip(preferences, weights) if weight > 0]
    if not retained and not verified:
        return {"status": "skipped_no_training_signal", "steps": 0}
    accepted = owner.snapshot()
    parameters = [parameter for parameter in owner.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("A training program has no declared trainable parameters")
    history = []
    try:
        for step in range(int(steps)):
            for parameter in parameters:
                parameter.grad = None
            contexts = [row.history for row, _ in retained] + [row.history for row in verified]
            samples, reference = [], []
            if settings.get("lambda_kl", 0.1) > 0:
                samples = [owner.sample(value, seed + 1000 * step + index)
                           for index, value in enumerate(contexts)]
                candidate = owner.snapshot()
                owner.restore(accepted)
                with torch.no_grad():
                    reference = [owner.log_probability(value, sample).detach()
                                 for value, sample in zip(contexts, samples)]
                owner.restore(candidate)
            # Restore the reference before constructing the current-policy autograd graph.
            zero = parameters[0].reshape(-1)[0] * 0
            if retained:
                first = torch.stack([owner.log_probability(row.history, row.first) for row, _ in retained])
                second = torch.stack([owner.log_probability(row.history, row.second) for row, _ in retained])
                signs = first.new_tensor([row.sign for row, _ in retained])
                scores = first.new_tensor([weight for _, weight in retained])
                preference = preference_loss(first, second, signs, scores, settings.get("beta", 1.0),
                                               settings.get("epsilon_z", 0.001))
            else:
                preference = zero
            supervised = (-torch.stack([owner.log_probability(row.history, row.target, row.requirement)
                                         for row in verified]).mean()) if verified else zero
            divergence = zero
            if samples:
                current = [owner.log_probability(history_value, sample)
                           for history_value, sample in zip(contexts, samples)]
                divergence = torch.stack([sequence_kl_sample(value, baseline)
                                          for value, baseline in zip(current, reference)]).mean()
            loss = preference + settings.get("lambda_verified", 0.1) * supervised + settings.get("lambda_kl", 0.1) * divergence
            if not torch.isfinite(loss):
                raise ValueError("The graph-conditioned training objective is nonfinite")
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in parameters):
                raise ValueError("Candidate gradients are nonfinite")
            with torch.no_grad():
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.add_(parameter.grad, alpha=-learning_rate)
            history.append({"step": step + 1, "loss": float(loss.detach()),
                            "preference_loss": float(preference.detach()), "verified_loss": float(supervised.detach()),
                            "sampled_sequence_kl": float(divergence.detach()), "learning_rate": learning_rate})
        candidate = owner.snapshot()
    finally:
        owner.restore(accepted)
        for parameter in parameters:
            parameter.grad = None
    return {"status": "trained", "steps": int(steps), "state": candidate, "history": history,
            "weighted_preferences": len(retained), "verified_targets": len(verified),
            "kl_estimator": "current_policy_sequence_samples_with_score_function_gradient"}
