from .confidence import anytime_radius


def legal_difference(graph, first, second, first_estimate, second_estimate, budget, delta, residual_allowance=None):
    for decision in (first, second):
        graph.validate(decision, budget)
    if not first_estimate.scores or not second_estimate.scores:
        raise ValueError("A legal contrast has observed continuation scores for both graphs")
    radius = sum(anytime_radius(delta, estimate.identifier, len(estimate.scores))
                 for estimate in (first_estimate, second_estimate))
    result = {"difference": first_estimate.mean - second_estimate.mean, "sampling_radius": radius}
    if residual_allowance is not None:
        if residual_allowance < 0:
            raise ValueError("Uniform residual allowances are nonnegative")
        result["residual_contrast_bound"] = 2 * residual_allowance
    return result


def group_interaction(graph, closure, first_group, second_group, estimates, budget, delta, residual_allowance=None):
    closure, first_group, second_group = set(closure), set(first_group), set(second_group)
    if closure & (first_group | second_group) or first_group & second_group:
        raise ValueError("Optional groups and prerequisite closure are disjoint")
    active_sets = [closure | first_group | second_group, closure | first_group, closure | second_group, closure]
    for active in active_sets:
        graph.validate(graph.decision(active), budget)
    if len(estimates) != 4 or any(not estimate.scores for estimate in estimates):
        raise ValueError("Group interaction uses four observed legal continuations in CAB, CA, CB, C order")
    result = {"interaction": sum(sign * estimate.mean for sign, estimate in zip((1, -1, -1, 1), estimates)),
              "sampling_radius": sum(anytime_radius(delta, estimate.identifier, len(estimate.scores)) for estimate in estimates)}
    if residual_allowance is not None:
        if residual_allowance < 0:
            raise ValueError("Uniform residual allowances are nonnegative")
        result["residual_interaction_bound"] = 4 * residual_allowance
    return result
