import copy
import math

import numpy as np
import pytest
import torch

from cog_ttt.confidence import ConfidenceRegistry, Estimate, anytime_radius, failure_allocation
from cog_ttt.graph import Graph
from cog_ttt.models import VersionStore
from cog_ttt.surrogate import SignedModel, residual_certificate
from cog_ttt.training import Preference, VerifiedTarget, preference_loss, sequence_kl_sample, train_program
from conftest import BinaryOwner


def test_anytime_budget_and_persistent_identifiers():
    assert anytime_radius(0.05, 2, 19) == pytest.approx(math.sqrt(2 / 19 * math.log(2 * math.pi ** 4 * 4 * 19 ** 2 / (36 * 0.05))))
    assert sum(failure_allocation(0.05, identifier, count) for identifier in range(1, 300)
               for count in range(1, 300)) < 0.05
    registry = ConfidenceRegistry()
    first = registry.key("state", 0, "null", "graph")
    registry.entries[first].add(0.5)
    restored = ConfidenceRegistry.restore(registry.record())
    second = restored.key("state", 1, "null", "graph")
    assert restored.entries[second].identifier > restored.entries[first].identifier
    assert restored.entries[first].scores == [0.5]
    broken = copy.deepcopy(restored.record())
    broken["entries"][second]["identifier"] = broken["entries"][first]["identifier"]
    with pytest.raises(ValueError):
        ConfidenceRegistry.restore(broken)


def test_credit_radius_uses_the_same_budget_and_nested_mean_event():
    from cog_ttt.confidence import credit_mean_radius
    for identifier, count in [(1, 1), (2, 100), (31, 42)]:
        expected = math.sqrt(math.log(2 / failure_allocation(0.05, identifier, count)) / (2 * count))
        assert credit_mean_radius(0.05, identifier, count) == pytest.approx(expected)
        assert 2 * credit_mean_radius(0.05, identifier, count) == pytest.approx(anytime_radius(0.05, identifier, count))


def test_preference_training_can_receive_signal_within_the_paper_probe_cap(settings):
    from cog_ttt.training import contrast_preference
    first, second = Estimate(3, [1.0] * 80), Estimate(4, [0.0] * 80)
    row = contrast_preference("shared", "graph", "state", "input", "yes", "no", [first, second], 0.05)
    assert len(first.scores) + len(second.scores) < settings["search"]["max_probes"]
    assert row.delta_radius < row.delta
    assert row.weight("graph") > 0
    owner = BinaryOwner()
    result = train_program(owner, "graph", "state", [row], [], 1, 0.1,
                           {**settings["training"], "lambda_verified": 0.0, "lambda_kl": 0.0})
    assert result["status"] == "trained"
    assert result["state"]["logit"] > 0
    assert float(owner.logit.detach()) == 0


def test_interaction_credit_radius_covers_each_of_the_four_means():
    from cog_ttt.confidence import credit_mean_radius
    from cog_ttt.training import contrast_preference
    estimates = [Estimate(index + 1, [score] * 40) for index, score in enumerate([1.0, 0.0, 0.5, 0.5])]
    row = contrast_preference("shared", "graph", "state", "input", "yes", "no", estimates, 0.05)
    radii = [credit_mean_radius(0.05, estimate.identifier, 40) for estimate in estimates]
    assert row.delta_radius == pytest.approx(sum(radii[:2]))
    assert row.interaction == pytest.approx(1.0)
    assert row.interaction_radius == pytest.approx(sum(radii))


def test_residual_envelope_uses_anytime_radius_and_global_cover():
    graph = Graph(nodes=[{"name": "a", "cost": 0, "operation": "act", "kind": "tool"}])
    model = SignedModel(graph, coefficients=[0.2], bias=0.4)
    estimate = Estimate(1, [0.4] * 20)
    certificate = residual_certificate(model, [(graph.decision([]), estimate)], 0.05, graph.enumerate(0))
    expected = anytime_radius(0.05, 1, 20) + 1.2
    assert certificate["allowance"] == pytest.approx(expected)
    assert certificate["residual_percentile_95"] == pytest.approx(0.0)
    assert certificate["covering_radius"] == 1


def test_graph_context_and_interaction_confidence_filter():
    row = Preference("shared", "graph", "state", "input", "yes", "no", 0.8, 0.2, 0.5, 0.1)
    assert row.weight("graph", maximum=2) == pytest.approx(0.6 * 1.4)
    assert row.weight("other") == 0
    assert Preference("shared", "graph", "state", "input", "yes", "no", 0.1, 0.2).weight("graph") == 0
    negative = Preference("shared", "graph", "state", "input", "yes", "no", -0.8, 0.2, -0.5, 0.1)
    assert negative.weight("graph", maximum=2) == row.weight("graph", maximum=2)


def test_weighted_sequence_loss_and_normalization_floor():
    first, second = torch.tensor([-1.0, -2.0]), torch.tensor([-2.0, -1.0])
    result = preference_loss(first, second, torch.tensor([1.0, -1.0]), torch.tensor([0.2, 0.3]))
    assert float(result) == pytest.approx(math.log1p(math.exp(-1)))
    assert float(preference_loss(first, second, torch.ones(2), torch.zeros(2))) == 0


def test_sampled_kl_has_the_sequence_kl_gradient_in_expectation():
    owner, reference = BinaryOwner(logit=0.7), BinaryOwner(logit=-0.4)
    probabilities = [owner.logit.sigmoid(), 1 - owner.logit.sigmoid()]
    expected = sum(probability * (owner.log_probability("input", token) - reference.log_probability("input", token))
                   for probability, token in zip(probabilities, ["yes", "no"]))
    exact_gradient = torch.autograd.grad(expected, owner.logit, retain_graph=True)[0]
    estimator = sum(probability.detach() * sequence_kl_sample(owner.log_probability("input", token),
                                                             reference.log_probability("input", token))
                    for probability, token in zip(probabilities, ["yes", "no"]))
    sampled_gradient = torch.autograd.grad(estimator, owner.logit)[0]
    assert float(sampled_gradient) == pytest.approx(float(exact_gradient))
    assert float(estimator.detach()) == pytest.approx(float(expected.detach()))


def test_candidate_training_restores_owner_and_preserves_other_versions(tmp_path, settings):
    first, second = BinaryOwner(), BinaryOwner("other", 0.4)
    store = VersionStore({first.name: first, second.name: second})
    row = VerifiedTarget("shared", "graph", "state", "input", "Return yes.", "yes",
                         {"passed": True, "tool_result_id": "check", "task_state": "state", "source": "execution_check"})
    result = train_program(first, "graph", "state", [], [row], 3, 0.5, settings["training"])
    assert result["status"] == "trained"
    assert float(first.logit.detach()) == 0
    assert float(result["state"]["logit"]) > 0
    store.add("pilot", "shared", result["state"], 0)
    manifest = store.save(tmp_path)
    store.activate("pilot")
    assert float(first.logit.detach()) > 0
    assert float(second.logit.detach()) == pytest.approx(0.4)
    store.activate()
    assert float(first.logit.detach()) == 0
    store.accept("pilot")
    assert store.version == 1
    store.restore(tmp_path, manifest)
    assert store.version == 0
    assert "pilot" in store.candidates


def test_immutable_parameter_files_survive_an_orphaned_save(tmp_path):
    first = BinaryOwner()
    initial = VersionStore({"shared": first})
    manifest = initial.save(tmp_path)
    initial.add("pilot", "shared", {"logit": torch.tensor(0.4, dtype=torch.float64)}, 0)
    orphan = initial.save(tmp_path)["candidates"]["pilot"]["state"]
    restored = VersionStore({"shared": BinaryOwner()})
    restored.restore(tmp_path, manifest)
    restored.add("pilot", "shared", {"logit": torch.tensor(0.8, dtype=torch.float64)}, 0)
    current = restored.save(tmp_path)
    assert current["candidates"]["pilot"]["state"] != orphan
    restored.restore(tmp_path, current)
    restored.activate("pilot")
    assert float(restored.owners["shared"].logit.detach()) == pytest.approx(0.8)


def test_local_model_content_identity_detects_same_shape_changes(tmp_path):
    from cog_ttt.identity import local_checkpoint_digest
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    weights = checkpoint / "weights.bin"
    weights.write_bytes(b"abcd")
    first = local_checkpoint_digest(checkpoint)
    weights.write_bytes(b"efgh")
    assert local_checkpoint_digest(checkpoint) != first


def test_unverified_targets_and_absent_signal_are_rejected(settings):
    with pytest.raises(ValueError):
        VerifiedTarget("shared", "graph", "state", "input", "requirement", "yes", {"passed": True, "source": "final_label"})
    result = train_program(BinaryOwner(), "graph", "state", [], [], 1, 0.1, settings["training"])
    assert result["status"] == "skipped_no_training_signal"


def test_disabled_verified_loss_cannot_materialize_an_update(settings, monkeypatch):
    owner = BinaryOwner()
    row = VerifiedTarget("shared", "graph", "state", "input", "Return yes.", "yes",
                         {"passed": True, "tool_result_id": "check", "task_state": "state", "source": "execution_check"})
    def unexpected_sample(*arguments, **keywords):
        pytest.fail("A program without training supervision should skip generation")
    monkeypatch.setattr(owner, "sample", unexpected_sample)
    result = train_program(owner, "graph", "state", [], [row], 2, 0.1,
                           {**settings["training"], "lambda_verified": 0.0})
    assert result == {"status": "skipped_no_training_signal", "steps": 0}
    assert float(owner.logit.detach()) == 0


def test_legal_group_interaction_cancels_singletons():
    from cog_ttt.measurements import group_interaction
    graph = Graph(nodes=[{"name": name, "cost": 0, "operation": "act", "kind": "tool"} for name in ("a", "b")])
    estimates = [Estimate(index + 1, [score] * 10) for index, score in enumerate([0.9, 0.4, 0.3, 0.1])]
    result = group_interaction(graph, [], ["a"], ["b"], estimates, 0, 0.05, residual_allowance=0.02)
    assert result["interaction"] == pytest.approx(0.3)
    assert result["residual_interaction_bound"] == pytest.approx(0.08)
    assert result["sampling_radius"] > 0


def test_real_causal_model_sequence_likelihood_and_reversible_gradient_steps(settings):
    from transformers import GPT2Config, GPT2LMHeadModel
    from cog_ttt.models import CausalOwner

    class Tokenizer:
        pad_token_id, eos_token_id = 0, 7

        def apply_chat_template(self, messages, **arguments):
            return [1, 2]

        def encode(self, text, **arguments):
            return [3, 4] if text == "yes" else [5]

        def decode(self, identifiers, **arguments):
            return "generated"

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=8, n_positions=32, n_embd=8, n_layer=1, n_head=1,
                                     eos_token_id=7, pad_token_id=0, bos_token_id=1,
                                     resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
    owner = CausalOwner("shared", model, Tokenizer(), {"context_tokens": 32, "max_output_tokens": 2})
    expected = torch.log_softmax(model(torch.tensor([[1, 2, 3, 4]])).logits.float(), dim=-1)
    assert float(owner.log_probability("input", "yes").detach()) == pytest.approx(float((expected[0, 1, 3] + expected[0, 2, 4]).detach()))
    initial = owner.snapshot()
    row = VerifiedTarget("shared", "graph", "state", "input", "Return yes.", "yes",
                         {"passed": True, "tool_result_id": "check", "task_state": "state", "source": "execution_check"})
    result = train_program(owner, "graph", "state", [], [row], 2, 0.01, settings["training"])
    assert any(not torch.equal(initial[name], result["state"][name]) for name in initial)
    assert all(torch.equal(parameter, owner.snapshot()[name]) for name, parameter in initial.items())
