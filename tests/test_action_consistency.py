import ast
from pathlib import Path

import pytest
import torch

from verl.trainer.ppo.action_consistency import action_consistency_mask, validate_action_filter_config
from verl.trainer.ppo.thinking_correspondence import (
    TurnIdentity, combine_cross_teacher_scores, select_sparse_sources,
)


def test_webshop_semantics_invalid_actions_and_threshold():
    teacher = ['click[A]', 'search[red shoes]', 'click[B]', 'invalid', 'click[Buy Now]']
    student = ['click[A]', 'search[crimson shoes]', 'click[B]', 'search[blue hat]', 'click[Buy Now]']
    sims = torch.ones(5, 5)
    sims[1, 1] = .85
    sims[1, 3] = .79
    valid = torch.ones(5, dtype=torch.bool)
    actual = action_consistency_mask(torch.ones(5, 5, dtype=torch.bool), sims,
        teacher, student, valid, valid, .8, webshop=True)
    assert actual.nonzero().tolist() == [[0, 0], [1, 1], [2, 2], [4, 4]]
    valid[1] = False
    actual = action_consistency_mask(torch.ones(5, 5, dtype=torch.bool), sims,
        teacher, student, valid, valid, .8, webshop=True)
    assert not actual[1].any() and not actual[:, 1].any()


def test_filtered_topk_no_replacement_renormalization_and_diagonal_fallback():
    identities = [TurnIdentity('task', str(i), 0) for i in range(4)]
    thinking = torch.tensor([[0., 0., 0., .99], [0., 0., 0., .9],
                             [0., 0., 0., .8], [0., 0., 0., 0.]])
    initial = select_sparse_sources(thinking, thinking.ge(.7), identities, top_k=2)
    assert initial == ((), (), (), (0, 1))
    candidates = torch.zeros_like(thinking, dtype=torch.bool)
    for target, sources in enumerate(initial):
        for source in sources:
            candidates[source, target] = True
    actions = ['click[wrong]', 'click[right]', 'click[right]', 'click[right]']
    valid = torch.ones(4, dtype=torch.bool)
    mask = action_consistency_mask(candidates, torch.ones(4, 4), actions, actions,
                                  valid, valid, .8, webshop=True)
    selected = tuple(tuple(source for source in sources if mask[source, target])
                     for target, sources in enumerate(initial))
    assert selected == ((), (), (), (1,))
    diagonal = torch.full((4, 2), -2.)
    student = torch.full((4, 2), -3.)
    cross = {(1, 3): torch.tensor([-1., -4.]), (2, 3): torch.tensor([-3., -5.])}
    result = combine_cross_teacher_scores(diagonal, student, thinking, selected, cross)
    torch.testing.assert_close(result['selected_source_weights'][3].sum(), torch.tensor(1.))
    torch.testing.assert_close(result['corrected_teacher_log_probs'][:3], diagonal[:3])
    assert result['correction_strength'][:3].eq(0).all()
    # Agreement does not filter negative gaps or force every correction upward.
    assert result['corrected_teacher_log_probs'][3, 1] < diagonal[3, 1]


@pytest.mark.parametrize('threshold', [float('nan'), float('inf'), -1.1, 1.1, True, '0.8'])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        validate_action_filter_config({'similarity_threshold': threshold})


def test_empty_pool_and_nonfinite_embedding_similarity():
    pool = torch.ones(2, 2, dtype=torch.bool)
    sims = torch.tensor([[float('nan'), float('inf')], [.79, -.9]])
    valid = torch.ones(2, dtype=torch.bool)
    assert not action_consistency_mask(pool, sims, ['a', 'b'], ['a', 'b'], valid, valid, .8).any()
    assert not action_consistency_mask(pool & False, torch.ones(2, 2), ['a', 'b'], ['a', 'b'], valid, valid, .8).any()


def test_config_default_and_legacy_cross_prompt():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from verl.trainer.main_beyond_timestamps import _validate_beyond_timestamps_config
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(root / 'verl/trainer/config')):
        baseline = compose(config_name='beyond_timestamps_trainer')
        filtered = compose(config_name='beyond_timestamps_trainer', overrides=[
            'algorithm.thinking_correspondence.action_filter.enabled=true'])
    _validate_beyond_timestamps_config(filtered)
    assert not baseline.algorithm.thinking_correspondence.action_filter.enabled
    assert OmegaConf.to_container(baseline.actor_rollout_ref) == OmegaConf.to_container(filtered.actor_rollout_ref)
    assert baseline.algorithm.mechanism2 == filtered.algorithm.mechanism2
    assert 'cross_context_mode' not in baseline.algorithm.thinking_correspondence
    # Production builder keeps source prompt and target response token identities.
    from verl import DataProto
    from verl.trainer.ppo.thinking_correspondence import compose_cross_teacher_inputs
    from verl.utils.model import compute_position_id_with_mask
    tree = ast.parse((root / 'verl/trainer/ppo/thinking_correspondence_trainer.py').read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_build_cross_teacher_batch')
    node.decorator_list = []; node.returns = None
    for arg in node.args.args: arg.annotation = None
    scope = dict(DataProto=DataProto, compose_cross_teacher_inputs=compose_cross_teacher_inputs,
                 compute_position_id_with_mask=compute_position_id_with_mask)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<builder>', 'exec'), scope)
    ids = torch.tensor([[0, 10, 11], [20, 21, 22]])
    responses = torch.tensor([[30, 31], [40, 41]])
    mask = ids.ne(0).long()
    batch = DataProto.from_dict(tensors={'input_ids': torch.cat([ids, responses], -1),
        'attention_mask': torch.cat([mask, torch.ones_like(responses)], -1), 'responses': responses})
    actual = scope['_build_cross_teacher_batch'](batch, batch, [(0, 1)]).batch
    assert actual['input_ids'].tolist() == [[0, 10, 11, 40, 41]]


@pytest.mark.parametrize('payload', ['click[A]', 'search[red shoes]'])
def test_real_response_parser_to_webshop_filter(payload):
    from verl.trainer.ppo.thinking_correspondence import parse_tagged_response
    parsed = parse_tagged_response(f'<think>inspect state</think><action>  {payload}  </action>')
    assert parsed.thinking == 'inspect state'
    assert parsed.action == payload
    valid = torch.tensor([parsed.action_valid])
    accepted = action_consistency_mask(torch.ones(1, 1, dtype=torch.bool),
        torch.ones(1, 1), [parsed.action], [parsed.action], valid, valid, .8, webshop=True)
    assert accepted.item()


def test_empty_action_is_invalid_after_extracting_inner_text():
    from verl.trainer.ppo.thinking_correspondence import parse_tagged_response
    parsed = parse_tagged_response('<think>reason</think><action>   </action>')
    assert parsed.action == ''
    assert not parsed.action_valid
