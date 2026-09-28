import gzip
import json
from types import SimpleNamespace as NS
from verl.trainer.ppo.generation_trace import capture_engine, token_hash

class Tokenizer:
    def decode(self, ids, **kwargs):
        return '|'.join(map(str, ids))

def test_engine_trace_preserves_all_tokens_and_stopping(tmp_path):
    ids=list(range(3000))
    output=NS(request_id='r1',prompt_token_ids=[1,2],outputs=[NS(token_ids=ids,finish_reason='length',stop_reason=None)])
    path=capture_engine(tmp_path,Tokenizer(),[{'prompt_token_ids':[1,2]}],[output],NS(max_tokens=3000,temperature=1),{'diagnostic_role':'teacher','diagnostic_step':9},2)
    with gzip.open(path,'rt') as f:r=json.loads(next(f))
    assert r['output_token_ids']==ids
    assert r['output_length']==3000 and r['finish_reason']=='length'
    assert r['prompt_sha256']==token_hash([1,2])
    assert r['rank']==2 and r['training_step']==9 and r['fully_captured']
    assert not list(tmp_path.rglob('*.partial'))

def test_trace_rejects_missing_backend_outputs(tmp_path):
    import pytest
    with pytest.raises(ValueError,match='count mismatch'):
        capture_engine(tmp_path,Tokenizer(),[{'prompt_token_ids':[1]}],[],NS(),{},0)


def test_environment_trace_keeps_full_active_transitions(tmp_path):
    from verl.trainer.ppo.generation_trace import capture_environment
    import numpy as np
    long_text = 'observation ' * 3000
    path = capture_environment(tmp_path, {'diagnostic_step':146}, 3,
        ['task','task'], ['a','b'], [True,False], {'text':[long_text,'inactive']},
        ['<think>x</think><action>search[x]</action>','unused'],
        {'text':['next','unused']}, np.array([1.,0.]), np.array([True,False]),
        [{'won':np.bool_(True)},{}])
    with gzip.open(path,'rt') as f: rows=[json.loads(line) for line in f]
    assert len(rows)==1 and rows[0]['observation']['text']==long_text
    assert rows[0]['step']==146 and rows[0]['turn_id']==3
    assert rows[0]['reward']==1 and rows[0]['done'] and rows[0]['info']['won']
