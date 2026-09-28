import ast
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
from verl.trainer.ppo.thinking_correspondence import EncoderDiagnostics


class EncoderAllTests(unittest.TestCase):
    def invoke(self, size, count=8, bad=False):
        path = Path(__file__).resolve().parents[1] / 'verl/trainer/ppo/thinking_correspondence_trainer.py'
        tree = ast.parse(path.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == '_encode_thinking_texts')
        scope = dict(torch=torch, time=time, EncoderDiagnostics=EncoderDiagnostics,
                     encode_thinking_on_actor_rank_zero=object())
        module = ast.Module(body=[ast.ImportFrom(module='__future__',
             names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
        def execute(name, funcs, shards, configs, offload):
            self.assertEqual(name, 'execute_func_rank_zero')
            self.assertEqual(len(shards), count)
            self.assertEqual(offload, [True]*count)
            self.assertEqual([len(s) for s in shards], [len(range(r,size,count)) for r in range(count)])
            results = []
            for rank, shard in enumerate(shards):
                emb = torch.tensor([[float(x), float(x)+1] for x in shard]).reshape(-1,2)
                results.append(dict(embeddings=emb, valid=torch.tensor([int(x)%2==0 for x in shard], dtype=torch.bool),
                    diagnostics=asdict(EncoderDiagnostics(num_texts=len(shard), encoded_tokens=len(shard)*3,
                        max_encoded_tokens=3 if shard else 0, forward_seconds=float(rank))),
                    load_seconds=0., stage_in_seconds=.1, encode_seconds=rank+1., stage_out_seconds=.2))
            if bad: results.pop()
            return results
        obj = SimpleNamespace(m1_encoder_execution='actor_all', _resolved_encoder_config=lambda: {},
                              actor_rollout_wg=SimpleNamespace(world_size=count, execute_all_sync=execute))
        return scope['_encode_thinking_texts'](obj, tuple(str(i) for i in range(size)))

    def test_uneven_and_empty_shards(self):
        for size in [0, 3, 19]:
            emb, valid, diag, timings = self.invoke(size)
            self.assertEqual(emb.shape, (size,2))
            self.assertEqual(emb[:,0].tolist(), list(range(size)))
            self.assertEqual(valid.tolist(), [i%2==0 for i in range(size)])
            self.assertEqual(diag.num_texts, size)
            self.assertEqual(diag.encoded_tokens, size*3)
            self.assertEqual(diag.forward_seconds, 7.)
            self.assertEqual(timings['encode_seconds'], 8.)

    def test_missing_worker_fails(self):
        with self.assertRaises(RuntimeError):
            self.invoke(19, bad=True)


if __name__ == '__main__':
    unittest.main()
