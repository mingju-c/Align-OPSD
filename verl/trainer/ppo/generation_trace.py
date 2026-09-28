"""Opt-in lossless text/token traces for offline diagnosis; no training mutations."""
from pathlib import Path
import dataclasses
import gzip
import hashlib
import json
import os
import time
import uuid


def token_hash(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()


def write_records(directory, category, records):
    target = Path(directory) / category
    target.mkdir(parents=True, exist_ok=True)
    path = target / f'{time.time_ns()}-pid{os.getpid()}-{uuid.uuid4().hex}.jsonl.gz'
    temp = path.with_suffix('.partial')
    with gzip.open(temp, 'wt', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)
    return str(path)


def capture_engine(directory, tokenizer, inputs, outputs, params, meta, rank):
    """Keep exact backend inputs/outputs BEFORE padding and parser processing."""
    if len(inputs) != len(outputs):
        raise ValueError('trace: backend request count mismatch')
    settings = {k: getattr(params, k, None) for k in
                ('max_tokens','temperature','top_p','top_k','seed','n','ignore_eos','stop','stop_token_ids','detokenize')}
    def records():
        for index, (request, result) in enumerate(zip(inputs, outputs)):
            ids = list(request['prompt_token_ids'])
            backend_prompt = getattr(result, 'prompt_token_ids', None)
            for completion in result.outputs:
                out = list(completion.token_ids)
                yield dict(schema_version=1, role=meta.get('diagnostic_role', 'student'),
                           training_step=meta.get('diagnostic_step'), environment_turn=meta.get('diagnostic_turn'),
                           validate=bool(meta.get('validate', False)), rank=rank, local_index=index,
                           request_id=str(getattr(result, 'request_id', '')), prompt_sha256=token_hash(ids),
                           prompt_token_ids=ids, backend_prompt_token_ids=backend_prompt,
                           prompt_text=tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                           output_token_ids=out,
                           output_text=tokenizer.decode(out, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                           parser_text=tokenizer.decode(out, skip_special_tokens=True),
                           output_length=len(out), finish_reason=getattr(completion, 'finish_reason', None),
                           stop_reason=getattr(completion, 'stop_reason', None), sampling=settings,
                           fully_captured=True)
    return write_records(directory, 'engine', records())


def capture_pairs(directory, step, identities, student, teacher, teacher_responses,
                  parsed_student, parsed_teacher, construction, tokenizer):
    s_len = student.batch['responses'].shape[-1]
    t_len = teacher.batch['responses'].shape[-1]
    def records():
        for i, identity in enumerate(identities):
            sp = student.batch['input_ids'][i, :-s_len]
            sm = student.batch['attention_mask'][i, :-s_len].bool()
            tp = teacher.batch['input_ids'][i, :-t_len]
            tm = teacher.batch['attention_mask'][i, :-t_len].bool()
            sp = sp[sm].detach().cpu().tolist(); tp = tp[tm].detach().cpu().tolist()
            sr = student.batch['responses'][i].detach().cpu().tolist()
            tr = teacher_responses[i].detach().cpu().tolist()
            yield dict(schema_version=1, step=int(step), canonical_index=i,
                       identity=dataclasses.asdict(identity), student_prompt_token_ids=sp,
                       teacher_prompt_token_ids=tp, student_prompt_sha256=token_hash(sp),
                       teacher_prompt_sha256=token_hash(tp),
                       student_prompt_text=tokenizer.decode(sp, skip_special_tokens=False),
                       teacher_prompt_text=tokenizer.decode(tp, skip_special_tokens=False),
                       student_output_padded_token_ids=sr, teacher_output_padded_token_ids=tr,
                       student_output_text=tokenizer.decode(sr, skip_special_tokens=True),
                       teacher_output_text=tokenizer.decode(tr, skip_special_tokens=True),
                       parsed_student=dataclasses.asdict(parsed_student[i]),
                       parsed_teacher=dataclasses.asdict(parsed_teacher[i]),
                       construction=construction[i], fully_captured=True)
    return write_records(directory, 'pairs', records())


def capture_environment(directory, meta, turn, task_ids, trajectory_ids, active,
                        observations, outputs, next_observations, rewards, dones, infos):
    """All active environment transitions; no sampling or text truncation."""
    def plain(value):
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        if hasattr(value, 'tolist'):
            return plain(value.tolist())
        return value
    def observation_at(obs, i):
        return {k: plain(v[i]) if v is not None else None for k, v in obs.items()}
    return write_records(directory, 'trajectories', (
        dict(schema_version=1, step=meta.get('diagnostic_step'),
             role=meta.get('diagnostic_role', 'student'), turn_id=int(turn),
             task_id=str(task_ids[i]), rollout_id=str(trajectory_ids[i]),
             observation=observation_at(observations, i),
             submitted_output=outputs[i], next_observation=observation_at(next_observations, i),
             reward=plain(rewards[i]), done=bool(dones[i]), info=plain(infos[i]),
             fully_captured=True)
        for i in range(len(active)) if active[i]
    ))
