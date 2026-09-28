# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create the modality/size placeholder parquet used by agent environments.

The benchmark environments replace the empty prompt during rollout; Geometry3K
was previously downloaded only to obtain rows with the right split sizes.  A
local Dataset is equivalent for text-agent training and keeps WebShop startup
fully offline.
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='visual', choices=['visual', 'text'])
    parser.add_argument('--local_dir', default=os.path.join(os.environ.get('DATA_ROOT', 'data'), 'verl-agent'))
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_data_size', default=256, type=int)
    parser.add_argument('--val_data_size', default=256, type=int)

    args = parser.parse_args()
    print(f"processing data for mode: {args.mode}")
    args.local_dir = os.path.join(args.local_dir, args.mode)

    instruction_following = {
        "visual": "<image>",
        "text": "",
        }

    def make_placeholder_dataset(split, size):
        prompt = instruction_following[args.mode]
        rows = []
        for index in range(size):
            row = {
                "data_source": args.mode,
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "agent",
                "extra_info": {"split": split, "index": index},
            }
            rows.append(row)
        return datasets.Dataset.from_list(rows)

    if args.mode == "text":
        train_dataset = make_placeholder_dataset("train", args.train_data_size)
        test_dataset = make_placeholder_dataset("test", args.val_data_size)
    else:
        # Preserve the original image-backed behavior for visual agents. The
        # offline placeholder is intentionally scoped to text environments.
        source = datasets.load_dataset("hiyouga/geometry3k")

        def prepare_visual(example, index, split):
            return {
                "data_source": args.mode,
                "prompt": [{"role": "user", "content": instruction_following[args.mode]}],
                "images": example["images"],
                "ability": "agent",
                "extra_info": {"split": split, "index": index},
            }

        train_dataset = source["train"].select(range(args.train_data_size)).map(
            prepare_visual,
            with_indices=True,
            fn_kwargs={"split": "train"},
            remove_columns=source["train"].column_names,
            num_proc=8,
        )
        test_dataset = source["test"].select(range(args.val_data_size)).map(
            prepare_visual,
            with_indices=True,
            fn_kwargs={"split": "test"},
            remove_columns=source["test"].column_names,
            num_proc=8,
        )

    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
