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
"""Preprocess the DAPO-Math-17k dataset into verl-compatible train/test parquet.

``BytedTsinghua-SIA/DAPO-Math-17k`` (config ``default``) is already stored in
verl's RL schema (``prompt`` / ``reward_model.ground_truth`` / ``extra_info`` /
``data_source``), but it ships a single ``train`` split. We deterministically
carve off a small held-out ``test`` split so the distillation trainer has
something to evaluate against (``trainer.test_freq``).

Example::

    python3 examples/data_preprocess/dapo_math_17k.py \
        --local_save_dir ~/data/dapo-math-17k --test_size 512
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

DATA_SOURCE = "BytedTsinghua-SIA/DAPO-Math-17k"


def make_map_fn(split):
    def process_fn(example, idx):
        # The rows are already in verl format; just (re)stamp split bookkeeping
        # in extra_info without clobbering the existing fields.
        extra_info = dict(example.get("extra_info") or {})
        extra_info["split"] = split
        extra_info.setdefault("index", idx)
        example["extra_info"] = extra_info
        return example

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS dir to copy the output to.")
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="Local path to the raw HF dataset, if already downloaded.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/dapo-math-17k",
        help="Directory to write train.parquet / test.parquet.",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=512,
        help="Number of examples to hold out for the test split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for the train/test split.")
    parser.add_argument(
        "--dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Deduplicate by prompt. The 'default' config repeats each of the ~17k unique "
            "problems ~100x (~1.79M rows); dedup keeps one row per problem so startup "
            "filtering is fast and epochs are sane. Pass --no-dedup to keep all rows."
        ),
    )

    args = parser.parse_args()

    source = args.local_dataset_path if args.local_dataset_path is not None else DATA_SOURCE
    dataset = datasets.load_dataset(source, "default")

    train_raw = dataset["train"]

    if args.dedup:
        seen: set[str] = set()
        keep_idx: list[int] = []
        for i, prompt in enumerate(train_raw["prompt"]):
            key = repr(prompt)
            if key not in seen:
                seen.add(key)
                keep_idx.append(i)
        before = len(train_raw)
        train_raw = train_raw.select(keep_idx)
        print(f"dedup: {before} -> {len(train_raw)} unique prompts")

    # DAPO-Math-17k only provides a train split; split it deterministically.
    split = train_raw.train_test_split(test_size=args.test_size, seed=args.seed, shuffle=True)
    train_dataset = split["train"].map(function=make_map_fn("train"), with_indices=True)
    test_dataset = split["test"].map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    print(f"Wrote {len(train_dataset)} train / {len(test_dataset)} test rows to {local_save_dir}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
