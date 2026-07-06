# Copyright 2026 Tilde Research
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

"""Cross-tokenizer alignment + ULD helpers (one-time Nitrobrew hack).

Ported, with light edits, from TRL's GOLD trainer (``ULDLoss``). These are
pure-torch / pure-python helpers used by the cross-tokenizer branch of
``verl.trainer.distillation.fsdp.nitrobrew_loss.compute_nitrobrew_kl``:

- ``align_by_byte_offsets``: dynamic-programming-free span alignment that walks
  two byte-offset arrays and closes a group whenever both tokenizers agree on a
  byte boundary. This is GOLD's ``_align_by_byte_offsets``.
- ``byte_offsets_from_token_ids``: completion-relative UTF-8 byte ``(start, end)``
  per *student* token, derived from the actual generated ids (so they line up
  1:1 with the student logits the actor produces).
- ``uld_sorted_l1_per_group``: the Universal Logit Distillation loss -- sort each
  side's probability vector descending, pad to the larger vocab, and take the L1
  distance. Vocab-size / vocab-identity agnostic, which is what makes it work
  across tokenizers.

Simplification vs full GOLD: multi-token groups use the first position's
distribution (no chain-rule merge), which avoids transporting the *other* side's
token ids into the loss. Fine for a one-time hack; revisit for higher fidelity.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def align_by_byte_offsets(
    s_offsets: list[tuple[int, int]],
    t_offsets: list[tuple[int, int]],
) -> tuple[list[list[int]], list[list[int]]]:
    """Group student/teacher token indices at shared byte boundaries.

    Walk both byte-offset arrays, advancing the side whose current token ends
    earlier. A group closes when both sides reach the same byte boundary -- the
    points where the two tokenizers agree on a split.
    """
    s_groups: list[list[int]] = []
    t_groups: list[list[int]] = []
    s_start = t_start = s = t = 0
    n_s, n_t = len(s_offsets), len(t_offsets)
    while s < n_s and t < n_t:
        s_end, t_end = s_offsets[s][1], t_offsets[t][1]
        if s_end < t_end:
            s += 1
        elif s_end > t_end:
            t += 1
        else:
            s += 1
            t += 1
            s_groups.append(list(range(s_start, s)))
            t_groups.append(list(range(t_start, t)))
            s_start, t_start = s, t
    if s < n_s or t < n_t:
        s_tail = list(range(s_start, n_s))
        t_tail = list(range(t_start, n_t))
        if s_tail and t_tail:
            s_groups.append(s_tail)
            t_groups.append(t_tail)
        elif s_groups:
            # One side ran out exactly at a shared boundary (byte totals can
            # drift when multi-byte chars split across tokens). Fold the
            # leftover into the last group instead of emitting an empty,
            # unpaired group -- merge uses the first position, so this is a
            # no-op for the loss but keeps the group pairing valid.
            s_groups[-1].extend(s_tail)
            t_groups[-1].extend(t_tail)
        # else: nothing aligned at all; drop the leftover (caller skips the sample).
    return s_groups, t_groups


def byte_offsets_from_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: list[int],
) -> list[tuple[int, int]]:
    """Completion-relative UTF-8 byte ``(start, end)`` for each token id.

    Derived by walking per-token decoded byte lengths so the offsets line up 1:1
    with ``token_ids`` (and therefore with the student logits). The byte string
    these offsets index into equals ``tokenizer.decode(token_ids)`` -- the same
    text the teacher re-tokenizes, so both sides share a byte coordinate system.

    NOTE: per-token decode is exact for ASCII / byte-level-BPE text (e.g. gsm8k
    math). Multi-byte chars split across tokens can drift by a few bytes; the
    alignment degrades gracefully (a slightly larger merged group) rather than
    crashing.
    """
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for tid in token_ids:
        piece = tokenizer.decode(
            [tid], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        nbytes = len(piece.encode("utf-8"))
        offsets.append((cursor, cursor + nbytes))
        cursor += nbytes
    return offsets


def char_offsets_to_byte_offsets(
    text: str,
    char_offsets: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Map fast-tokenizer char ``(start, end)`` offsets to UTF-8 byte offsets."""
    # Prefix byte length at each char index. prefix_bytes[i] = bytes of text[:i].
    prefix_bytes = [0] * (len(text) + 1)
    acc = 0
    for i, ch in enumerate(text):
        acc += len(ch.encode("utf-8"))
        prefix_bytes[i + 1] = acc
    out: list[tuple[int, int]] = []
    for cs, ce in char_offsets:
        # Special tokens / unmapped pieces report (0, 0); keep them zero-width.
        if ce <= cs:
            out.append((0, 0))
        else:
            out.append((prefix_bytes[cs], prefix_bytes[ce]))
    return out


def merge_groups_first_position(
    probs: torch.Tensor,
    groups: list[list[int]],
) -> torch.Tensor:
    """Reduce per-position probs to per-group probs using the first position.

    ``probs``: ``[n_positions, vocab]``. Returns ``[n_groups, vocab]``. This is
    the simplified (no chain-rule) merge described in the module docstring.
    """
    if not groups:
        return probs[:0]
    first_idx = torch.tensor([g[0] for g in groups], device=probs.device, dtype=torch.long)
    return probs.index_select(0, first_idx)


def uld_sorted_l1_per_group(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
) -> torch.Tensor:
    """Universal Logit Distillation loss per aligned group.

    Sort each row descending, right-pad the narrower vocab with zeros, and take
    the per-row L1 distance. Returns ``[n_groups]`` (one scalar loss per group),
    differentiable wrt ``student_probs``.
    """
    if student_probs.shape[0] == 0 or teacher_probs.shape[0] == 0:
        return student_probs.new_zeros(student_probs.shape[0])

    student_sorted = student_probs.sort(dim=-1, descending=True).values
    teacher_sorted = teacher_probs.sort(dim=-1, descending=True).values

    v_s = student_sorted.size(-1)
    v_t = teacher_sorted.size(-1)
    v_max = max(v_s, v_t)
    if v_s < v_max:
        student_sorted = F.pad(student_sorted, (0, v_max - v_s))
    if v_t < v_max:
        teacher_sorted = F.pad(teacher_sorted, (0, v_max - v_t))

    return (student_sorted - teacher_sorted).abs().sum(dim=-1)


def pad_byte_offsets(
    offsets: list[tuple[int, int]],
    target_len: int,
) -> torch.Tensor:
    """Right-pad a list of ``(start, end)`` offsets to ``[target_len, 2]`` with (0, 0)."""
    out = torch.zeros(target_len, 2, dtype=torch.long)
    n = min(len(offsets), target_len)
    if n > 0:
        out[:n] = torch.tensor(offsets[:n], dtype=torch.long)
    return out


def select_completion_positions(byte_offsets: torch.Tensor) -> torch.Tensor:
    """Boolean mask of completion tokens: end-byte > 0 (prompt/pad are (0, 0))."""
    return byte_offsets[:, 1] > 0


__all__ = [
    "align_by_byte_offsets",
    "byte_offsets_from_token_ids",
    "char_offsets_to_byte_offsets",
    "merge_groups_first_position",
    "uld_sorted_l1_per_group",
    "pad_byte_offsets",
    "select_completion_positions",
]
