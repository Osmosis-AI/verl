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

"""Nitrobrew teacher infrastructure (vLLM AsyncLLM pooling backend).

Used when ``distillation.distillation_loss.loss_mode == "nitrobrew"``.

Architecture::

                      +----------------------------------+
    AgentLoopWorker   |  NitrobrewAsyncTeacherManager    |
                      |  (round-robin Ray remote calls)  |
                      +-----------------+----------------+
                                        | compute_hidden_states.remote(seq_ids)
                      +-----------------v----------------+
                      |  NitrobrewTeacherWorker (Ray)    |
                      |  - vLLM AsyncLLM (pooling+embed) |
                      |  - PoolingType.ALL -> [S, D]     |
                      |  - on-actor [S, D] @ P_down      |
                      |    -> [S, d_comp]                |
                      +----------------------------------+

Per-teacher SVD: ``W_T (lm_head) ~= W_up @ P_down.T`` with
``W_up [V, d_comp]`` (sent to the actor for student-side logit
reconstruction) and ``P_down [D, d_comp]`` (held by every teacher worker for
on-device projection of hidden states before the Ray RPC boundary).

Loading ``lm_head`` directly from safetensors avoids constructing a full HF
model on the driver just to read a single matrix.

Why this bypasses :class:`~verl.workers.rollout.replica.RolloutReplica` (the
shared rollout/teacher abstraction used by the topk teacher in
:mod:`~verl.experimental.teacher_loop.teacher_model`):

1. vLLM's HTTP server (``/v1/embeddings``) returns one vector per *sequence*,
   not per *token*. The ``runner="pooling"`` + ``convert="embed"`` +
   ``PoolingParams(task="token_embed")`` (ALL pool) path is only reachable
   via :meth:`AsyncLLM.encode` directly, so the standard ``vLLMHttpServer``
   would need a custom RPC method to expose it.
2. The ``P_down @ h`` projection from ``[S, D]`` to ``[S, d_comp]`` must run
   on the actor *before* the network boundary -- otherwise the wire payload
   is ``S * D``, defeating the whole nitrobrew communication-budget claim.
3. Frozen teachers don't need the ``CheckpointEngineWorker`` weight-sync
   stack that ``RolloutReplica.init_colocated`` spawns under the hood.

TODO(nitrobrew): factor into a ``NitrobrewReplica(RolloutReplica)`` once
vLLM exposes per-token embeddings via the HTTP server, so this module can
collapse to a ``launch_servers`` override + a tiny ``NitrobrewServer`` actor.
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Optional

import ray
import torch

logger = logging.getLogger(__name__)


def _load_lm_head_weight(model_path: str) -> torch.Tensor:
    """Read the teacher's lm_head (or tied embed_tokens) weight from safetensors.

    Returns a CPU fp32 tensor of shape ``[V, D]``. Avoids constructing a full
    ``AutoModelForCausalLM`` on the driver.
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig

    from verl.utils.fs import copy_to_local

    # copy_to_local resolves HDFS paths and returns local paths unchanged. It
    # does not fetch HF Hub IDs (those still need snapshot_download below).
    resolved = copy_to_local(model_path)
    folder = (
        resolved
        if os.path.isdir(resolved)
        else snapshot_download(resolved, allow_patterns=["*.safetensors*", "*.json"])
    )

    cfg = AutoConfig.from_pretrained(folder, trust_remote_code=True)
    target_keys = (
        ["model.embed_tokens.weight", "embed_tokens.weight"] if cfg.tie_word_embeddings else ["lm_head.weight"]
    )

    index_path = os.path.join(folder, "model.safetensors.index.json")
    shard_for_key: dict[str, str] = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            shard_for_key = json.load(f)["weight_map"]

    for key in target_keys:
        if shard_for_key:
            shard = shard_for_key.get(key)
            if shard is None:
                continue
            tensors = load_file(os.path.join(folder, shard))
        else:
            tensors = load_file(os.path.join(folder, "model.safetensors"))
        if key in tensors:
            return tensors[key].float()

    raise ValueError(f"Could not find lm_head/embed_tokens weight in {model_path}")


def _compute_svd(w_t: torch.Tensor, d_comp: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncated SVD of the teacher's unembed: ``w_t ~= w_up @ p_down.T``.

    Returns ``(w_up [V, d_comp], p_down [D, d_comp])`` cast to ``dtype``.

    When ``d_comp >= D`` (the teacher hidden size) no compression is requested,
    so the SVD round-trip is an exact identity. Skip the (expensive) full SVD
    and return ``w_up = w_t`` with an identity ``p_down``: hidden states pass
    through unprojected and the student reconstructs the exact teacher logits.
    The returned ``p_down`` then has width ``D`` (not the requested ``d_comp``);
    callers should derive the effective ``d_comp`` from ``p_down.shape[1]``.
    """
    d = w_t.shape[1]
    if d_comp >= d:
        w_up = w_t.to(dtype)
        p_down = torch.eye(d, dtype=dtype)
        return w_up, p_down

    u, sigma, vh = torch.linalg.svd(w_t, full_matrices=False)
    u_r = u[:, :d_comp]
    sigma_r = sigma[:d_comp]
    vh_r = vh[:d_comp, :]
    w_up = (u_r * sigma_r.unsqueeze(0)).to(dtype)
    p_down = vh_r.T.to(dtype)
    return w_up, p_down


@ray.remote
class NitrobrewTeacherWorker:
    """vLLM-backed teacher serving PCA-compressed hidden states.

    Wraps vLLM's :class:`AsyncLLM` in pooling+embed mode with per-token output
    (``PoolingType.ALL``). Hidden states are projected on the actor's GPU to
    ``d_comp`` *before* crossing the Ray RPC boundary, so the network payload
    matches nitrobrew's communication budget regardless of teacher ``D``.
    """

    async def setup(
        self,
        model_path: str,
        d_comp: int,
        p_down_list: list,
        dtype: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_num_seqs: int,
        max_model_len: int,
        enforce_eager: bool,
    ) -> None:
        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        torch_dtype = getattr(torch, dtype)
        self._d_comp = d_comp
        self._dtype = torch_dtype
        self._max_model_len = max_model_len
        self._p_down_cpu = torch.tensor(p_down_list, dtype=torch_dtype)  # [D, d_comp]
        self._p_down_dev: dict[torch.device, torch.Tensor] = {}

        logger.warning(
            "NitrobrewTeacherWorker: launching AsyncLLM (model=%s, tp=%d, max_num_seqs=%d, "
            "max_model_len=%d, gmu=%.2f, enforce_eager=%s, dtype=%s)",
            model_path,
            tensor_parallel_size,
            max_num_seqs,
            max_model_len,
            gpu_memory_utilization,
            enforce_eager,
            dtype,
        )

        # ALL pooling cannot resume across steps, so prefill must complete in a
        # single scheduler step. Cap per-step token budget at 4x max_model_len:
        # enough to batch a few full-length sequences per wave without
        # exhausting KV-cache headroom during vLLM's memory profile pass.
        max_num_batched_tokens = min(max_num_seqs, 4) * max_model_len
        engine_args = AsyncEngineArgs(
            model=model_path,
            runner="pooling",
            convert="embed",
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            enforce_eager=enforce_eager,
            # ALL pooling rejects partial prefills; chunked prefill and prefix
            # caching can both produce them, so disable both.
            enable_chunked_prefill=False,
            enable_prefix_caching=False,
            disable_log_stats=True,
        )
        self._engine = AsyncLLM.from_engine_args(engine_args)
        logger.warning("NitrobrewTeacherWorker: AsyncLLM ready")

    def _p_down_on(self, device: torch.device) -> torch.Tensor:
        cached = self._p_down_dev.get(device)
        if cached is None:
            cached = self._p_down_cpu.to(device, non_blocking=True)
            self._p_down_dev[device] = cached
        return cached

    async def compute_hidden_states(self, sequence_ids: list[int]) -> list:
        """Encode one sequence; return PCA-compressed hidden states ``[S, d_comp]``."""
        from vllm import PoolingParams
        from vllm.inputs import TokensPrompt

        # Cross-tokenizer sequences are re-tokenized with the teacher's tokenizer
        # and chat template, so they can exceed the student-derived context budget
        # by a few tokens. Truncate the tail instead of crashing the run: the
        # byte-offset alignment degrades gracefully when the teacher side is
        # shorter (callers size byte offsets to the returned hidden states).
        if len(sequence_ids) > self._max_model_len:
            logger.warning(
                "NitrobrewTeacherWorker: truncating sequence from %d to max_model_len=%d tokens",
                len(sequence_ids),
                self._max_model_len,
            )
            sequence_ids = sequence_ids[: self._max_model_len]

        params = PoolingParams(task="token_embed", use_activation=False)
        prompt = TokensPrompt(prompt_token_ids=sequence_ids)

        final = None
        async for out in self._engine.encode(
            prompt=prompt,
            pooling_params=params,
            request_id=uuid.uuid4().hex,
        ):
            final = out
        assert final is not None, "AsyncLLM.encode produced no output"

        h = final.outputs.data  # [S, D]
        p = self._p_down_on(h.device)
        z = torch.mm(h.to(self._dtype), p)  # [S, d_comp]
        return z.cpu().tolist()


class NitrobrewAsyncTeacherManager:
    """Async client that round-robins hidden-state requests across workers.

    Mirrors the interface of ``AsyncTeacherLLMServerManager`` but returns
    teacher hidden states instead of ``(teacher_ids, teacher_logprobs)``.

    CROSS-TOKENIZER (one-time hack)
    -------------------------------
    The same-tokenizer path (``compute_teacher_hidden_states_single``) feeds the
    teacher the *student's* token ids -- meaningless when the tokenizers differ.
    For cross-tokenizer distillation, pass ``teacher_tokenizer_paths`` and call
    ``compute_teacher_uld_single`` instead: it re-tokenizes the student's decoded
    text with the TEACHER's own tokenizer, runs the teacher on those ids, and
    returns ``(hidden_states, teacher_byte_offsets)``. The byte offsets are
    completion-relative (prompt positions zeroed) so the loss can line the teacher
    tokens up against the student tokens by shared byte boundaries.
    """

    def __init__(
        self,
        worker_handles: dict[str, list[Any]],
        teacher_tokenizer_paths: Optional[dict[str, str]] = None,
    ):
        self._worker_handles = worker_handles
        self._counters = {key: 0 for key in worker_handles}
        self._lock = asyncio.Lock()

        # Cross-tokenizer: load one teacher tokenizer per routing key (CPU, in the
        # AgentLoopWorker process). Lazy import keeps the non-cross-tokenizer path
        # free of the transformers tokenizer dependency at construction time.
        self._teacher_tokenizers: dict[str, Any] = {}
        if teacher_tokenizer_paths:
            from transformers import AutoTokenizer

            for key, path in teacher_tokenizer_paths.items():
                if path is None:
                    continue
                tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                if not tok.is_fast:
                    raise ValueError(
                        f"Cross-tokenizer distillation needs a fast teacher tokenizer "
                        f"(byte offsets) for {key!r} ({path})."
                    )
                self._teacher_tokenizers[key] = tok

    def _resolve_key(self, routing_key: Optional[str]) -> str:
        if len(self._worker_handles) == 1:
            return next(iter(self._worker_handles))
        if routing_key is None or routing_key not in self._worker_handles:
            raise ValueError(
                f"Routing key {routing_key!r} not found in nitrobrew workers: {sorted(self._worker_handles)}"
            )
        return routing_key

    async def compute_teacher_hidden_states_single(
        self,
        sequence_ids: list[int],
        routing_key: Optional[str] = None,
    ) -> torch.Tensor:
        key = self._resolve_key(routing_key)
        handles = self._worker_handles[key]
        async with self._lock:
            idx = self._counters[key] % len(handles)
            self._counters[key] += 1

        result: list = await handles[idx].compute_hidden_states.remote(sequence_ids)
        return torch.tensor(result, dtype=torch.bfloat16)  # [S, d_comp]

    async def compute_teacher_uld_single(
        self,
        prompt_text: str,
        completion_text: str,
        routing_key: Optional[str] = None,
        prompt_messages: Optional[list] = None,
        enable_thinking: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross-tokenizer (one-time hack) teacher signal.

        Re-tokenizes the prompt + ``completion_text`` with the *teacher* tokenizer,
        runs the teacher to get per-token hidden states, and returns
        ``(hidden_states [S_t, d_comp], teacher_byte_offsets [S_t, 2])``. Byte
        offsets are completion-relative: prompt positions are ``(0, 0)`` so the
        loss can pick out completion tokens by ``end > 0``.

        Thinking: the prompt only *conditions* the teacher (its positions are masked
        out of the loss), so when ``prompt_messages`` (the raw chat turns) are given
        we build the prompt with the TEACHER's own chat template and
        ``enable_thinking=True``. That puts e.g. Qwen3 into reasoning mode, so its
        next-token distribution favours ``<think>`` -- which is what the student
        learns to imitate on-policy. Without this, the teacher just continues the
        Llama-formatted text and never thinks, so no thinking can be distilled.
        """
        from verl.trainer.distillation.fsdp.uld_align import (
            char_offsets_to_byte_offsets,
            pad_byte_offsets,
        )

        key = self._resolve_key(routing_key)
        tok = self._teacher_tokenizers.get(key)
        if tok is None:
            raise ValueError(
                f"No teacher tokenizer loaded for {key!r}; pass teacher_tokenizer_paths "
                "to NitrobrewAsyncTeacherManager for cross-tokenizer distillation."
            )

        # Prompt only conditions the teacher; its positions are masked out of the loss.
        if prompt_messages is not None:
            # Use the teacher's own chat template (+ thinking) so the teacher is in
            # reasoning mode. enable_thinking is Qwen3-specific; fall back if the
            # template doesn't accept it.
            messages = [dict(m) for m in prompt_messages]
            try:
                prompt_ids = tok.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, enable_thinking=enable_thinking
                )
            except TypeError:
                prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        else:
            prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        comp_enc = tok(completion_text, add_special_tokens=False, return_offsets_mapping=True)
        completion_ids = comp_enc["input_ids"]
        comp_byte_offsets = char_offsets_to_byte_offsets(completion_text, comp_enc["offset_mapping"])

        sequence_ids = list(prompt_ids) + list(completion_ids)
        byte_offsets = [(0, 0)] * len(prompt_ids) + comp_byte_offsets

        handles = self._worker_handles[key]
        async with self._lock:
            idx = self._counters[key] % len(handles)
            self._counters[key] += 1

        result: list = await handles[idx].compute_hidden_states.remote(sequence_ids)
        hidden = torch.tensor(result, dtype=torch.bfloat16)  # [S_t, d_comp]
        offsets = pad_byte_offsets(byte_offsets, hidden.shape[0])  # [S_t, 2]
        return hidden, offsets


class NitrobrewTeacherModelManager:
    """Manage a pool of vLLM-backed :class:`NitrobrewTeacherWorker` Ray actors.

    Exposes:
      - ``worker_handles: dict[teacher_key, list[ActorHandle]]`` for
        :class:`AgentLoopWorker`.
      - ``w_up: torch.Tensor [V, d_comp]`` for actor-side
        ``set_teacher_unembed``.

    ``nitrobrew_d_comp`` and the inference settings (TP / GMU /
    max_num_seqs / max_model_len / enforce_eager / dtype) are read from each
    teacher's :class:`DistillationTeacherModelConfig`.
    """

    def __init__(
        self,
        teacher_model_configs: dict,
        gpus_per_replica: int,
    ):
        self.worker_handles: dict[str, list] = {}
        self.w_up: Optional[torch.Tensor] = None

        for key, teacher_cfg in teacher_model_configs.items():
            d_comp = teacher_cfg.nitrobrew_d_comp
            if d_comp is None:
                raise ValueError(
                    f"teacher_models['{key}'].nitrobrew_d_comp must be set when using the nitrobrew loss mode."
                )

            num_replicas = teacher_cfg.num_replicas
            model_path = teacher_cfg.model_path
            inference = teacher_cfg.inference
            dtype = inference.dtype
            tp = inference.tensor_model_parallel_size
            gmu = inference.gpu_memory_utilization
            max_num_seqs = inference.max_num_seqs
            enforce_eager = inference.enforce_eager
            # validate_and_prepare_for_distillation rewrites prompt_length to
            # (prompt + response) and response_length to 1, so their sum is the
            # required teacher context.
            max_model_len = inference.max_model_len or (inference.prompt_length + inference.response_length)

            torch_dtype = getattr(torch, dtype)

            logger.warning(
                "NitrobrewTeacherModelManager: loading lm_head for '%s' (%s) (requested d_comp=%d)",
                key,
                model_path,
                d_comp,
            )
            w_t = _load_lm_head_weight(model_path)
            hidden_dim = w_t.shape[1]
            skip_svd = d_comp >= hidden_dim
            if skip_svd:
                logger.warning(
                    "NitrobrewTeacherModelManager: d_comp=%d >= hidden_dim=%d, skipping SVD "
                    "(identity projection, exact teacher logits)",
                    d_comp,
                    hidden_dim,
                )
            else:
                logger.warning("NitrobrewTeacherModelManager: computing SVD (d_comp=%d)", d_comp)
            w_up, p_down = _compute_svd(w_t, d_comp, torch_dtype)
            del w_t
            # _compute_svd may clamp the identity path to the full hidden dim, so
            # keep d_comp in sync with the projection width sent to the workers.
            d_comp = p_down.shape[1]
            logger.warning(
                "NitrobrewTeacherModelManager: projection ready w_up %s, p_down %s (d_comp=%d)",
                tuple(w_up.shape),
                tuple(p_down.shape),
                d_comp,
            )

            handles = []
            for _ in range(num_replicas):
                worker = NitrobrewTeacherWorker.options(
                    num_gpus=gpus_per_replica,
                    max_concurrency=max(max_num_seqs * 2, 256),
                ).remote()
                handles.append(worker)
            self.worker_handles[key] = handles

            p_down_list = p_down.cpu().tolist()
            ray.get(
                [
                    h.setup.remote(
                        model_path,
                        d_comp,
                        p_down_list,
                        dtype,
                        tp,
                        gmu,
                        max_num_seqs,
                        max_model_len,
                        enforce_eager,
                    )
                    for h in handles
                ]
            )

            if self.w_up is None:
                self.w_up = w_up

        assert self.w_up is not None, "No teacher models configured."
