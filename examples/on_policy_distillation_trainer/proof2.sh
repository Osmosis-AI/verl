#!/usr/bin/env bash
# Cross-tokenizer (one-time Nitrobrew hack) on-policy distillation.
#
# Teacher: Qwen/Qwen3-4B (hidden_dim 2560, vocab ~151936)
# Student: meta-llama/Llama-3.2-3B-Instruct (hidden_dim 3072, vocab ~128256)
#
# Distills across tokenizers by hijacking the nitrobrew hidden-states path:
# the teacher reconstructs dense full-vocab logits on the student device, then a
# byte-offset span alignment + Universal Logit Distillation (sorted-L1) loss
# compares the two distributions without a shared vocabulary. See
# verl/trainer/distillation/fsdp/nitrobrew_loss.py (_compute_cross_tokenizer_uld).
#
# Notes / guardrails (enforced in code):
#   - FSDP only; ulysses_sequence_parallel_size MUST be 1.
#   - use_policy_gradient=False (supervised backprop) and use_task_rewards=False.
#   - use_remove_padding=True is REQUIRED: the distillation logits-processor only
#     runs on the remove-padding path.
#   - nitrobrew_d_comp=2560 == Qwen3-4B hidden_dim -> identity SVD -> exact teacher
#     logits (no PCA error in the distillation target).
#   - Llama-3.2-3B-Instruct is gated on HF: export HF_TOKEN before running.
set -xeuo pipefail

############################ Quick Config ############################

ROLLOUT_NAME="vllm" # sglang or vllm

STUDENT_MODEL=meta-llama/Llama-3.2-3B-Instruct
TEACHER_MODEL=Qwen/Qwen3-4B-Instruct-2507
TEACHER_TOKENIZER=Qwen/Qwen3-4B-Instruct-2507
TEACHER_HIDDEN_DIM=2560 # Qwen3-4B hidden size -> identity projection (exact logits)

DISTILLATION_LOSS_MODE="nitrobrew"
USE_POLICY_GRADIENT=False
USE_FUSED_KERNELS=False

PROJECT_NAME='verl_cross_tokenizer_distillation_gsm8k'

MAX_PROMPT=512
MAX_RESPONSE_LENGTH=4096
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))
# The teacher re-tokenizes the student's text with its OWN tokenizer + chat
# template (thinking mode), which can need more tokens than the student budget.
# Give it headroom; the worker also truncates as a last resort.
TEACHER_MAX_NUM_TOKENS=$(( MAX_NUM_TOKENS + 512 ))
TRAIN_PROMPT_BSZ=64
STUDENT_MICRO_BATCH_SIZE_PER_GPU=1
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
USE_DYNAMIC_BSZ=True

STUDENT_WORLD_SIZE=4
TEACHER_WORLD_SIZE=4

# Cross-tokenizer alignment runs a per-sample Python loop; sequence parallel must
# be 1 (enforced by the loss).
SP=1

EXP_NAME="fsdp/xtok/student-Llama-3.2-1B-Instruct/teacher-Qwen3-4B-Instruct-2507/loss-${DISTILLATION_LOSS_MODE}"

ENFORCE_EAGER=False # true for faster debugging

############################ Paths ############################

gsm8k_train_path=/root/gsm8k/train.parquet
gsm8k_test_path=/root/gsm8k/test.parquet

TRAIN_FILES="['$gsm8k_train_path']"
TEST_FILES="['$gsm8k_test_path']"

############################ Parameter Groups ############################

DATA=(
    data.train_files="$TRAIN_FILES"
    data.val_files="$TEST_FILES"
    data.max_prompt_length=$MAX_PROMPT
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.train_batch_size=$TRAIN_PROMPT_BSZ
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    # REQUIRED: the distillation logits-processor only runs on the remove-padding path.
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=$USE_FUSED_KERNELS
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER
)

DISTILLATION=(
    distillation.enabled=True
    distillation.n_gpus_per_node=$TEACHER_WORLD_SIZE
    distillation.nnodes=1
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}"
    +distillation.teacher_models.teacher_model.teacher_tokenizer_path="${TEACHER_TOKENIZER}"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1
    distillation.teacher_models.teacher_model.inference.name=$ROLLOUT_NAME
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.8
    distillation.teacher_models.teacher_model.inference.enforce_eager=$ENFORCE_EAGER
    distillation.teacher_models.teacher_model.inference.max_model_len=$TEACHER_MAX_NUM_TOKENS
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=$TEACHER_MAX_NUM_TOKENS
    distillation.teacher_models.teacher_model.inference.max_num_seqs=$TEACHER_MAX_NUM_TOKENS
    +distillation.teacher_models.teacher_model.nitrobrew_d_comp=$TEACHER_HIDDEN_DIM
    distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
    distillation.distillation_loss.cross_tokenizer=True
    distillation.distillation_loss.uld_student_temperature=1.0
    distillation.distillation_loss.uld_teacher_temperature=1.0
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=$USE_POLICY_GRADIENT
)

STUDENT=(
    actor_rollout_ref.actor.optim.lr=3e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_BSZ
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3
    actor_rollout_ref.rollout.calculate_log_probs=False
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.n=1
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXP_NAME
    trainer.n_gpus_per_node=$STUDENT_WORLD_SIZE
    trainer.nnodes=1
    # gsm8k is ~116 steps at train_batch_size=64 for 1 epoch, so save_freq must be
    # < steps/epoch or nothing ever checkpoints. Saves at steps 50 and 100.
    trainer.save_freq=10
    trainer.test_freq=1
    trainer.total_epochs=1
    trainer.val_before_train=True
    trainer.resume_mode=disable
    trainer.log_val_generations=1
)

############################ Launch ############################

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${DISTILLATION[@]}" \
    "${ROLLOUT[@]}" \
    "${STUDENT[@]}" \
    "${TRAINER[@]}" \
    "$@"
