# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Gulp AI Inc. (Osmosis AI)
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
"""RemoteAgentLoop for delegating trajectory generation to external RolloutServer.

This agent loop is a drop-in replacement for local agent loops (SingleTurnAgentLoop,
ToolAgentLoop) that delegates the actual rollout execution to an external RolloutServer
while keeping LLM inference on the training cluster.

Key Features:
- Registered as @register("remote_agent") in verl's agent loop registry
- Shared RolloutClient with connection pooling across all instances
- Concurrency control via semaphore
- Pre-flight health check for early error detection
- Graceful degradation on rollout failures

See design doc section 4.3 for specifications.
"""

import asyncio
import logging
import os
from typing import Any, ClassVar, Optional

import httpx
from omegaconf import DictConfig
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.workers.rollout.remote.rollout_client import (
    RolloutClient,
    RolloutClientError,
)
from verl.workers.rollout.remote.schemas import (
    Message,
    RolloutOutput,
    RolloutRequest,
    SamplingParams,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("remote_agent")
class RemoteAgentLoop(AgentLoopBase):
    """Agent loop that delegates rollout to external RolloutServer.

    This class manages communication with an external RolloutServer that handles
    the agent loop logic (state machine, tool orchestration). LLM inference
    still happens on the training cluster via LLMProxy.

    Class-level shared state is initialized once per worker and reused across
    all instances within that worker.
    """

    # Class-level shared state (initialized once per worker)
    rollout_client: ClassVar[Optional[RolloutClient]] = None
    remote_config: ClassVar[Optional[DictConfig]] = None
    semaphore: ClassVar[Optional[asyncio.Semaphore]] = None
    tokenizer: ClassVar[Optional[AutoTokenizer]] = None
    _class_initialized: ClassVar[bool] = False

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer, processor: AutoProcessor, **kwargs):
        """Initialize shared RolloutClient (called once per worker).

        Reads configuration from config.actor_rollout_ref.rollout.remote and
        initializes the shared RolloutClient with connection pooling.

        Args:
            config: Trainer configuration (DictConfig from verl)
            tokenizer: Tokenizer for local tokenization (error fallback)
            processor: Processor for multi-modal data (not used in remote mode)
            **kwargs: Additional kwargs from config file
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True

        logger.info("Performing class-level RemoteAgentLoop initialization")

        # Extract remote rollout config
        remote_config = config.actor_rollout_ref.rollout.remote
        cls.remote_config = remote_config

        # Store tokenizer for error fallback path
        cls.tokenizer = tokenizer

        # Initialize RolloutClient with connection pooling
        # Validate and retrieve RolloutServer API key
        api_key = None
        if remote_config.get("api_key_env"):
            env_var_name = remote_config.api_key_env
            api_key = os.environ.get(env_var_name)
            if api_key is None:
                logger.warning(
                    f"API key environment variable '{env_var_name}' is configured "
                    f"but not set. Remote rollout may fail authentication."
                )

        cls.rollout_client = RolloutClient(
            base_url=remote_config.rollout_server_url,
            api_key=api_key,
            timeout_seconds=remote_config.get("timeout_seconds", 300),
            max_retries=remote_config.get("max_retries", 3),
            max_connections=remote_config.get("max_concurrent_rollouts", 50) * 2,
        )

        # Validate LLMProxy API key (if proxy configured)
        if remote_config.get("llm_proxy_url") and remote_config.get("llm_proxy_api_key_env"):
            env_var_name = remote_config.llm_proxy_api_key_env
            llm_api_key = os.environ.get(env_var_name)
            if llm_api_key is None:
                logger.warning(
                    f"LLM Proxy API key environment variable '{env_var_name}' is configured "
                    f"but not set. LLM requests may fail authentication."
                )

        # Concurrency control via semaphore
        max_concurrent = remote_config.get("max_concurrent_rollouts", 50)
        cls.semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(
            f"RemoteAgentLoop initialized: "
            f"rollout_server={remote_config.rollout_server_url}, "
            f"max_concurrent={max_concurrent}"
        )

        # Pre-flight health check: verify connectivity before training starts
        cls._preflight_check(remote_config)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Execute remote rollout and return AgentLoopOutput.

        Builds a RolloutRequest from the dataset sample and configuration,
        sends it to the RolloutServer with concurrency control, and converts
        the response to AgentLoopOutput for training.

        Args:
            sampling_params: LLM sampling parameters (temperature, top_p, etc.)
            **kwargs: Dataset fields including raw_prompt, index, extra_info

        Returns:
            AgentLoopOutput with token sequences and masks for training
        """
        # Build RolloutRequest
        request = self._build_request(sampling_params, **kwargs)

        try:
            # Execute with concurrency control
            async with self.semaphore:
                output = await self.rollout_client.rollout(request)

            # Convert to AgentLoopOutput
            return self._convert_output(output, **kwargs)

        except RolloutClientError as e:
            # Graceful degradation: return empty response with original prompt
            logger.warning(f"Rollout failed for session {request.session_id}: {e}")
            return self._build_error_output(request, str(e), **kwargs)

    def _build_request(self, sampling_params: dict[str, Any], **kwargs) -> RolloutRequest:
        """Build RolloutRequest from dataset sample and config.

        Args:
            sampling_params: LLM sampling parameters
            **kwargs: Dataset fields

        Returns:
            RolloutRequest ready to send to RolloutServer
        """
        # Get LLM proxy API key
        llm_api_key = None
        if self.remote_config.get("llm_proxy_api_key_env"):
            llm_api_key = os.environ.get(self.remote_config.llm_proxy_api_key_env, None)

        # Convert raw_prompt to Message format
        messages = self._convert_messages(kwargs.get("raw_prompt", []))

        return RolloutRequest(
            session_id=self._build_session_id(**kwargs),
            llm_url=self.remote_config.llm_proxy_url,
            llm_api_key=llm_api_key,
            model=self.remote_config.get("model_name", ""),
            messages=messages,
            sampling_params=SamplingParams(
                temperature=sampling_params.get("temperature", 1.0),
                top_p=sampling_params.get("top_p", 1.0),
                max_tokens=sampling_params.get("max_tokens", 512),
                stop=sampling_params.get("stop"),
                logprobs=sampling_params.get("logprobs", 1),
            ),
            tool_server_url=self.remote_config.get("tool_server_url"),
            max_turns=self.remote_config.get("max_turns", 10),
            max_tokens_total=self.remote_config.get("max_tokens_total", 8192),
            metadata={
                "job_id": kwargs.get("extra_info", {}).get("job_id"),
                "step": kwargs.get("extra_info", {}).get("step"),
                "index": kwargs.get("index"),
                "rollout_n": kwargs.get("rollout_n", 0),
            },
        )

    def _convert_output(self, output: RolloutOutput, **kwargs) -> AgentLoopOutput:
        """Convert RolloutOutput to AgentLoopOutput.

        Maps the RolloutServer response to verl's internal AgentLoopOutput format
        for downstream processing by the trainer.

        Args:
            output: RolloutOutput from RolloutServer
            **kwargs: Original dataset fields for extra_fields

        Returns:
            AgentLoopOutput compatible with verl training pipeline
        """
        return AgentLoopOutput(
            prompt_ids=output.prompt_ids,
            response_ids=output.response_ids,
            response_mask=output.response_mask,
            response_logprobs=output.response_logprobs,
            num_turns=output.num_turns,
            metrics=AgentLoopMetrics(
                generate_sequences=output.metrics.llm_latency_ms / 1000.0,
                tool_calls=output.metrics.tool_latency_ms / 1000.0,
            ),
            extra_fields={
                **output.extra_fields,
                "finish_reason": output.finish_reason,
                "success": output.success,
            },
        )

    def _build_error_output(self, request: RolloutRequest, error_message: str, **kwargs) -> AgentLoopOutput:
        """Build AgentLoopOutput for failed rollouts.

        When RolloutServer fails, we return an output with the original prompt
        preserved and empty response. This allows training to continue with
        degraded data rather than failing entirely.

        Args:
            request: The original RolloutRequest that failed
            error_message: Error description for logging
            **kwargs: Original dataset fields

        Returns:
            AgentLoopOutput with preserved prompt_ids and empty response
        """
        # Get original prompt IDs using local tokenizer
        prompt_ids = self._get_original_prompt_ids(**kwargs)

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=[],  # Empty response for failed rollout
            response_mask=[],
            response_logprobs=None,
            num_turns=0,
            metrics=AgentLoopMetrics(
                generate_sequences=0.0,
                tool_calls=0.0,
            ),
            extra_fields={
                "finish_reason": "error",
                "success": False,
                "error_message": error_message,
            },
        )

    def _build_session_id(self, **kwargs) -> str:
        """Build unique session ID for tracking.

        Format: {job_id}-step{step}-idx{index}-n{rollout_n}

        Args:
            **kwargs: Dataset fields containing extra_info, index, rollout_n

        Returns:
            Unique session identifier string
        """
        job_id = kwargs.get("extra_info", {}).get("job_id", "unknown")
        step = kwargs.get("extra_info", {}).get("step", 0)
        index = kwargs.get("index", 0)
        rollout_n = kwargs.get("rollout_n", 0)
        return f"{job_id}-step{step}-idx{index}-n{rollout_n}"

    def _convert_messages(self, raw_prompt: list[dict[str, Any]]) -> list[Message]:
        """Convert raw_prompt to Message format.

        Args:
            raw_prompt: List of message dictionaries from dataset

        Returns:
            List of Message objects for RolloutRequest
        """
        messages = []
        for msg in raw_prompt:
            if isinstance(msg, dict):
                messages.append(
                    Message(
                        role=msg.get("role", "user"),
                        content=msg.get("content", ""),
                        tool_calls=msg.get("tool_calls"),
                        tool_call_id=msg.get("tool_call_id"),
                    )
                )
            elif isinstance(msg, Message):
                messages.append(msg)
        return messages

    def _get_original_prompt_ids(self, **kwargs) -> list[int]:
        """Get original prompt token IDs for error fallback.

        Uses the tokenizer to encode the raw_prompt, ensuring failed samples
        maintain consistent prompt representation for downstream processing.
        This is called when RolloutServer returns an error and we need to
        preserve the original prompt in the AgentLoopOutput.

        CRITICAL: Must use the SAME tools parameter as normal operation to ensure
        the chat template produces identical token sequences. Tool-enabled templates
        generate different prefixes than non-tool templates.

        Args:
            **kwargs: Must contain 'raw_prompt' with the original messages.

        Returns:
            List of token IDs representing the original prompt.
        """
        raw_prompt = kwargs.get("raw_prompt", [])
        if not raw_prompt:
            return []

        # Apply chat template to get prompt token IDs (matches local behavior)
        # Note: In remote mode, tools are fetched from MCP by RolloutServer,
        # so we don't have the tool list here. For error fallback, we tokenize
        # without tools - this may produce slightly different tokens but is
        # acceptable for error cases where the sample won't be used for training.
        try:
            return self.tokenizer.apply_chat_template(
                raw_prompt,
                tools=None,  # No tools in error fallback path
                add_generation_prompt=True,
                tokenize=True,
            )
        except Exception as e:
            logger.warning(f"Failed to tokenize prompt for error fallback: {e}")
            return []

    @classmethod
    def _preflight_check(cls, remote_config: DictConfig):
        """Verify connectivity to external services before training starts.

        This catches configuration errors (wrong URLs, auth issues, unreachable
        services) early, rather than failing on the first batch. Unlike local
        rollout which has no network dependencies, remote rollout benefits from
        early validation of the external service chain.

        Args:
            remote_config: Remote rollout configuration

        Raises:
            RuntimeError: If any required service is unreachable or unhealthy
        """
        errors = []

        # Check RolloutServer health
        try:
            api_key = os.environ.get(remote_config.api_key_env, None) if remote_config.get("api_key_env") else None
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

            response = httpx.get(
                f"{remote_config.rollout_server_url}/health",
                timeout=10.0,
                headers=headers,
            )
            if response.status_code != 200:
                errors.append(f"RolloutServer unhealthy: status {response.status_code}")
        except httpx.RequestError as e:
            errors.append(f"RolloutServer unreachable at {remote_config.rollout_server_url}: {e}")

        # Check LLMProxy health (if configured)
        llm_proxy_url = remote_config.get("llm_proxy_url")
        if llm_proxy_url:
            # Derive health URL from completion URL
            health_url = llm_proxy_url.replace("/v1/completions", "/health")
            if "/v1/completions" not in llm_proxy_url:
                # If the URL doesn't contain /v1/completions, just append /health
                health_url = llm_proxy_url.rstrip("/") + "/health"

            try:
                response = httpx.get(health_url, timeout=10.0)
                if response.status_code != 200:
                    errors.append(f"LLMProxy unhealthy: status {response.status_code}")
            except httpx.RequestError as e:
                errors.append(f"LLMProxy unreachable at {health_url}: {e}")

        if errors:
            error_msg = "Pre-flight check failed for remote rollout:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info("Pre-flight check passed: RolloutServer and LLMProxy are healthy")

    @classmethod
    async def cleanup_class(cls):
        """Cleanup shared resources (called on worker shutdown).

        This should be called when the AgentLoopWorker is shutting down
        to properly close the HTTP connection pool and release resources.
        """
        if cls.rollout_client:
            await cls.rollout_client.close()
            cls.rollout_client = None
        cls.semaphore = None
        cls.remote_config = None
        cls.tokenizer = None
        cls._class_initialized = False
        logger.info("RemoteAgentLoop cleanup completed")
