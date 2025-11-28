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
"""Data contracts for remote rollout communication.

These schemas define the request/response format between RemoteAgentLoop (verl)
and the external RolloutServer. They ensure consistent data exchange and enable
validation on both sides.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message for conversation history.

    Supports both text-only and structured content (for multi-modal).
    """

    role: str  # "system", "user", "assistant", "tool"
    content: str | list[dict[str, Any]]  # Text or structured content
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class SamplingParams(BaseModel):
    """LLM sampling parameters for generation.

    These parameters control the randomness and output length of LLM generation.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 512
    stop: Optional[list[str]] = None
    logprobs: Optional[int] = Field(
        default=1,
        description="None = no logprobs, N >= 1 = return top N logprobs. "
        "Note: vLLM requires logprobs >= 1 to return token_logprobs. "
        "Default is 1 (return logprob for sampled token only).",
    )


class RolloutRequest(BaseModel):
    """Request body for POST /rollout to RolloutServer.

    This is sent from RemoteAgentLoop to initiate a trajectory generation.
    The RolloutServer will execute the agent loop and call back to LLMProxy
    for each LLM turn.
    """

    # Session identification
    session_id: str = Field(
        ...,
        description="Unique ID for tracking: {job_id}-step{step}-idx{index}-n{rollout_n}",
    )

    # LLM configuration - LLMProxy URL (external-facing)
    llm_url: str = Field(..., description="URL to LLMProxy /v1/completions endpoint")
    llm_api_key: Optional[str] = Field(default=None, description="API key for LLMProxy auth")
    model: str = Field(..., description="Model identifier for LLM requests")

    # Conversation
    messages: list[Message] = Field(..., description="Initial messages (system + user prompt)")

    # Generation parameters
    sampling_params: SamplingParams

    # Tool configuration (optional)
    tool_server_url: Optional[str] = Field(
        default=None,
        description="URL to ToolServer (MCP endpoint). RolloutServer fetches tools from MCP.",
    )

    # Constraints
    max_turns: int = Field(default=10, description="Maximum conversation turns")
    max_tokens_total: int = Field(default=8192, description="Maximum total tokens in trajectory")

    # Metadata (for logging/tracing)
    metadata: dict[str, Any] = Field(default_factory=dict, description="tenant_id, job_id, step, etc.")


class ToolCall(BaseModel):
    """Record of a tool call made during rollout.

    Used for debugging and observability.
    """

    id: str
    name: str
    arguments: str  # JSON string
    result: Optional[str] = None
    latency_ms: Optional[float] = None


class RolloutMetrics(BaseModel):
    """Rollout performance metrics.

    Provides timing and resource usage information for monitoring.
    """

    total_latency_ms: float = Field(..., description="Total rollout wall-clock time")
    llm_latency_ms: float = Field(..., description="Time spent in LLM inference")
    tool_latency_ms: float = Field(default=0.0, description="Time spent in tool execution")
    num_llm_calls: int = Field(..., description="Number of LLM completion calls")
    num_tool_calls: int = Field(default=0, description="Number of tool executions")
    prompt_tokens: int = Field(..., description="Token count in initial prompt")
    response_tokens: int = Field(..., description="Token count in all responses")
    max_context_tokens: int = Field(
        default=0,
        description="Peak context window usage across all turns",
    )


class RolloutOutput(BaseModel):
    """Response body for POST /rollout from RolloutServer.

    Contains the complete trajectory data needed for PPO training.
    """

    # Token sequences (required for training)
    prompt_ids: list[int] = Field(..., description="Token IDs for initial prompt")
    response_ids: list[int] = Field(
        ...,
        description="Token IDs for all responses (LLM + tool outputs)",
    )
    response_mask: list[int] = Field(
        ...,
        description="Binary mask: 1 = LLM token (include in loss), 0 = tool/env token",
    )

    # Logprobs (same length as response_ids, 0.0 for tool tokens)
    response_logprobs: Optional[list[float]] = Field(
        default=None,
        description="Log probabilities for response tokens. 0.0 for tool tokens.",
    )

    # Trajectory metadata
    num_turns: int = Field(..., description="Number of conversation turns")
    finish_reason: str = Field(
        ...,
        description="Termination reason: 'stop', 'max_turns', 'max_tokens', 'error'",
    )

    # Error handling (for graceful degradation)
    success: bool = Field(default=True, description="False if rollout failed")
    error_message: Optional[str] = Field(default=None, description="Error description if success=False")

    # Debugging/observability
    messages: list[Message] = Field(default_factory=list, description="Full conversation history")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="Structured tool call records")
    metrics: RolloutMetrics

    # Passthrough for extra data
    extra_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional data: turn_scores, tool_rewards, etc.",
    )
