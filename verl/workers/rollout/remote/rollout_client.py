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
"""HTTP client for communicating with external RolloutServer.

RolloutClient handles the HTTP communication between RemoteAgentLoop (in verl)
and the external RolloutServer. It provides:
- Connection pooling for efficiency
- Retry logic with exponential backoff
- Timeout handling
- Error classification

See design doc section 4.2 for specifications.
"""

import asyncio
import logging
import os
from typing import Optional

import httpx
from pydantic import ValidationError

from verl.workers.rollout.remote.schemas import RolloutOutput, RolloutRequest

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RolloutClientError(Exception):
    """Base exception for RolloutClient errors."""

    pass


class RolloutTransportError(RolloutClientError):
    """Network/transport level errors (timeouts, connection refused)."""

    pass


class RolloutServerError(RolloutClientError):
    """Server returned 5xx error."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Server error {status_code}: {detail}")


class RolloutValidationError(RolloutClientError):
    """Response validation failed."""

    pass


class RolloutClient:
    """HTTP client for RolloutServer with connection pooling.

    This client is designed to be shared across multiple RemoteAgentLoop instances
    within a single worker process. It maintains a persistent connection pool
    for efficient HTTP communication.

    Example:
        client = RolloutClient(
            base_url="http://rollout-server:8080",
            api_key="sk-xxx",
            timeout_seconds=300,
        )
        output = await client.rollout(request)
        await client.close()
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout_seconds: int = 300,
        max_retries: int = 3,
        max_connections: int = 100,
    ):
        """Initialize client with configuration.

        Args:
            base_url: Base URL of the RolloutServer (e.g., http://rollout-server:8080)
            api_key: Optional API key for authentication
            timeout_seconds: Request timeout in seconds (default: 300 for long rollouts)
            max_retries: Maximum retry attempts for transient errors (default: 3)
            max_connections: Maximum connections in the pool (default: 100)
        """
        self.base_url = base_url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        # Persistent connection pool with keepalive
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections // 2),
            headers=self._build_headers(),
        )

        logger.info(
            f"RolloutClient initialized: base_url={base_url}, "
            f"timeout={timeout_seconds}s, max_retries={max_retries}, "
            f"max_connections={max_connections}"
        )

    async def rollout(self, request: RolloutRequest) -> RolloutOutput:
        """Execute rollout and return trajectory.

        Sends a rollout request to the RolloutServer and waits for the complete
        trajectory response. Implements retry logic for transient errors.

        Args:
            request: RolloutRequest with LLM URL, messages, and configuration

        Returns:
            RolloutOutput with token sequences, masks, and metrics

        Raises:
            RolloutServerError: On server errors (5xx) after retries exhausted
            RolloutTransportError: On network errors after retries exhausted
            RolloutValidationError: On invalid response format
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                response = await self._client.post(
                    "/rollout",
                    json=request.model_dump(),
                )

                # Handle HTTP errors
                if response.status_code >= 400:
                    error_detail = response.text[:500] if response.text else "No error detail"

                    # Retry on 5xx errors with backoff
                    if response.status_code >= 500 and attempt < self.max_retries - 1:
                        wait_time = 2**attempt
                        logger.warning(
                            f"RolloutServer returned {response.status_code}, "
                            f"retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries}): "
                            f"{error_detail}"
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    # No retry for 4xx or exhausted retries
                    raise RolloutServerError(response.status_code, error_detail)

                # Parse and validate response
                try:
                    return RolloutOutput.model_validate(response.json())
                except ValidationError as e:
                    raise RolloutValidationError(f"Invalid response format: {e}") from e

            except httpx.TimeoutException as e:
                last_error = RolloutTransportError(f"Request timeout after {self.timeout_seconds}s: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Request timeout, retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise last_error from e

            except httpx.RequestError as e:
                last_error = RolloutTransportError(f"Network error: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Network error, retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise last_error from e

        # Should not reach here, but handle edge case
        if last_error:
            raise last_error
        raise RolloutClientError("Unknown error during rollout")

    async def health_check(self) -> bool:
        """Check if RolloutServer is healthy.

        Returns:
            True if server responds with 200 on /health, False otherwise
        """
        try:
            response = await self._client.get("/health", timeout=10.0)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def close(self):
        """Close the HTTP client and release resources.

        Should be called when the client is no longer needed to properly
        release connection pool resources.
        """
        await self._client.aclose()
        logger.info("RolloutClient closed")

    def _build_headers(self) -> dict:
        """Build request headers including auth if configured."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
