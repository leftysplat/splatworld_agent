"""
World Labs Marble API Client for SplatWorld Agent.

Converts images to 3D Gaussian splats via the Marble API.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import httpx


API_BASE = "https://api.worldlabs.ai/marble/v1"
MARBLE_COST_PER_GENERATION = 1.50


class MarbleError(Exception):
    """Base exception for Marble API errors."""
    pass


class MarbleAuthError(MarbleError):
    """Authentication error."""
    pass


class MarbleTimeoutError(MarbleError):
    """Operation timed out."""
    pass


class MarbleConversionError(MarbleError):
    """Conversion failed."""
    pass


@dataclass
class Operation:
    """Represents a Marble operation."""
    operation_id: str
    done: bool = False
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    response: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "Operation":
        return cls(
            operation_id=data.get("operation_id") or data.get("name", "").split("/")[-1],
            done=data.get("done", False),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
            response=data.get("response", {}),
        )


@dataclass
class MarbleResult:
    """Result from a Marble conversion."""
    world_id: str
    splat_url: Optional[str] = None
    mesh_url: Optional[str] = None
    pano_url: Optional[str] = None
    cost_usd: float = MARBLE_COST_PER_GENERATION

    @property
    def viewer_url(self) -> str:
        """Get the Marble web viewer URL."""
        return f"https://worldlabs.ai/viewer/{self.world_id}"


class MarbleClient:
    """Client for the World Labs Marble API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Marble client.

        Args:
            api_key: Marble API key. If not provided, reads from WORLDLABS_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("WORLDLABS_API_KEY")
        if not self.api_key:
            raise MarbleAuthError(
                "Marble API key not provided. Set WORLDLABS_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self._client = httpx.Client(timeout=60.0)

    def _request(
        self,
        endpoint: str,
        method: str = "GET",
        json_data: dict = None,
        retries: int = 3,
    ) -> dict:
        """Make an authenticated request to the Marble API with retry logic."""
        headers = {
            "Content-Type": "application/json",
            "WLT-Api-Key": self.api_key,
        }

        last_error = None
        for attempt in range(1, retries + 1):
            try:
                if method == "GET":
                    response = self._client.get(f"{API_BASE}{endpoint}", headers=headers)
                elif method == "POST":
                    response = self._client.post(f"{API_BASE}{endpoint}", headers=headers, json=json_data)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                if response.status_code == 401:
                    raise MarbleAuthError("Invalid Marble API key")

                if response.status_code == 429:
                    if attempt < retries:
                        time.sleep(10 * attempt)
                        continue
                    raise MarbleError("Marble API rate limit exceeded")

                if response.status_code >= 500:
                    if attempt < retries:
                        time.sleep(5 * attempt)
                        continue
                    raise MarbleError(f"Marble API server error: {response.status_code}")

                if not response.is_success:
                    raise MarbleError(f"Marble API error {response.status_code}: {response.text}")

                return response.json()

            except httpx.RequestError as e:
                last_error = e
                if attempt < retries:
                    time.sleep(5 * attempt)
                    continue
                raise MarbleError(f"Request failed: {e}")

        raise last_error or MarbleError("Request failed after retries")

    def generate_world(
        self,
        image_base64: str,
        mime_type: str = "image/png",
        display_name: str = "Generated World",
        is_panorama: bool = False,
    ) -> Operation:
        """
        Start world generation from an image.

        Args:
            image_base64: Base64 encoded image data
            mime_type: MIME type of the image
            display_name: Name for the generated world
            is_panorama: Whether the image is an equirectangular panorama

        Returns:
            Operation object with operation_id for polling
        """
        request_body = {
            "display_name": display_name,
            "world_prompt": {
                "type": "image",
                "image_prompt": {
                    "source": "data_base64",
                    "data_base64": image_base64,
                    "mime_type": mime_type,
                    "is_pano": is_panorama,
                },
            },
        }

        response = self._request("/worlds:generate", method="POST", json_data=request_body)
        return Operation.from_dict(response)

    def generate_world_from_url(
        self,
        image_url: str,
        display_name: str = "Generated World",
        is_panorama: bool = False,
    ) -> Operation:
        """
        Start world generation from an image URL.

        Args:
            image_url: URL of the image
            display_name: Name for the generated world
            is_panorama: Whether the image is an equirectangular panorama

        Returns:
            Operation object with operation_id for polling
        """
        request_body = {
            "display_name": display_name,
            "world_prompt": {
                "type": "image",
                "image_prompt": {
                    "source": "uri",
                    "uri": image_url,
                    "is_pano": is_panorama,
                },
            },
        }

        response = self._request("/worlds:generate", method="POST", json_data=request_body)
        return Operation.from_dict(response)

    def poll_operation(self, operation_id: str) -> Operation:
        """Poll an operation's status."""
        response = self._request(f"/operations/{operation_id}")
        return Operation.from_dict(response)

    def wait_for_completion(
        self,
        operation_id: str,
        timeout: float = 600.0,
        poll_interval: float = 10.0,
        on_progress: Optional[Callable[[str, str], None]] = None,
    ) -> Operation:
        """
        Wait for an operation to complete.

        Args:
            operation_id: The operation ID to wait for
            timeout: Maximum wait time in seconds (default 10 minutes)
            poll_interval: Time between polls in seconds
            on_progress: Optional callback called with (status, description)

        Returns:
            Completed Operation object
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            operation = self.poll_operation(operation_id)

            if operation.done:
                if operation.error:
                    raise MarbleConversionError(f"Generation failed: {operation.error}")
                return operation

            if on_progress and operation.metadata:
                progress = operation.metadata.get("progress", {})
                status = progress.get("status", "UNKNOWN")
                description = progress.get("description", "")
                on_progress(status, description)

            time.sleep(poll_interval)

        raise MarbleTimeoutError(f"Operation {operation_id} timed out after {timeout} seconds")

    def generate_and_wait(
        self,
        image_base64: str,
        mime_type: str = "image/png",
        display_name: str = "Generated World",
        is_panorama: bool = False,
        timeout: float = 600.0,
        on_progress: Optional[Callable[[str, str], None]] = None,
    ) -> MarbleResult:
        """
        Generate a world and wait for completion.

        This is the main method for converting an image to a 3D splat.

        Args:
            image_base64: Base64 encoded image data
            mime_type: MIME type of the image
            display_name: Name for the world
            is_panorama: Whether the image is a panorama
            timeout: Max wait time in seconds
            on_progress: Optional progress callback

        Returns:
            MarbleResult with all URLs
        """
        operation = self.generate_world(
            image_base64=image_base64,
            mime_type=mime_type,
            display_name=display_name,
            is_panorama=is_panorama,
        )

        completed = self.wait_for_completion(
            operation.operation_id,
            timeout=timeout,
            on_progress=on_progress,
        )

        world_id = completed.metadata.get("world_id", "")
        response = completed.response

        splat_url = None
        if response.get("splats", {}).get("spz_urls"):
            splat_url = response["splats"]["spz_urls"].get("full_res")

        mesh_url = response.get("mesh", {}).get("collider_mesh_url")
        pano_url = response.get("imagery", {}).get("pano_url")

        return MarbleResult(
            world_id=world_id,
            splat_url=splat_url,
            mesh_url=mesh_url,
            pano_url=pano_url,
        )

    def download_file(self, url: str, output_path: Path) -> Path:
        """
        Download a file from a URL.

        Args:
            url: URL of the file
            output_path: Local path to save

        Returns:
            Output path
        """
        response = self._client.get(url, follow_redirects=True)
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)

        return output_path

    def estimate_cost(self, count: int) -> float:
        """Estimate cost for N conversions."""
        return count * MARBLE_COST_PER_GENERATION

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
