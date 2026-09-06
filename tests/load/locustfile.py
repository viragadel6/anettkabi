"""Locust load profile for the Video-to-Video SFX API.

Run against a staging deployment:

    VSFX_API_KEY=vsfx_... VSFX_BASE_URL=https://staging... \
        locust -f tests/load/locustfile.py --headless -u 20 -r 2 -t 5m

Weighted toward cheap endpoints (health/list/get) with a small share of full
prediction creations, mirroring real portal traffic without flooding the GPU
queue.
"""

from __future__ import annotations

import os
import random
import time

from locust import HttpUser, between, task

API_PREFIX = "/api/v1"
PROMPTS = (
    "cinematic whoosh with deep sub-bass impact",
    "gentle rain on a tin roof, distant thunder",
    "retro synth stab, tape saturation",
    "gritty footsteps on gravel, wind bed",
    "metallic drone slowly rising",
)
VIDEO_URL = os.environ.get("VSFX_LOAD_VIDEO_URL", "")


class SfxPortalUser(HttpUser):
    """Simulates a studio user browsing and occasionally generating."""

    wait_time = between(1.0, 4.0)
    weight = 8

    def on_start(self) -> None:
        """Attach the bearer token from the environment.

        Raises:
            RuntimeError: Without VSFX_API_KEY set.
        """
        api_key = os.environ.get("VSFX_API_KEY", "")
        if not api_key:
            raise RuntimeError("set VSFX_API_KEY for load tests")
        self.client.headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }

    @task(6)
    def list_predictions(self) -> None:
        """Page the prediction list."""
        self.client.get(f"{API_PREFIX}/predictions?limit=10", name="/predictions")

    @task(3)
    def model_card(self) -> None:
        """Fetch the model card."""
        self.client.get(f"{API_PREFIX}/models/video-to-video-sfx", name="/models")

    @task(2)
    def usage(self) -> None:
        """Fetch account usage."""
        self.client.get(f"{API_PREFIX}/account/usage", name="/account/usage")

    @task(1)
    def create_prediction(self) -> None:
        """Submit a generation (only when a source URL is configured)."""
        if not VIDEO_URL:
            self.client.get(f"{API_PREFIX}/healthz", name="/healthz")
            return
        self.client.post(
            f"{API_PREFIX}/predictions/video-to-video-sfx",
            json={
                "video": VIDEO_URL,
                "prompt": random.choice(PROMPTS),
                "num_inference_steps": 8,
                "guidance_scale": 2.0,
            },
            name="/predictions:create",
        )


class ProbeUser(HttpUser):
    """Unauthenticated monitor hitting the probes."""

    wait_time = between(0.5, 2.0)
    weight = 2

    @task
    def probes(self) -> None:
        """Hit health and version."""
        with self.client.get(f"{API_PREFIX}/healthz", name="/healthz", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"healthz {response.status_code}")
        self.client.get(f"{API_PREFIX}/version", name="/version")


class ResultPoller(HttpUser):
    """Simulates clients polling an existing prediction to completion."""

    wait_time = between(2.0, 3.0)
    weight = 1

    def on_start(self) -> None:
        """Reuse the portal auth and seed a poll target."""
        api_key = os.environ.get("VSFX_API_KEY", "")
        if api_key:
            self.client.headers = {"Authorization": f"Bearer {api_key}"}
        self.target_id: str | None = None
        self.deadline = time.monotonic() + 600.0

    @task
    def poll(self) -> None:
        """Poll the seeded prediction until terminal."""
        if self.target_id is None:
            listing = self.client.get(f"{API_PREFIX}/predictions?limit=1", name="/predictions").json()
            rows = listing.get("data", {}).get("results", []) if isinstance(listing, dict) else []
            self.target_id = rows[0]["id"] if rows else None
            if self.target_id is None:
                return
        if time.monotonic() > self.deadline:
            self.target_id = None
            self.deadline = time.monotonic() + 600.0
            return
        with self.client.get(
            f"{API_PREFIX}/predictions/{self.target_id}", name="/predictions:get", catch_response=True
        ) as response:
            if response.status_code == 404:
                self.target_id = None
                response.success()
