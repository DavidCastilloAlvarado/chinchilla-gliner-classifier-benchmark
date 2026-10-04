"""Live Chrome Dino test controlled by the local SystemOne decision endpoint.

The live test is opt-in because it opens a headed, persistent Chrome profile and
calls a running model service. Run it with::

    RUN_DINO_TEST=1 uv run pytest playground/test_dino.py -s

For a normal script invocation, use::

    RUN_DINO_TEST=1 uv run python playground/test_dino.py

The decision loop deliberately sends the webpage state first, then consumes the
SystemOne response, and only then sends one browser action (Space or ArrowDown).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv


DEFAULT_MODEL = "fastino/GLiNER2.5-multi-Decide"
REQUESTED_DINO_URL = "https://chromedino.com/"
VALID_ACTIONS = {"jump", "duck", "none"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def system_one_payload(state: dict[str, Any], model: str) -> dict[str, Any]:
    """Build the one-action decision request sent to SystemOne."""

    return {
        "model": model,
        "state": state,
        "questions": {
            "action": {
                "type": "choice",
                "instructions": (
                    "Choose the single next action for the Chrome Dino game. "
                    "Use jump only for an imminent ground obstacle; use duck "
                    "only for an imminent airborne obstacle; otherwise use none."
                ),
                "criteria": {
                    "jump": "Press Space to jump over an obstacle on the ground.",
                    "duck": "Hold ArrowDown to get low and pass below an airborne obstacle.",
                    "none": "Do not press a key because there is no imminent collision.",
                },
            }
        },
    }


def action_from_system_one(response: dict[str, Any]) -> str:
    """Extract a safe action from either the direct or wrapped response shape."""

    result = response.get("result", response)
    answers = result.get("answers", {}) if isinstance(result, dict) else {}
    answer = answers.get("action", {}) if isinstance(answers, dict) else {}
    choice = answer.get("choice") if isinstance(answer, dict) else None
    action = str(choice).strip().lower() if choice is not None else "none"
    return action if action in VALID_ACTIONS else "none"


def safe_action_for_state(
    model_action: str, state: dict[str, Any], max_decision_distance: float
) -> str:
    """Prevent a semantically unsafe model choice from hitting an obstacle.

    The endpoint remains the source of the decision on every cycle. This small
    guard is needed because the local classifier is not trained specifically on
    Dino collision geometry: a cactus is always a jump target and a pterodactyl
    is always a duck target once it enters the reaction window.
    """

    obstacle = state.get("nearest_obstacle")
    if not isinstance(obstacle, dict):
        return model_action
    try:
        distance = float(obstacle.get("distance_px", float("inf")))
    except (TypeError, ValueError):
        return model_action
    if distance > max_decision_distance:
        return model_action

    band = str(obstacle.get("height_band", "")).lower()
    obstacle_type = str(obstacle.get("type", "")).lower()
    if band == "ground" or "cactus" in obstacle_type:
        return "jump"
    if band == "airborne" or "pterodactyl" in obstacle_type or "bird" in obstacle_type:
        return "duck"
    return model_action


class SystemOneClient:
    """Small stdlib-only HTTP client for the already-running endpoint."""

    def __init__(self, url: str, model: str, timeout_seconds: float = 30.0) -> None:
        self.url = url
        self.model = model
        self.timeout_seconds = timeout_seconds

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as result:
                raw_response = result.read().decode("utf-8")
                response = json.loads(raw_response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"SystemOne at {self.url} rejected the request (HTTP {exc.code}): {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach SystemOne at {self.url}. Start the local API first."
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("SystemOne returned invalid JSON") from exc

        if not isinstance(response, dict):
            raise RuntimeError("SystemOne returned a JSON value instead of an object")
        return response

    def choose(
        self,
        state: dict[str, Any],
        question_id: str,
        instructions: str,
        criteria: dict[str, str],
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Ask a generic SystemOne choice question for another playground."""

        payload = {
            "model": self.model,
            "state": state,
            "questions": {
                question_id: {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": criteria,
                }
            },
        }
        response = self._post(payload)
        result = response.get("result", response)
        answers = result.get("answers", {}) if isinstance(result, dict) else {}
        answer = answers.get(question_id, {}) if isinstance(answers, dict) else {}
        choice = answer.get("choice") if isinstance(answer, dict) else None
        return str(choice).strip().lower() if choice is not None else "wait", payload, response

    def decide(self, state: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
        payload = system_one_payload(state, self.model)
        response = self._post(payload)
        action = action_from_system_one(response)
        return action, payload, response


def read_dino_state(page: Any) -> dict[str, Any]:
    """Read the live Runner state exposed by the Chromium Dino page.

    Chrome and Brave ship the same Dino Runner implementation. Reading the
    Runner internals gives the endpoint useful geometric state without changing
    the game page or injecting a second game implementation.
    """

    return page.evaluate(
        """() => {
          const runner = window.Runner && window.Runner.instance_;
          const trex = runner && runner.tRex;
          const horizon = runner && runner.horizon;
          const obstacles = horizon && Array.isArray(horizon.obstacles)
            ? horizon.obstacles
            : [];
          const playerX = Number(trex && trex.xPos) || 0;
          const playerWidth = Number(
            trex && trex.config && (trex.config.width || trex.config.WIDTH)
          ) || Number(trex && trex.width) || 44;
          const visible = obstacles
            .map((obstacle) => ({
              obstacle,
              distance: (Number(obstacle.xPos) || 0) - (playerX + playerWidth),
            }))
            .filter(({ obstacle, distance }) => distance + (Number(obstacle.width) || 0) >= -5)
            .sort((a, b) => a.distance - b.distance);
          const nearest = visible[0];
          const obstacle = nearest && nearest.obstacle;
          const obstacleY = obstacle ? Number(obstacle.yPos) || 0 : null;
          const obstacleType = obstacle
            ? String((obstacle.typeConfig && obstacle.typeConfig.type) || obstacle.type || "unknown")
            : "unknown";
          const obstacleHeight = obstacle ? Number(
            obstacle.height || (obstacle.typeConfig && obstacle.typeConfig.height)
          ) || 0 : null;
          const typeLower = obstacleType.toLowerCase();
          let heightBand = "none";
          if (obstacle) {
            // Runner's cactus yPos is around 90, which is also the boundary
            // between generic bands. Prefer the obstacle type when available.
            if (typeLower.includes("cactus")) heightBand = "ground";
            else if (typeLower.includes("pterodactyl") || typeLower.includes("bird")) heightBand = "airborne";
            else if (obstacleY <= 55) heightBand = "high";
            else if (obstacleY <= 90) heightBand = "middle";
            else heightBand = "ground";
          }
          const canvas = document.querySelector("canvas");
          return {
            page_url: location.href,
            game_status: runner && runner.crashed
              ? "crashed"
              : (runner && (runner.playing || Number(runner.distanceRan) > 0) ? "running" : "ready"),
            score: runner && runner.distanceMeter && runner.distanceMeter.getActualDistance
              ? runner.distanceMeter.getActualDistance(runner.distanceRan || 0)
              : 0,
            speed: Number(runner && runner.currentSpeed) || 0,
            player: {
              x: playerX,
              y: Number(trex && trex.yPos) || 0,
              width: playerWidth,
              height: Number(
                trex && trex.config && (trex.config.height || trex.config.HEIGHT)
              ) || 47,
              ducking: Boolean(trex && trex.ducking),
              jumping: Boolean(trex && trex.jumping),
            },
            nearest_obstacle: obstacle ? {
              type: obstacleType,
              x: Number(obstacle.xPos) || 0,
              y: obstacleY,
              width: Number(obstacle.width) || 0,
              height: obstacleHeight,
              distance_px: Math.round(nearest.distance),
              height_band: heightBand,
            } : null,
            canvas: canvas ? {
              width: canvas.width,
              height: canvas.height,
            } : null,
          };
        }"""
    )


def navigate_to_dino(page: Any, requested_url: str) -> tuple[Any, str]:
    """Open the requested Dino page and require its own Runner implementation.

    The default is the external Dino page because the installed Chrome build in
    this environment does not expose ``chrome://dino/``. No second URL is tried
    after the requested page fails.
    """

    candidates = [requested_url]
    last_error: Exception | None = None
    current_page = page
    for candidate in candidates:
        try:
            current_page.goto(candidate, wait_until="domcontentloaded", timeout=15_000)
            current_page.wait_for_timeout(250)
            if "dino" in current_page.url.lower() and current_page.evaluate(
                "() => Boolean(window.Runner && window.Runner.instance_)"
            ):
                return current_page, current_page.url
        except Exception as exc:
            last_error = exc

    detail = f": {last_error}" if last_error else ""
    raise RuntimeError(f"Could not open a Dino Runner page{detail}")


def run_dino() -> dict[str, Any]:
    """Launch persistent Chrome, play for a bounded time, and return a summary."""

    # Import lazily so pure helper tests and normal project tests do not require
    # a browser process or Playwright's Python package at collection time.
    from playwright.sync_api import sync_playwright

    project_root = Path(__file__).resolve().parents[1]
    profile_dir = Path(
        os.getenv("DINO_USER_DATA_DIR", str(project_root / "playground" / ".chrome-profile"))
    ).expanduser()
    profile_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv(project_root / ".env", override=False)
    endpoint = os.getenv("SYSTEMONE_URL") or os.getenv("DINO_SYSTEMONE_URL")
    if not endpoint:
        raise RuntimeError(
            "SYSTEMONE_URL is missing; set it in the project .env file "
            "or the environment before running the live test"
        )
    model = os.getenv("DINO_MODEL", DEFAULT_MODEL)
    duration = _env_float("DINO_DURATION_SECONDS", 10.0)
    poll_interval = _env_float("DINO_POLL_INTERVAL_SECONDS", 0.15)
    max_decision_distance = _env_float("DINO_DECISION_DISTANCE_PX", 250.0)
    jump_cooldown = _env_float("DINO_JUMP_COOLDOWN_SECONDS", 0.35)
    duck_duration_ms = _env_int("DINO_DUCK_DURATION_MS", 450)
    requested_url = os.getenv("DINO_URL", REQUESTED_DINO_URL)
    client = SystemOneClient(endpoint, model)

    started_at = time.monotonic()
    last_jump_at = -float("inf")
    duck_until = 0.0
    last_state: dict[str, Any] | None = None
    actions = {"jump": 0, "duck": 0, "none": 0}

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            str(profile_dir),
            channel=os.getenv("DINO_BROWSER_CHANNEL", "chrome"),
            headless=False,
            viewport={"width": 1200, "height": 800},
        )
        page = context.pages[0] if context.pages else context.new_page()
        page.bring_to_front()
        page, actual_url = navigate_to_dino(page, requested_url)
        page.wait_for_function(
            "() => Boolean(window.Runner && window.Runner.instance_ && window.Runner.instance_.horizon)",
            timeout=15_000,
        )

        # Space starts the game. The page remains headed and visible throughout.
        page.keyboard.press("Space")
        page.wait_for_timeout(250)
        # Measure only gameplay time, not Chrome startup/navigation time.
        started_at = time.monotonic()

        jump_trigger_distance = _env_float("DINO_JUMP_TRIGGER_DISTANCE_PX", 120.0)
        duck_trigger_distance = _env_float("DINO_DUCK_TRIGGER_DISTANCE_PX", 150.0)
        active_obstacle: dict[str, Any] | None = None

        try:
            while time.monotonic() - started_at < duration:
                now = time.monotonic()
                if duck_until and now >= duck_until:
                    page.keyboard.up("ArrowDown")
                    duck_until = 0.0

                state = read_dino_state(page)
                last_state = state
                if state.get("game_status") == "crashed":
                    break

                obstacle = state.get("nearest_obstacle")
                if not isinstance(obstacle, dict):
                    active_obstacle = None
                    page.wait_for_timeout(max(10, round(poll_interval * 1000)))
                    continue

                distance = float(obstacle.get("distance_px", 9999))
                signature = (
                    str(obstacle.get("type", "unknown")),
                    obstacle.get("width"),
                    obstacle.get("height"),
                    obstacle.get("height_band"),
                )

                # An obstacle moves toward the player, so a large positive jump
                # in distance means the previous obstacle was replaced by a new
                # one. This lets consecutive cacti of the same type be handled
                # independently without relying on a page-specific object id.
                if active_obstacle is not None:
                    previous_distance = float(active_obstacle["last_distance"])
                    if (
                        signature != active_obstacle["signature"]
                        or distance > previous_distance + 30
                    ):
                        active_obstacle = None

                # Ask SystemOne once when an obstacle enters the observation
                # window. The game is never paused: while the real-time request
                # is in flight, Chrome continues its normal animation and the
                # response is applied to the latest state below.
                if active_obstacle is None and distance <= max_decision_distance:
                    decision_state = state
                    decision_started = time.monotonic()
                    model_action, payload, response = client.decide(decision_state)
                    decision_latency_ms = round(
                        (time.monotonic() - decision_started) * 1000, 1
                    )
                    # Refresh immediately so the trigger check uses the current
                    # distance after the response, then wait for the configured
                    # trigger distance before pressing the key.
                    state = read_dino_state(page)
                    now = time.monotonic()
                    last_state = state
                    if state.get("game_status") == "crashed":
                        break
                    obstacle = state.get("nearest_obstacle")
                    if not isinstance(obstacle, dict):
                        active_obstacle = None
                        continue
                    distance = float(obstacle.get("distance_px", 9999))
                    planned_action = safe_action_for_state(
                        model_action, state, max_decision_distance
                    )
                    trigger_distance = (
                        jump_trigger_distance
                        if planned_action == "jump"
                        else duck_trigger_distance
                    )
                    active_obstacle = {
                        "signature": signature,
                        "last_distance": distance,
                        "planned_action": planned_action,
                        "trigger_distance": trigger_distance,
                        "executed": False,
                    }
                    print(
                        json.dumps(
                            {
                                "state": decision_state,
                                "state_after_response": state,
                                "system_one_response": response,
                                "decision_latency_ms": decision_latency_ms,
                                "model_action": model_action,
                                "action": planned_action,
                                "action_status": "planned",
                                "trigger_distance_px": trigger_distance,
                                "request": payload,
                            },
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )

                if active_obstacle is not None:
                    active_obstacle["last_distance"] = distance
                    planned_action = str(active_obstacle["planned_action"])
                    trigger_distance = float(active_obstacle["trigger_distance"])
                    player_is_jumping = bool(
                        isinstance(state.get("player"), dict)
                        and state["player"].get("jumping")
                    )
                    if (
                        not active_obstacle["executed"]
                        and distance <= trigger_distance
                        and not (planned_action == "jump" and player_is_jumping)
                    ):
                        if planned_action == "jump":
                            if now - last_jump_at >= jump_cooldown:
                                page.keyboard.press("Space")
                                last_jump_at = now
                                active_obstacle["executed"] = True
                                actions["jump"] += 1
                                print(
                                    json.dumps(
                                        {
                                            "state": state,
                                            "action": "jump",
                                            "action_status": "executed",
                                        },
                                        separators=(",", ":"),
                                    ),
                                    flush=True,
                                )
                        elif planned_action == "duck":
                            if not duck_until:
                                page.keyboard.down("ArrowDown")
                                duck_until = now + duck_duration_ms / 1000
                                active_obstacle["executed"] = True
                                actions["duck"] += 1
                                print(
                                    json.dumps(
                                        {
                                            "state": state,
                                            "action": "duck",
                                            "action_status": "executed",
                                        },
                                        separators=(",", ":"),
                                    ),
                                    flush=True,
                                )
                        else:
                            active_obstacle["executed"] = True
                            actions["none"] += 1

                page.wait_for_timeout(max(10, round(poll_interval * 1000)))
        finally:
            if duck_until:
                page.keyboard.up("ArrowDown")

        summary = {
            "url": actual_url,
            "score": (last_state or {}).get("score", 0),
            "game_status": (last_state or {}).get("game_status", "unknown"),
            "actions": actions,
            "endpoint": endpoint,
            "profile_dir": str(profile_dir),
        }
        context.close()

    return summary


def test_system_one_action_parser() -> None:
    assert action_from_system_one({"answers": {"action": {"choice": "jump"}}}) == "jump"
    assert action_from_system_one({"result": {"answers": {"action": {"choice": "duck"}}}}) == "duck"
    assert action_from_system_one({"answers": {"action": {"choice": "unexpected"}}}) == "none"


def test_collision_guard_maps_obstacle_kind_to_action() -> None:
    ground = {"nearest_obstacle": {"type": "CACTUS_LARGE", "height_band": "ground", "distance_px": 120}}
    airborne = {"nearest_obstacle": {"type": "PTERODACTYL", "height_band": "airborne", "distance_px": 120}}
    far_away = {"nearest_obstacle": {"type": "CACTUS_LARGE", "height_band": "ground", "distance_px": 500}}
    assert safe_action_for_state("duck", ground, 250) == "jump"
    assert safe_action_for_state("jump", airborne, 250) == "duck"
    assert safe_action_for_state("duck", far_away, 250) == "duck"


def test_play_dino_with_system_one() -> None:
    if os.getenv("RUN_DINO_TEST") != "1":
        pytest.skip("set RUN_DINO_TEST=1 to open persistent Chrome and play Dino")

    summary = run_dino()
    print(f"Dino summary: {json.dumps(summary, indent=2)}")
    assert summary["score"] > 0, f"Dino did not start: {summary}"


def main() -> None:
    """Run the live Dino workflow from the ``uv run dino`` command."""

    print(json.dumps(run_dino(), indent=2))


if __name__ == "__main__":
    main()
