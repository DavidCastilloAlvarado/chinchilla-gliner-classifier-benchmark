"""Real-time ViZDoom playground controlled by the SystemOne endpoint.

Run with::

    uv run doom

The endpoint URL is loaded from ``.env`` using ``SYSTEMONE_URL``. The game is
advanced only after the SystemOne response arrives, so each decision is applied
to the exact ViZDoom state that was sent to the endpoint.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from playground.test_dino import SystemOneClient


ACTION_BUTTONS = {
    "move_forward": "MOVE_FORWARD",
    "move_backward": "MOVE_BACKWARD",
    # Turning is a rotation pulse only. Movement is a separate action.
    "turn_left": "TURN_LEFT",
    "turn_right": "TURN_RIGHT",
    # Internal navigation corrections: rotate and advance for one short tic.
    "advance_turn_left": ("TURN_LEFT", "MOVE_FORWARD"),
    "advance_turn_right": ("TURN_RIGHT", "MOVE_FORWARD"),
    "strafe_left": "MOVE_LEFT",
    "strafe_right": "MOVE_RIGHT",
    "attack": "ATTACK",
    "forward_attack": ("MOVE_FORWARD", "ATTACK"),
    "wait": (),
}

ACTION_CRITERIA = {
    "move_forward": "Walk forward through the Doom level.",
    "move_backward": "Back away from danger or reposition.",
    "turn_left": "Briefly rotate left to reorient, then release the button.",
    "turn_right": "Briefly rotate right to reorient, then release the button.",
    "strafe_left": "Strafe left while keeping the current facing direction.",
    "strafe_right": "Strafe right while keeping the current facing direction.",
    "attack": "Fire the current weapon at a visible or nearby enemy.",
    "forward_attack": "Move forward while firing at a visible or nearby enemy.",
    "wait": "Do not press a button for this frame.",
}


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


def _screen_ascii(screen_buffer: Any, width: int = 12, height: int = 6) -> str:
    """Compact the current visual observation into text for SystemOne."""

    if screen_buffer is None:
        return ""
    image = np.asarray(screen_buffer)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 3:
        image = image.mean(axis=2)
    if image.ndim != 2 or image.size == 0:
        return ""

    y_indices = np.linspace(0, image.shape[0] - 1, height).astype(int)
    x_indices = np.linspace(0, image.shape[1] - 1, width).astype(int)
    sample = image[np.ix_(y_indices, x_indices)].astype(float)
    low, high = float(sample.min()), float(sample.max())
    if high > low:
        sample = (sample - low) / (high - low)
    palette = " .:-=+*#%@"
    rows = []
    for row in sample:
        rows.append("".join(palette[min(len(palette) - 1, int(pixel * len(palette)))] for pixel in row))
    return "\n".join(rows)


MONSTER_NAME_HINTS = (
    "zombie",
    "imp",
    "demon",
    "cacodemon",
    "baron",
    "hell",
    "lostsoul",
    "revenant",
    "mancubus",
    "arachnotron",
    "spider",
    "spectre",
    "shotgunguy",
)

GOAL_NAME_HINTS = ("greenarmor", "vest")


def _iter_labels(labels: Any) -> list[Any]:
    return [] if labels is None else list(labels)


def _label_state(labels: Any) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for label in _iter_labels(labels)[:24]:
        category = str(getattr(label, "object_category", "unknown"))
        # Weapon pickups are intentionally outside the controller's scope.
        if category.lower() == "weapon":
            continue
        observations.append(
            {
                "object": str(getattr(label, "object_name", "unknown")),
                "category": category,
                "x": round(float(getattr(label, "object_position_x", 0.0)), 2),
                "y": round(float(getattr(label, "object_position_y", 0.0)), 2),
                "screen_x": round(float(getattr(label, "x", 0.0)), 2),
                "screen_y": round(float(getattr(label, "y", 0.0)), 2),
                "screen_width": round(float(getattr(label, "width", 0.0)), 2),
                "screen_height": round(float(getattr(label, "height", 0.0)), 2),
                "health": round(float(getattr(label, "object_health", 0.0)), 2),
            }
        )
    return observations


def _visible_monsters(labels: Any, screen_width: int = 320) -> list[dict[str, Any]]:
    """Return visible monsters with their horizontal aiming error."""

    monsters: list[dict[str, Any]] = []
    midpoint = screen_width / 2.0
    for label in _iter_labels(labels):
        name = str(getattr(label, "object_name", "unknown"))
        category = str(getattr(label, "object_category", ""))
        name_lower = name.lower()
        category_lower = category.lower()
        # Dead corpses and gore can contain names such as DeadZombieman, but
        # they are not valid targets and must never steal the aim target.
        if name_lower.startswith("dead") or category_lower in {"gore", "gibs", "blood"}:
            continue
        is_monster = category_lower == "monster" or any(
            hint in name_lower for hint in MONSTER_NAME_HINTS
        )
        if not is_monster:
            continue
        left = float(getattr(label, "x", 0.0))
        width = float(getattr(label, "width", 0.0))
        # Ignore one-pixel edge remnants; use a real visible target instead.
        if width < 4:
            continue
        center_x = left + width / 2.0
        monsters.append(
            {
                "object": name,
                "screen_center_x": round(center_x, 2),
                "horizontal_error_px": round(center_x - midpoint, 2),
                "screen_width": round(width, 2),
            }
        )
    return sorted(monsters, key=lambda monster: abs(monster["horizontal_error_px"]))


def _visible_goals(labels: Any, screen_width: int = 320) -> list[dict[str, Any]]:
    """Return visible end-of-level goals such as deadly corridor's vest."""

    goals: list[dict[str, Any]] = []
    midpoint = screen_width / 2.0
    for label in _iter_labels(labels):
        name = str(getattr(label, "object_name", "unknown"))
        category = str(getattr(label, "object_category", "")).lower()
        if not any(hint in name.lower() for hint in GOAL_NAME_HINTS):
            continue
        if category not in {"armor", "item", "goal", "unknown"}:
            continue
        width = float(getattr(label, "width", 0.0))
        if width < 2:
            continue
        center_x = float(getattr(label, "x", 0.0)) + width / 2.0
        goals.append(
            {
                "object": name,
                "screen_center_x": round(center_x, 2),
                "horizontal_error_px": round(center_x - midpoint, 2),
                "screen_width": round(width, 2),
            }
        )
    return sorted(goals, key=lambda goal: abs(goal["horizontal_error_px"]))


def _button_name(button: Any) -> str:
    name = getattr(button, "name", None)
    return name or str(button).split(".")[-1]


def doom_state(game: Any, state: Any, episode: int) -> dict[str, Any]:
    variables: dict[str, float] = {}
    available_variables = list(game.get_available_game_variables())
    values = list(state.game_variables) if state.game_variables is not None else []
    for variable, value in zip(available_variables, values):
        variables[str(variable).split(".")[-1]] = round(float(value), 3)

    screen = np.asarray(state.screen_buffer) if state.screen_buffer is not None else None
    screen_width = int(screen.shape[-1]) if screen is not None and screen.ndim >= 2 else 320
    visible_monsters = _visible_monsters(state.labels, screen_width)
    visible_goals = _visible_goals(state.labels, screen_width)
    # Keep monster targets close to the crosshair; Doom's hitscan shot is narrow.
    aim_tolerance = _env_float("DOOM_AIM_TOLERANCE", 0.02)
    monsters_in_front = [
        monster["object"]
        for monster in visible_monsters
        if abs(monster["horizontal_error_px"]) <= screen_width * aim_tolerance
    ]
    goals_in_front = [
        goal["object"]
        for goal in visible_goals
        if abs(goal["horizontal_error_px"]) <= screen_width * aim_tolerance
    ]
    return {
        "game": "Doom 1 via ViZDoom",
        "episode": episode,
        "episode_time": int(game.get_episode_time()),
        "total_reward": round(float(game.get_total_reward()), 3),
        "variables": variables,
        "available_buttons": [_button_name(button) for button in game.get_available_buttons()],
        "visible_labels": _label_state(state.labels),
        "visible_monsters": visible_monsters,
        "monsters_in_front": monsters_in_front,
        "visible_goals": visible_goals,
        "goals_in_front": goals_in_front,
        "aim_tolerance": aim_tolerance,
        "screen_ascii": _screen_ascii(state.screen_buffer),
    }


def action_vector(game: Any, action_name: str) -> list[int]:
    available = list(game.get_available_buttons())
    requested = ACTION_BUTTONS.get(action_name, ACTION_BUTTONS["wait"])
    if isinstance(requested, str):
        requested = (requested,)
    requested_names = set(requested)
    return [int(_button_name(button) in requested_names) for button in available]


def _player_position(state: dict[str, Any]) -> tuple[float, float, float] | None:
    variables = state.get("variables", {})
    keys = ("POSITION_X", "POSITION_Y", "POSITION_Z")
    if not all(key in variables for key in keys):
        return None
    return tuple(float(variables[key]) for key in keys)


def _active_tics(action_name: str, frame_skip: int) -> int:
    """Return how long a discrete button pulse is held before explicit release."""

    frame_skip = max(1, frame_skip)
    if action_name in {"turn_left", "turn_right", "advance_turn_left", "advance_turn_right"}:
        return min(frame_skip, max(1, _env_int("DOOM_TURN_TICS", 1)))
    if action_name == "attack":
        # The weapon needs several tics to complete its firing state; a one-tic
        # click can be released before a projectile is spawned.
        return min(frame_skip, max(1, _env_int("DOOM_ATTACK_TICS", 4)))
    return frame_skip


def run_doom() -> dict[str, Any]:
    """Run one or more bounded ViZDoom episodes using SystemOne choices."""

    try:
        import vizdoom as vzd
    except ImportError as exc:
        raise RuntimeError(
            "ViZDoom is not installed. Run `uv sync` before using `uv run doom`."
        ) from exc

    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env", override=False)
    endpoint = os.getenv("SYSTEMONE_URL") or os.getenv("DOOM_SYSTEMONE_URL") or os.getenv("DINO_SYSTEMONE_URL")
    if not endpoint:
        raise RuntimeError(
            "SYSTEMONE_URL is missing; set it in the project .env file before running `uv run doom`"
        )

    # deadly_corridor provides multiple enemies together with navigable space;
    # basic.cfg contains only a single target and ends quickly after it is killed.
    scenario_name = os.getenv("DOOM_SCENARIO", "deadly_corridor.cfg")
    scenario_path = Path(scenario_name)
    if not scenario_path.is_file():
        scenario_path = Path(vzd.scenarios_path) / scenario_name
    if not scenario_path.is_file():
        raise FileNotFoundError(f"ViZDoom scenario was not found: {scenario_name}")

    visible = os.getenv("DOOM_HEADLESS", "0") != "1"
    frame_skip = _env_int("DOOM_FRAME_SKIP", 4)
    max_seconds = _env_float("DOOM_MAX_SECONDS", 30.0)
    # Zero means keep restarting episodes until DOOM_MAX_SECONDS expires.
    max_episodes = _env_int("DOOM_EPISODES", 0)
    model = os.getenv("DOOM_MODEL", os.getenv("DINO_MODEL", "fastino/GLiNER2.5-multi-Decide"))
    client = SystemOneClient(endpoint, model, timeout_seconds=_env_float("DOOM_SYSTEMONE_TIMEOUT_SECONDS", 5.0))

    game = vzd.DoomGame()
    game.load_config(str(scenario_path))
    # basic.cfg exposes only left/right/attack by default. Add the locomotion
    # buttons explicitly; otherwise `move_forward` becomes an all-zero vector.
    for button_name in (
        "MOVE_FORWARD",
        "MOVE_BACKWARD",
        "TURN_LEFT",
        "TURN_RIGHT",
        "MOVE_LEFT",
        "MOVE_RIGHT",
        "ATTACK",
    ):
        game.add_available_button(getattr(vzd.Button, button_name))
    # basic.cfg only requests AMMO2. Add movement and combat telemetry so the
    # endpoint can distinguish exploration from a stationary turn or death.
    for variable_name in (
        "HEALTH",
        "ARMOR",
        "KILLCOUNT",
        "AMMO1",
        "AMMO2",
        "POSITION_X",
        "POSITION_Y",
        "POSITION_Z",
        "ANGLE",
        "PITCH",
    ):
        game.add_available_game_variable(getattr(vzd.GameVariable, variable_name))
    game.set_labels_buffer_enabled(True)
    game.set_window_visible(visible)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()

    actions_taken = 0
    episodes_finished = 0
    started = time.monotonic()
    last_state: dict[str, Any] = {}
    last_action = "none"
    turn_streak = 0
    previous_position: tuple[float, float, float] | None = None
    stuck_steps = 0
    stuck_turn_action: str | None = None
    try:
        game.new_episode()
        while time.monotonic() - started < max_seconds:
            if game.is_episode_finished():
                episodes_finished += 1
                if max_episodes > 0 and episodes_finished >= max_episodes:
                    break
                # A death or completed scenario should not terminate `uv run doom`;
                # start the next Doom episode until the configured time limit.
                game.new_episode()
                previous_position = None
                stuck_steps = 0
                stuck_turn_action = None
                turn_streak = 0

            observation = game.get_state()
            if observation is None:
                continue
            current_state = doom_state(game, observation, episodes_finished + 1)
            position = _player_position(current_state)
            movement_actions = {
                "move_forward",
                "move_backward",
                "strafe_left",
                "strafe_right",
                "forward_attack",
                "advance_turn_left",
                "advance_turn_right",
            }
            if position is not None and previous_position is not None and last_action in movement_actions:
                distance = sum((a - b) ** 2 for a, b in zip(position, previous_position)) ** 0.5
                if distance < 0.5:
                    stuck_steps += 1
                    # Keep turning in one direction. Alternating left/right
                    # every tic cancels the rotation and leaves the player at
                    # the wall forever.
                    if stuck_turn_action is None:
                        stuck_turn_action = "advance_turn_left"
                else:
                    stuck_steps = 0
                    stuck_turn_action = None
            elif last_action not in movement_actions:
                stuck_steps = 0
                stuck_turn_action = None
            previous_position = position
            current_state["last_action"] = last_action
            current_state["turn_streak"] = turn_streak
            current_state["stuck_steps"] = stuck_steps
            current_state["stuck_turn_action"] = stuck_turn_action
            decision_started = time.monotonic()
            model_action, payload, response = client.choose(
                current_state,
                question_id="doom_action",
                instructions=(
                    "Choose one action for the next Doom game frame using the visual "
                    "observation, visible labels, health, ammunition, and reward. "
                    "A monster must be centered under the crosshair before attacking; "
                    "if it is left or right of center, turn toward it first. After "
                    "all visible monsters are cleared, follow the end-of-level goal "
                    "when it is visible. Ignore weapon pickups completely. Prefer "
                    "forward movement when no enemy or goal is visible. Return exactly "
                    "one criterion."
                ),
                criteria=ACTION_CRITERIA,
            )
            decision_latency_ms = round((time.monotonic() - decision_started) * 1000, 1)
            if model_action not in ACTION_BUTTONS:
                model_action = "wait"
            action_name = model_action
            override_reason = None
            # Aim before firing: turn toward a visible monster until its label
            # overlaps the crosshair, then fire. This does not trust a generic
            # "monster visible" signal because Doom's hitscan weapon needs aim.
            visible_monsters = current_state["visible_monsters"]
            if visible_monsters:
                target = visible_monsters[0]
                if target["object"] in current_state["monsters_in_front"]:
                    # Clear every currently visible threat before resuming
                    # navigation. Do not walk into a centered enemy.
                    action_name = "attack"
                    override_reason = "monster_centered"
                elif target["horizontal_error_px"] < 0:
                    action_name = "turn_left"
                    override_reason = "aim_left"
                else:
                    action_name = "turn_right"
                    override_reason = "aim_right"
            # If position telemetry says a movement action made no progress,
            # turn out of the obstacle instead of repeatedly pressing forward.
            elif stuck_steps:
                action_name = stuck_turn_action or "advance_turn_left"
                override_reason = "movement_stuck"
            elif current_state["visible_goals"]:
                goal = current_state["visible_goals"][0]
                if goal["object"] in current_state["goals_in_front"]:
                    action_name = "move_forward"
                    override_reason = "goal_centered"
                elif goal["horizontal_error_px"] < 0:
                    action_name = "advance_turn_left"
                    override_reason = "goal_left"
                else:
                    action_name = "advance_turn_right"
                    override_reason = "goal_right"
            elif action_name == "forward_attack":
                # There is no target in view, so do not waste an attack while
                # navigating; reserve shooting for the visible-target branch.
                action_name = "move_forward"
                override_reason = "no_visible_target"
            # A turn is a one-tic pulse. Require a movement/non-turn decision
            # before allowing another turn pulse, preventing repeated model
            # choices from becoming a continuous camera spin.
            elif turn_streak and action_name in {"turn_left", "turn_right"}:
                action_name = "move_forward"
                override_reason = "turn_cooldown"
            turn_streak = turn_streak + 1 if action_name in {"turn_left", "turn_right"} else 0
            last_action = action_name
            vector = action_vector(game, action_name)
            executed_buttons = [
                _button_name(button)
                for button, pressed in zip(game.get_available_buttons(), vector)
                if pressed
            ]
            active_tics = _active_tics(action_name, frame_skip)
            reward = game.make_action(vector, active_tics)
            release_tics = max(0, frame_skip - active_tics)
            if release_tics:
                # Explicitly release every button for the rest of the decision
                # interval. The next decision starts from a neutral input state.
                reward += game.make_action([0] * len(vector), release_tics)
            actions_taken += 1
            last_state = current_state
            print(
                json.dumps(
                    {
                        "state": current_state,
                        "system_one_response": response,
                        "decision_latency_ms": decision_latency_ms,
                        "model_action": model_action,
                        "action": action_name,
                        "override_reason": override_reason,
                        "executed_buttons": executed_buttons,
                        "button_tics": active_tics,
                        "release_tics": release_tics,
                        "reward": round(float(reward), 3),
                        "request": payload,
                    },
                    separators=(",", ":"),
                    default=str,
                ),
                flush=True,
            )
    finally:
        final_summary = {
            "scenario": str(scenario_path),
            "endpoint": endpoint,
            "actions": actions_taken,
            "episodes_finished": episodes_finished,
            "last_state": last_state,
        }
        game.close()

    print(json.dumps(final_summary, indent=2, default=str), flush=True)
    return final_summary


def main() -> None:
    run_doom()


if __name__ == "__main__":
    main()
