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
    # Each choice maps to one physical button; translation and rotation are
    # separate decisions, never combined with each other or with attack.
    "move_forward": "MOVE_FORWARD",
    "move_backward": "MOVE_BACKWARD",
    "move_left": "MOVE_LEFT",
    "move_right": "MOVE_RIGHT",
    "turn_left": "TURN_LEFT",
    "turn_right": "TURN_RIGHT",
    "attack": "ATTACK",
    "wait": (),
}

TRANSLATION_ACTIONS = frozenset(
    {"move_forward", "move_backward", "move_left", "move_right"}
)

ACTION_CRITERIA = {
    "move_forward": "Press only MOVE_FORWARD to walk forward when no living enemy is visible and navigation is appropriate.",
    "move_backward": "Press only MOVE_BACKWARD to retreat when no living enemy is visible.",
    "move_left": "Press only MOVE_LEFT to strafe left when no living enemy is visible.",
    "move_right": "Press only MOVE_RIGHT to strafe right when no living enemy is visible.",
    "turn_left": "Press only TURN_LEFT for a short pulse to center the view on a visible enemy or reorient.",
    "turn_right": "Press only TURN_RIGHT for a short pulse to center the view on a visible enemy or reorient.",
    "attack": "Press only ATTACK when combat_status is shoot_ready; otherwise turn to aim before firing.",
    "wait": "Press no buttons for this decision.",
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
        # Even a partially clipped enemy counts as visible for the combat
        # movement lock; do not walk while any living monster label is present.
        if width <= 0:
            continue
        center_x = left + width / 2.0
        monsters.append(
            {
                "object": name,
                "world_x": round(float(getattr(label, "object_position_x", 0.0)), 2),
                "world_y": round(float(getattr(label, "object_position_y", 0.0)), 2),
                "screen_center_x": round(center_x, 2),
                "horizontal_error_px": round(center_x - midpoint, 2),
                "screen_width": round(width, 2),
            }
        )
    # Keep the state list deterministic without ranking targets in controller code.
    return sorted(monsters, key=lambda monster: (monster["object"], monster["world_x"], monster["world_y"]))


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
        world_x = float(getattr(label, "object_position_x", 0.0))
        world_y = float(getattr(label, "object_position_y", 0.0))
        goals.append(
            {
                "object": name,
                "world_x": round(world_x, 2),
                "world_y": round(world_y, 2),
                "screen_center_x": round(center_x, 2),
                "horizontal_error_px": round(center_x - midpoint, 2),
                "screen_width": round(width, 2),
            }
        )
    return sorted(goals, key=lambda goal: abs(goal["horizontal_error_px"]))


def _button_name(button: Any) -> str:
    name = getattr(button, "name", None)
    return name or str(button).split(".")[-1]


def _game_variables(game: Any, state: Any) -> dict[str, float]:
    available = list(game.get_available_game_variables())
    values = list(state.game_variables) if state.game_variables is not None else []
    return {
        str(variable).split(".")[-1]: float(value)
        for variable, value in zip(available, values)
    }


def _position_tag(center_x: float, screen_width: int) -> str:
    """Normalize horizontal position into left / on_spot / right."""

    normalized_x = center_x / max(screen_width, 1)
    if normalized_x < 0.4:
        return "left"
    if normalized_x > 0.6:
        return "right"
    return "on_spot"


def _distance_tag(distance: float) -> str:
    if distance <= 128:
        return "near"
    if distance <= 320:
        return "medium"
    return "far"


def _combat_status(enemies: list[dict[str, Any]], must_aim_after_shot: bool) -> str:
    if not enemies:
        return "no_enemies"
    if must_aim_after_shot and any(enemy["position"] != "on_spot" for enemy in enemies):
        return "turn_after_shot"
    if any(enemy["shoot_status"] == "ready_to_shoot" for enemy in enemies):
        return "shoot_ready"
    return "aim_required"


def _level_tag(value: float, *, critical: float, low: float) -> str:
    if value <= critical:
        return "critical"
    if value <= low:
        return "low"
    return "normal"


def _player_position(game: Any, state: Any) -> tuple[float, float, float] | None:
    variables = _game_variables(game, state)
    keys = ("POSITION_X", "POSITION_Y", "POSITION_Z")
    if not all(key in variables for key in keys):
        return None
    return tuple(variables[key] for key in keys)


def doom_state(game: Any, state: Any, episode: int | None = None) -> dict[str, Any]:
    """Build a compact discrete state for SystemOne."""

    variables = _game_variables(game, state)
    screen = np.asarray(state.screen_buffer) if state.screen_buffer is not None else None
    screen_width = int(screen.shape[-1]) if screen is not None and screen.ndim >= 2 else 320
    player_x = variables.get("POSITION_X", 0.0)
    player_y = variables.get("POSITION_Y", 0.0)

    enemies = []
    for index, monster in enumerate(_visible_monsters(state.labels, screen_width), start=1):
        position = _position_tag(monster["screen_center_x"], screen_width)
        distance = (
            (monster["world_x"] - player_x) ** 2
            + (monster["world_y"] - player_y) ** 2
        ) ** 0.5
        enemies.append(
            {
                "id": f"enemy_{index}",
                "type": monster["object"],
                "position": position,
                "shoot_status": "ready_to_shoot" if position == "on_spot" else "aim_required",
                "distance": _distance_tag(distance),
            }
        )

    goals = []
    for goal in _visible_goals(state.labels, screen_width):
        distance = ((goal["world_x"] - player_x) ** 2 + (goal["world_y"] - player_y) ** 2) ** 0.5
        goals.append(
            {
                "type": goal["object"],
                "position": _position_tag(goal["screen_center_x"], screen_width),
                "distance": _distance_tag(distance),
            }
        )

    ammo = sum(value for key, value in variables.items() if key.startswith("AMMO"))
    ammo_status = "empty" if ammo <= 0 else "low" if ammo <= 10 else "available"
    armor = variables.get("ARMOR", 0.0)
    armor_status = "none" if armor <= 0 else "low" if armor <= 25 else "protected"
    return {
        "player": {
            "health": _level_tag(variables.get("HEALTH", 100.0), critical=25, low=50),
            "armor": armor_status,
            "ammo": ammo_status,
        },
        "enemies": enemies,
        "goal": goals,
    }


def action_vector(game: Any, action_name: str) -> list[int]:
    available = list(game.get_available_buttons())
    requested = ACTION_BUTTONS.get(action_name, ACTION_BUTTONS["wait"])
    if isinstance(requested, str):
        requested = (requested,)
    requested_names = set(requested)
    return [int(_button_name(button) in requested_names) for button in available]


def apply_movement_safety(
    action_name: str,
    *,
    visible_enemy_count: int,
    defensive_scenario: bool,
) -> tuple[str, str | None]:
    """Block translation in combat and on stationary defense maps."""

    if action_name in TRANSLATION_ACTIONS and visible_enemy_count:
        return "wait", "no_movement_while_enemies_visible"
    if action_name in TRANSLATION_ACTIONS and defensive_scenario:
        return "wait", "stationary_defense_map"
    return action_name, None


def apply_combat_safety(
    action_name: str,
    combat_status: str,
    enemies: list[dict[str, Any]],
    last_action: str,
) -> tuple[str, str | None]:
    """Apply the discrete aim/shoot contract without combining buttons."""

    if combat_status == "shoot_ready":
        if action_name != "attack":
            return "attack", "shoot_ready_priority"
        return action_name, None

    sides = [enemy["position"] for enemy in enemies if enemy["position"] in {"left", "right"}]
    if not sides:
        return ("wait", f"attack_blocked_{combat_status}") if action_name == "attack" else (action_name, None)

    side_set = set(sides)
    if side_set == {"left"}:
        required_turn = "turn_left"
    elif side_set == {"right"}:
        required_turn = "turn_right"
    elif action_name in {"turn_left", "turn_right"}:
        required_turn = action_name
    else:
        left_count = sides.count("left")
        right_count = sides.count("right")
        if left_count != right_count:
            required_turn = "turn_left" if left_count > right_count else "turn_right"
        else:
            required_turn = "turn_right" if last_action == "turn_left" else "turn_left"

    if action_name != required_turn:
        return required_turn, f"aim_toward_{required_turn.removeprefix('turn_')}"
    return action_name, None


def _attack_duration(frame_skip: int) -> int:
    return max(frame_skip, max(1, _env_int("DOOM_ATTACK_TICS", 4)))


def _active_tics(action_name: str, frame_skip: int) -> int:
    """Return the bounded single-button pulse duration."""

    frame_skip = max(1, frame_skip)
    if action_name in {"turn_left", "turn_right"}:
        return min(frame_skip, max(1, _env_int("DOOM_TURN_TICS", 1)))
    if action_name == "attack":
        return _attack_duration(frame_skip)
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
    scenario_id = Path(scenario_name).name.lower()
    defensive_scenario = scenario_id in {"defend_the_line.cfg", "defend_the_center.cfg"}
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
    previous_position: tuple[float, float, float] | None = None
    stuck_steps = 0
    must_aim_after_shot = False
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
                last_action = "none"
                last_state = {}
                must_aim_after_shot = False

            observation = game.get_state()
            if observation is None:
                continue
            current_state = doom_state(game, observation)
            position = _player_position(game, observation)
            if position is not None and previous_position is not None and last_action in TRANSLATION_ACTIONS:
                distance = sum((a - b) ** 2 for a, b in zip(position, previous_position)) ** 0.5
                stuck_steps = stuck_steps + 1 if distance < 0.5 else 0
            elif last_action not in TRANSLATION_ACTIONS:
                stuck_steps = 0
            previous_position = position
            enemy_visible = bool(current_state["enemies"])
            goal = current_state["goal"][0] if current_state["goal"] else None
            if enemy_visible:
                movement_status = "blocked_by_enemy"
            elif defensive_scenario:
                movement_status = "blocked_by_defense"
            else:
                movement_status = "allowed"
            current_state["mode"] = "stationary_defense" if defensive_scenario else "navigate"
            off_spot_enemies = [
                enemy for enemy in current_state["enemies"]
                if enemy["position"] != "on_spot"
            ]
            if must_aim_after_shot and not off_spot_enemies:
                must_aim_after_shot = False
            combat_status = _combat_status(current_state["enemies"], must_aim_after_shot)
            current_state["combat_status"] = combat_status
            current_state["movement"] = {
                "status": movement_status,
                "goal_direction": goal["position"] if goal else "not_visible",
                "goal_distance": goal["distance"] if goal else "not_visible",
                "last_action": last_action,
                "progress": "stuck" if stuck_steps >= 2 else "normal",
            }
            decision_started = time.monotonic()
            model_action, payload, response = client.choose(
                current_state,
                question_id="doom_action",
                instructions=(
                    "Choose exactly one action (one key). State values are discrete. "
                    "HARD SHOOTING RULE: when combat_status is shoot_ready, attack "
                    "now. In every other combat_status, do not attack. If status is "
                    "aim_required, turn toward an enemy tagged left or right. If it is "
                    "turn_after_shot, do not attack again yet; take a turn decision "
                    "toward another off-center enemy first. An enemy tagged on_spot "
                    "is ready_to_shoot. left means turn_left; right means turn_right. "
                    "Prioritize near enemies, but do not over-rank small distance "
                    "differences. When movement.status is blocked_by_enemy or "
                    "blocked_by_defense, never choose move_forward, move_backward, "
                    "move_left, or move_right. When movement.status is allowed and "
                    "there are no enemies, turn toward movement.goal_direction; move "
                    "forward when it is on_spot. If mode is stationary_defense, never "
                    "move. Ignore weapons. Return exactly one criterion."
                ),
                criteria=ACTION_CRITERIA,
            )
            decision_latency_ms = round((time.monotonic() - decision_started) * 1000, 1)
            if model_action not in ACTION_BUTTONS:
                model_action = "wait"
            action_name = model_action
            override_reason = None
            # Enforce unambiguous discrete combat states; SystemOne selects a
            # side when enemies appear on both sides and owns goal navigation.
            action_name, override_reason = apply_movement_safety(
                action_name,
                visible_enemy_count=len(current_state["enemies"]),
                defensive_scenario=defensive_scenario,
            )
            action_name, combat_override = apply_combat_safety(
                action_name,
                current_state["combat_status"],
                current_state["enemies"],
                last_action,
            )
            override_reason = combat_override or override_reason

            last_action = action_name
            vector = action_vector(game, action_name)
            executed_buttons = [
                _button_name(button)
                for button, pressed in zip(game.get_available_buttons(), vector)
                if pressed
            ]
            active_tics = _active_tics(action_name, frame_skip)
            reward = game.make_action(vector, active_tics)
            # Always send an explicit neutral action to release the key before
            # the next SystemOne decision, including attack pulses.
            release_tics = max(1, frame_skip - active_tics)
            reward += game.make_action([0] * len(vector), release_tics)
            if action_name in {"turn_left", "turn_right"}:
                must_aim_after_shot = False
            elif action_name == "attack" and off_spot_enemies:
                # After one shot, require a new aim decision while other
                # visible enemies remain off-center.
                must_aim_after_shot = True
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
