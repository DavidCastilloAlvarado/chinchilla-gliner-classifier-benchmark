"""Contract tests for the Doom playground's SystemOne action boundary."""

from types import SimpleNamespace

import numpy as np

from playground.doom import (
    ACTION_BUTTONS,
    ACTION_CRITERIA,
    _combat_status,
    apply_combat_safety,
    apply_movement_safety,
    apply_navigation_policy,
    doom_state,
    _distance_tag,
    _position_tag,
    _visible_monsters,
)


def _label(
    name: str,
    category: str,
    *,
    world_x: float = 0,
    world_y: float = 0,
    x: float = 280,
    y: float = 200,
    width: float = 40,
    height: float = 60,
):
    return SimpleNamespace(
        object_name=name,
        object_category=category,
        object_position_x=world_x,
        object_position_y=world_y,
        x=x,
        y=y,
        width=width,
        height=height,
        object_health=0,
    )


def test_every_systemone_action_maps_to_at_most_one_button():
    assert set(ACTION_BUTTONS) == set(ACTION_CRITERIA)
    assert all(isinstance(button, str) for button in ACTION_BUTTONS.values() if button)
    assert ACTION_BUTTONS["wait"] == ()


def test_visible_enemies_block_translation_but_allow_turn_and_attack():
    for movement in ("move_forward", "move_backward", "move_left", "move_right"):
        action, reason = apply_movement_safety(
            movement, visible_enemy_count=2, defensive_scenario=False
        )
        assert action == "wait"
        assert reason == "no_movement_while_enemies_visible"

    for combat_action in ("turn_left", "turn_right", "attack", "wait"):
        action, reason = apply_movement_safety(
            combat_action, visible_enemy_count=2, defensive_scenario=False
        )
        assert action == combat_action
        assert reason is None


def test_position_and_distance_values_are_normalized_categories():
    assert _position_tag(100, 640) == "left"
    assert _position_tag(320, 640) == "on_spot"
    assert _position_tag(540, 640) == "right"
    assert [_distance_tag(value) for value in (100, 200, 400)] == ["near", "medium", "far"]


def test_combat_status_requests_a_turn_after_one_shot():
    enemies = [
        {"position": "on_spot", "shoot_status": "ready_to_shoot", "distance": "near"},
        {"position": "left", "shoot_status": "aim_required", "distance": "near"},
    ]
    assert _combat_status(enemies, must_aim_after_shot=False) == "shoot_ready"
    assert _combat_status(enemies, must_aim_after_shot=True) == "turn_after_shot"
    assert _combat_status(enemies[1:], must_aim_after_shot=False) == "aim_required"
    assert _combat_status([], must_aim_after_shot=False) == "no_enemies"


def test_combat_policy_shoots_when_ready_and_aims_toward_available_side():
    enemies = [
        {"position": "on_spot", "shoot_status": "ready_to_shoot"},
        {"position": "left", "shoot_status": "aim_required"},
    ]
    assert apply_combat_safety("turn_right", "shoot_ready", enemies, "turn_left") == (
        "attack",
        "shoot_ready_priority",
    )
    assert apply_combat_safety("attack", "turn_after_shot", enemies, "attack") == (
        "turn_left",
        "aim_toward_left",
    )
    assert apply_combat_safety("turn_right", "aim_required", enemies[1:], "turn_left") == (
        "turn_left",
        "aim_toward_left",
    )
    assert apply_combat_safety("attack", "no_enemies", [], "wait") == (
        "wait",
        "attack_blocked_no_enemies",
    )


def test_partially_visible_living_enemy_still_counts_as_visible():
    enemies = _visible_monsters([_label("Zombieman", "Monster", width=1)])
    assert len(enemies) == 1


def test_navigation_finds_and_follows_a_visible_goal():
    assert apply_navigation_policy(
        "turn_right", mode="navigate", enemy_count=0, goal_direction="not_visible"
    ) == ("move_forward", "navigate_toward_goal")
    assert apply_navigation_policy(
        "move_forward", mode="navigate", enemy_count=0, goal_direction="left"
    ) == ("turn_left", "navigate_toward_goal")
    assert apply_navigation_policy(
        "turn_left", mode="navigate", enemy_count=0, goal_direction="right"
    ) == ("turn_right", "navigate_toward_goal")
    assert apply_navigation_policy(
        "wait", mode="navigate", enemy_count=0, goal_direction="on_spot"
    ) == ("move_forward", "navigate_toward_goal")
    assert apply_navigation_policy(
        "turn_right", mode="navigate", enemy_count=1, goal_direction="left"
    ) == ("turn_right", None)
    assert apply_navigation_policy(
        "turn_right", mode="stationary_defense", enemy_count=0, goal_direction="left"
    ) == ("turn_right", None)


def test_defensive_scenarios_block_all_translation():
    for movement in ("move_forward", "move_backward", "move_left", "move_right"):
        action, reason = apply_movement_safety(
            movement, visible_enemy_count=0, defensive_scenario=True
        )
        assert action == "wait"
        assert reason == "stationary_defense_map"


def test_only_tightly_centered_enemies_are_shoot_ready_regardless_of_distance():
    game = SimpleNamespace(
        get_available_game_variables=lambda: ["POSITION_X", "POSITION_Y", "POSITION_Z"],
        get_available_buttons=lambda: [],
    )
    state = SimpleNamespace(
        game_variables=np.array([0, 0, 0]),
        screen_buffer=np.zeros((3, 480, 640), dtype=np.uint8),
        labels=[_label("DoomImp", "Monster", world_x=400, x=300, width=40)],
    )

    result = doom_state(game, state)

    assert result["enemies"][0]["position"] == "on_spot"
    assert result["enemies"][0]["distance"] == "far"
    assert result["enemies"][0]["shoot_status"] == "ready_to_shoot"
    assert _combat_status(result["enemies"], False) == "shoot_ready"
    assert _position_tag(0.47 * 640, 640) == "left"
    assert _position_tag(0.50 * 640, 640) == "on_spot"
    assert _position_tag(0.53 * 640, 640) == "right"


def test_doom_state_is_compact_and_uses_discrete_enemy_goal_tags():
    labels = [
        _label("Zombieman", "Monster", world_x=100, world_y=20, x=300),
        _label("DeadShotgunGuy", "Gore", world_x=80, world_y=10),
        _label("Shotgun", "Weapon", world_x=30, world_y=5),
        _label("GreenArmor", "Armor", world_x=200, world_y=0, x=300),
    ]
    game = SimpleNamespace(
        get_available_game_variables=lambda: ["POSITION_X", "POSITION_Y", "POSITION_Z", "ANGLE", "PITCH"],
        get_episode_time=lambda: 1,
        get_total_reward=lambda: 0.0,
        get_available_buttons=lambda: list(ACTION_BUTTONS.values())[:-1],
    )
    state = SimpleNamespace(
        game_variables=np.array([10, 20, 0, 90, 0]),
        screen_buffer=np.zeros((3, 480, 640), dtype=np.uint8),
        labels=labels,
    )

    result = doom_state(game, state)

    assert result["player"] == {"health": "normal", "armor": "none", "ammo": "empty"}
    assert result["enemies"] == [
        {
            "id": "enemy_1",
            "type": "Zombieman",
            "position": "on_spot",
            "shoot_status": "ready_to_shoot",
            "distance": "near",
        }
    ]
    assert result["goal"] == [
        {"type": "GreenArmor", "position": "on_spot", "distance": "medium"}
    ]
    assert set(result) == {"player", "enemies", "goal"}
    assert "DeadShotgunGuy" not in str(result)
    assert "Shotgun" not in str(result)
    assert not any(isinstance(value, (int, float)) for value in result.values())
