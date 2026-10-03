# Chrome Dino SystemOne playground

This folder contains an opt-in live test that opens a **headed persistent Google
Chrome** profile, navigates to Dino, reads the current game state, asks the local
SystemOne-compatible endpoint for one action, and then sends that action to the
page.

The loop's order is:

1. Read the Dino `Runner` state from the page (`player`, nearest obstacle,
   distance, speed, and score).
2. POST that state to `POST /v1/systemone`.
3. Read `answers.action.choice` from the response.
4. Apply the response to the latest game state and wait for the distance trigger
   before sending exactly one browser action: `Space` for `jump`, `ArrowDown` for
   `duck`, or no key for `none`. The Runner is never paused or modified; Chrome
   continues its normal animation while SystemOne responds. A small collision
   guard corrects a semantically unsafe classifier choice for a clearly identified
   cactus or airborne obstacle; the raw model choice is logged as `model_action`
   and the executed key as `action`; `decision_latency_ms` records the live HTTP
round-trip time.

The endpoint implementation is not imported or changed. Both playgrounds load
the shared `SYSTEMONE_URL` from the project `.env` file using `python-dotenv`:

```dotenv
SYSTEMONE_URL=http://192.168.18.200:8000/v1/systemone
```

`DINO_SYSTEMONE_URL` remains supported for backward compatibility. The Swagger
URL is documentation; the actual value must be the POST URL ending in
`/v1/systemone`. The model defaults to `fastino/GLiNER2.5-multi-Decide`.

## Run

The direct command is:

```bash
uv run dino
```

It loads `SYSTEMONE_URL` from `.env`, opens the headed persistent Chrome
profile, and runs the live workflow. For pytest-based execution, use:

```bash
RUN_DINO_TEST=1 uv run pytest playground/test_dino.py -s
```

The equivalent Python command is:

```bash
uv run python playground/test_dino.py
```

Playwright uses a persistent profile at `playground/.chrome-profile/` and opens
the requested Dino page. The default is `https://chromedino.com/`, because the
installed Chrome build does not expose `chrome://dino/`. Set `DINO_URL=chrome://dino/`
only when running against a Chrome build that supports that internal page. No
second URL is tried after the requested page fails. The browser is headed for the
duration of the run and closes when the bounded test finishes.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `RUN_DINO_TEST` | unset | Must be `1` for the live test; prevents accidental browser runs. |
| `SYSTEMONE_URL` | project `.env` | Shared SystemOne POST URL; required for live playgrounds. |
| `DINO_SYSTEMONE_URL` | fallback environment name | Backward-compatible Dino endpoint variable. |
| `DINO_MODEL` | `fastino/GLiNER2.5-multi-Decide` | Loaded model identifier. |
| `DINO_URL` | `https://chromedino.com/` | Requested Dino page; no second URL is tried. |
| `DINO_BROWSER_CHANNEL` | `chrome` | Playwright browser channel. |
| `DINO_USER_DATA_DIR` | `playground/.chrome-profile` | Persistent Chrome user-data directory. |
| `DINO_DURATION_SECONDS` | `10` | Maximum play time per run. |
| `DINO_POLL_INTERVAL_SECONDS` | `0.15` | Delay between state/decision cycles. |
| `DINO_DECISION_DISTANCE_PX` | `250` | Ask the model once when an obstacle enters this observation window. |
| `DINO_JUMP_TRIGGER_DISTANCE_PX` | `120` | Press `Space` when a ground obstacle reaches this distance. |
| `DINO_DUCK_TRIGGER_DISTANCE_PX` | `150` | Wait until an airborne obstacle is this close before holding `ArrowDown`. |
| `DINO_DUCK_DURATION_MS` | `450` | How long `ArrowDown` is held. |

## Doom 1 with ViZDoom

The Doom playground uses ViZDoom's native `DoomGame` API and the same SystemOne
endpoint. Each decision sends a compact visual observation (downsampled ASCII
screen, visible labels, position, angle, health, armor, kill count, and ammo) and
executes one real-time action after the response:

```bash
uv run doom
```

The default scenario is `deadly_corridor.cfg`, which provides multiple enemies
and navigable space. The runner enables forward/backward movement, turning,
strafing, and attack buttons even when a scenario exposes only a subset of them.
Turn inputs are one-tic pulses followed by neutral input, so buttons are never
held indefinitely. When living monsters are visible, the runner aims at them and
fires only when their screen bounding box is close to the crosshair. It clears all
currently visible living monsters before resuming forward navigation. Dead corpses
are filtered out, and position telemetry detects forward movement that made no
progress so the player can turn away from a wall.

### ViZDoom scenarios/maps

`DOOM_SCENARIO` accepts a scenario filename or a path to a custom `.cfg` file.
The filenames below are the scenarios shipped in the installed ViZDoom package.
Some scenarios reference an external Doom/Doom II/Freedoom IWAD and will require
that WAD to be installed separately.

| Scenario | Use |
|---|---|
| `deadly_corridor.cfg` | Multiple enemies in a navigable corridor; default. |
| `defend_the_line.cfg` | Several enemies visible across a line; combat-focused. |
| `defend_the_center.cfg` | Defend a central position against spawning enemies. |
| `basic.cfg` | Basic single-target smoke test. |
| `basic_audio.cfg` | Basic scenario with audio state. |
| `basic_notifications.cfg` | Basic scenario with notification state. |
| `simpler_basic.cfg` | Reduced/simplified basic scenario. |
| `rocket_basic.cfg` | Basic scenario using rocket combat. |
| `deathmatch.cfg` | Single-player deathmatch scenario. |
| `multi.cfg` | Multi-player deathmatch scenario. |
| `multi_duel.cfg` | Multi-player duel scenario. |
| `health_gathering.cfg` | Collect health items while navigating. |
| `health_gathering_supreme.cfg` | Extended health-gathering scenario. |
| `take_cover.cfg` | Find and use cover under attack. |
| `my_way_home.cfg` | Navigate toward a goal/home location. |
| `learning.cfg` | Small learning-oriented environment. |
| `predict_position.cfg` | Predict and track moving positions. |
| `cig.cfg` | CIG benchmark scenario. |
| `oblige.cfg` | Oblige-generated level scenario; may need its WAD. |
| `doom.cfg` | Full Doom scenario; requires `doom.wad` if not installed. |
| `doom2.cfg` | Full Doom II scenario; requires `doom2.wad` if not installed. |
| `freedoom1.cfg` | Freedoom 1 scenario; requires the Freedoom IWAD if not bundled. |
| `freedoom2.cfg` | Freedoom 2 scenario; requires the Freedoom IWAD if not bundled. |

Examples:

```bash
# Default: several enemies plus a navigable corridor
uv run doom

# Several enemies visible simultaneously in a stationary combat layout
DOOM_SCENARIO=defend_the_line.cfg uv run doom

# Original one-target smoke test
DOOM_SCENARIO=basic.cfg DOOM_MAX_SECONDS=10 uv run doom

# Use a custom scenario config
DOOM_SCENARIO=/path/to/my_scenario.cfg uv run doom
```

### Doom environment variables

All values can be placed in the project `.env` file or supplied inline in the
shell. `SYSTEMONE_URL` is preferred; the two legacy names are fallback aliases.

| Variable | Default | Purpose |
|---|---:|---|
| `SYSTEMONE_URL` | required | Shared SystemOne `POST` URL. |
| `DOOM_SYSTEMONE_URL` | fallback | Doom-specific SystemOne URL alias. |
| `DINO_SYSTEMONE_URL` | fallback | Legacy endpoint alias shared with Dino. |
| `DOOM_SCENARIO` | `deadly_corridor.cfg` | ViZDoom scenario/config filename or path. |
| `DOOM_HEADLESS` | `0` | Set to `1` to hide the ViZDoom window. |
| `DOOM_MAX_SECONDS` | `30` | Maximum wall-clock duration of the run. |
| `DOOM_EPISODES` | `0` | Maximum episodes; `0` means restart until the time limit. |
| `DOOM_FRAME_SKIP` | `4` | Tics allocated to each decision interval. |
| `DOOM_TURN_TICS` | `1` | Tics for a turn pulse. |
| `DOOM_ATTACK_TICS` | `4` | Tics for an attack; enough for the weapon to fire. |
| `DOOM_AIM_TOLERANCE` | `0.02` | Fraction of screen width allowed between target center and crosshair. |
| `DOOM_SYSTEMONE_TIMEOUT_SECONDS` | `5` | HTTP timeout for each SystemOne decision. |
| `DOOM_MODEL` | `DINO_MODEL` or `fastino/GLiNER2.5-multi-Decide` | SystemOne model identifier. |

For example:

```bash
DOOM_SCENARIO=deadly_corridor.cfg \\
DOOM_MAX_SECONDS=300 \\
DOOM_FRAME_SKIP=4 \\
DOOM_TURN_TICS=1 \\
DOOM_ATTACK_TICS=4 \\
DOOM_AIM_TOLERANCE=0.02 \\
uv run doom
```

By default, a dead or completed episode automatically restarts until
`DOOM_MAX_SECONDS` expires. Set `DOOM_EPISODES=1` to stop after the first episode.
The game advances with `make_action` only after the SystemOne response, so the
request/response/action order remains explicit and real-time.

The project dev dependencies include Playwright and ViZDoom. The Dino workflow
uses the installed Google Chrome channel rather than Playwright's bundled browser,
so a separate `playwright install` is normally not needed.
