# create3-laya

Tell an iRobot Create 3 (or a TurtleBot 4, which is a Create 3 with a camera and lidar) what to
do in plain English. A **local** Laya model works out what you meant; code turns that into
routes, odometry moves, or a visual search; the robot drives it.

```
"go to where the pallets are stored, then come back to where you started"
        │
        ▼  Laya (typed-decisions checkpoint, local GPU): skill, place, object, direction, ...
 find(pallet)                                go_to(start)
        │                                          │
        ▼  code                                    ▼  code
 scan with the camera (local OWLv2 detector)  →  route (A* on a known map, or Nav2)
 → bearing + depth → map position → approach      → driver (code rules, or fine-tuned Laya)
        │                                          │
        ▼                                          ▼
 safety layer at 20 Hz (IR speed limits, bump/cliff reflex)  →  cmd_vel
```

Built on [iRobotEducation/create3_sim](https://github.com/iRobotEducation/create3_sim) (ROS 2
Jazzy + Gazebo Harmonic) and, for vision, [turtlebot4_simulator](https://github.com/turtlebot/turtlebot4_simulator).
The robot geometry, IR sensor poses and intensity model, bumper zones and speed limit come from
create3_sim; both simulators load its `maze.sdf`.

Laya is a non-generative decision model ([convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya),
Apache-2.0): you send it a state and typed multiple-choice questions, and it returns a
probability for every option. It never writes text, so it never invents a plan. Code splits
instructions, extracts numbers and does the geometry. Laya makes the judgments, and the robot
asks you to rephrase when Laya is unsure.

## The launcher

Double-click **`Launcher.cmd`** (it runs on `.venv-laya`; see "Set up the local models").
One window covers everything:

- **Local models:** start and stop Laya and the OWLv2 vision server, with live status. Their
  output goes to `runs\logs\`, so they keep running after the launcher closes if you choose.
- **Simulator:** the 2D maze (drawn live in the window, no WSL needed), the Create 3 in
  Gazebo, or the TurtleBot 4 in Gazebo. Start launches the models it needs, warms Laya up,
  then starts the simulator. Gazebo opens its own window through WSLg.
- **Prompt:** type a task in plain English, or pick one under "Try saying". The conversation
  shows Laya's plan, each step, sightings, clarification questions and refusals. You can type
  while the robot is busy ("go to the east room instead", "when you are done, …"), and "Stop
  robot" halts it at once. The Logs tab has the servers' and Gazebo's output.
- **Speed:** 1×, 2× or 4× real time, for the 2D maze and Gazebo alike. Gazebo runs as fast
  as it can up to that: in the depot about 2× with its window open, 3.5× without. The speed it
  reaches is shown under the simulator, from the robot's heartbeats. Gazebo shares the GPU, so
  a game or other 3D program can slow it to a crawl; the launcher says so when the robot goes
  quiet.
- **Theme:** System (follows Windows' app mode), Light or Dark, bottom right; remembered in
  `runs\launcher.json`. `--theme dark` sets it from the command line.

Closing the window stops the simulator and asks whether to stop the model servers too. A 2D
session saves a replay when it stops. For a scripted start:

```powershell
Launcher.cmd --sim tb4 --start --prompt "go to where the pallets are stored"
Launcher.cmd --sim 2d --mock --start          # 2D maze with the offline stand-in, no models
```

## What you can say

| Kind | Examples | How it runs |
|---|---|---|
| Named place (maze) | "go to the east hall", "head to the north west room" | A* route over the known map |
| Something visible (TurtleBot 4) | "go to where the pallets are stored", "drive to the pallet before you", "find the forklift on your left", "go to the shelves behind you", "drive to the pallet near the shelves with red barrels" | look (that side first, holding still while a few frames come in), then drive to the nearest one, favouring what is ahead. A side named in the text ("before you", "on your left", "behind you") is strict: only what is that way counts. For "X near Y" (or behind / in front of Y) it looks all round, then goes to the X closest to the middle of what matches Y (shelves are long: the pallet in front of them, not one past their end). OWLv2 scores most depot pallets 0.15–0.25, so a weak detection counts once it is seen twice in the same place. When a full look misses (OWLv2 misses pallets 4–5 m off that it finds at 3 m), it looks once more from closer: next to Y, or 2.5 m further the asked-for way. Only then does it take one elsewhere, and the result says so. Sightings from earlier count too, so a pallet the robot stands at, too close for the camera, is still "the pallet before you" |
| Odometry | "go 10 meters forward", "move 2 m to the left", "back up 10 cm", "turn right 90 degrees", "turn around" | turn in place, then a straight drive that holds its line; past the Create 3's 15 cm reverse limit, "backward" turns around first |
| Memory | "come back to where you started", "go back to the pallets" (after seeing them) | `start` and `seen_*` places |
| Other | "return to the dock", "explore three rooms", "wait 10 seconds", "stop" | |

Instructions can chain: "turn around, then drive forward 1 meter, wait 2 seconds, and then
spin right 90 degrees". While the robot is busy, "stop" halts at once. Otherwise Laya judges
from the wording whether a new instruction **replaces** the current task ("go to the west room
instead") or **waits** for it ("when you are done, go to the south room"). Things a robot
without arms can't do ("pick up the cup", "water the plants") are refused.

## Three ways to run it

| | 2D simulator (`sim2d/`) | Create 3 in Gazebo | TurtleBot 4 in Gazebo |
|---|---|---|---|
| World | create3_sim's maze, kinematic | create3_sim maze | depot (no place labels) |
| Senses | IR, bumper, odometry | IR, bumper, odometry | + OAK-D camera, RPLIDAR |
| Navigation | A* + driver | A* + driver | SLAM + Nav2 |
| Needs | Python 3.10+ only | WSL2 / Ubuntu 24.04, ROS 2 Jazzy | + `turtlebot4_simulator` |
| Run | `sim2d\run_sim.py` | `scripts\gazebo_test.ps1` | `scripts\tb4_test.ps1` |

## Set up the local models (once)

```powershell
python -m venv .venv-laya
.venv-laya\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.venv-laya\Scripts\python.exe -m pip install "laya[serve]==0.4.2" pillow scipy
powershell -ExecutionPolicy Bypass -File scripts\serve_laya.ps1      # Laya on 127.0.0.1:8000
powershell -ExecutionPolicy Bypass -File scripts\serve_vision.ps1    # OWLv2 on 127.0.0.1:8001 (TurtleBot 4 only)
```

Pick the PyTorch CUDA build your driver supports (or drop the index URL for CPU, with
`-Device cpu`). The first start of each server downloads its weights from Hugging Face.
`serve_laya.ps1` also serves the fine-tuned driving checkpoint once `models\create3-driving`
exists. Both servers listen on 127.0.0.1 only. `laya-serve` would otherwise bind 0.0.0.0
without authentication.

Timings on an RTX 4060 laptop GPU: about 0.1 s to interpret a task, 87 ms per driving
decision, and 0.4 s per camera frame. On CPU, interpreting a task takes 1–4 s.

## 2D simulator

Needs nothing beyond the system Python:

```powershell
python sim2d\run_sim.py --task "go to the east hall, then come back to the dock"
python sim2d\run_sim.py --task "explore two rooms" --task "@30 stop"          # interrupt at t=30 s
python sim2d\run_sim.py --task "go to the north east corner" --task "@20 when you are done, go to the south room"
python sim2d\run_sim.py --interactive
python sim2d\run_sim.py --driver laya --task "go to the south room"          # fine-tuned Laya drives
python sim2d\run_sim.py --mock --task "..."   # no server: keyword heuristic stand-in, not a model
```

Each run writes `runs\<time>\replay.html`. Open it to scrub through the maze, the route, the
IR rays, each task with Laya's probabilities, and every driving decision.

## Gazebo on Windows (WSL2)

**One-time setup:**

```powershell
wsl --install -d Ubuntu-24.04 --no-launch
# %UserProfile%\.wslconfig:  [wsl2]  networkingMode=mirrored     then: wsl --shutdown
wsl -d Ubuntu-24.04 -u root -- bash -c "WITH_TB4=1 bash '/mnt/c/<path to>/create3-laya/scripts/wsl/setup_ros_jazzy.sh'"
wsl --terminate Ubuntu-24.04
```

The script does the following:

- creates a `ros` user;
- installs ROS 2 Jazzy desktop, Gazebo Harmonic (`ros-gz`) and CycloneDDS, plus the TurtleBot 4
  simulator and navigation with `WITH_TB4=1`;
- clones create3_sim and irobot_create_msgs into `~/create3_ws` and symlinks this repo's
  `create3_laya` there; with `WITH_TB4=1`, also overlays `turtlebot4_description` from source;
- fixes the maze world's name and lowers the simulated sensor rates (both below), builds the
  workspace, and sources `scripts/wsl/ros_env.sh` from `~/.bashrc`.

**Create 3, maze world** (start `serve_laya.ps1` first):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gazebo_test.ps1 "go to the east hall, then come back to the dock"
```

**TurtleBot 4, depot world** (start `serve_laya.ps1` and `serve_vision.ps1` first):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\tb4_test.ps1 "go 2 meters forward" "go to where the pallets are stored" "come back to where you started"
```

Each test script launches Gazebo (the window opens through WSLg) and the Laya node, unpauses
the simulation, waits for the node to report pose and IR (and, on the TurtleBot 4, for Nav2 to
turn lifecycle-active), sends each task, prints its progress, and shuts everything down.
Inside WSL, set `DRIVER=laya`, `KEEP_OPEN=1`, `POSE_SOURCE=odom`, `WORLD=warehouse` or (TurtleBot
4) `GUI=false` for variations. Undocking backs the robot off the dock and turns it around, so
"drive forward" right after leaves the dock behind.

By hand on any Ubuntu 24.04 with ROS 2 Jazzy:

```bash
source scripts/wsl/ros_env.sh
ros2 launch create3_laya laya_create3.launch.py pose_source:=ground_truth   # or laya_tb4.launch.py
ros2 run create3_laya laya_task "go to the east hall"
ros2 topic echo --no-daemon /task_status
```

### What the direct Gazebo test turned up

Each item below broke a real run. All are handled now.

| Problem | Fix |
|---|---|
| Fast DDS discovery fails under WSL mirrored networking (nodes never see each other) | `ros_env.sh` switches to CycloneDDS on loopback |
| **Mirrored networking sends `127.0.0.1` through a 1500-byte-MTU interface to Windows, where IP-fragmented UDP is lost.** CycloneDDS packs up to 14.7 KB per datagram, so every ROS message over ~1.3 KB vanished for best-effort subscribers: lidar scans (no SLAM map, so Nav2 never started), camera frames, and create3_sim's model poses (no ground truth, so the Dock action never saw itself docked) | DDS runs on `127.0.0.2`, which mirrored mode leaves on Linux's own loopback (MTU 64 KiB): `ros_env.sh` adds the address, `cyclonedds_localhost.xml` uses it. Every ROS message used to cross Hyper-V twice, cut to fit the MTU (a 300 KB depth frame was 273 packets); now 0 packets take that path. Without the address it falls back to `127.0.0.1` with datagrams under the MTU |
| The `ros2` CLI daemon hangs under mirrored networking | scripts use `--no-daemon` and rclpy probes |
| create3_sim starts Gazebo paused, and ros2_control can't activate controllers while paused | test scripts unpause as soon as the world is up |
| create3_sim's `maze.sdf` names its world `default`, so IR/bumper/cliff bridges listen on `/world/maze/...` and get nothing | setup renames the world to `maze` |
| Gazebo IR readings carry no `frame_id`, so `/ir_intensity` can't tell its seven sensors apart | the node reads `_internal/ir_intensity_<sensor>` |
| Wheel odometry drifted 0.32 m over a ~20 m maze round trip, enough to steer a map route into a wall corner | `pose_source:=ground_truth` in simulation; on the TurtleBot 4, SLAM localizes |
| WSLg's Mesa renders on the CPU (llvmpipe) | `ros_env.sh` selects D3D12 on the NVIDIA GPU |
| Gazebo renders every IR and cliff sensor as a GPU lidar at 62 Hz, the TurtleBot 4's lidar at 62 Hz and its camera at 30 Hz, each with a GPU round trip under WSLg: the depot ran at 0.03–0.06× real time, GPU or not | setup lowers the simulated rates (IR and cliff 20 Hz, lidar and camera 10 Hz): about 0.75× |
| The simulation stops stepping while the sensors due at that moment render, and under WSLg a render pass costs tens of ms. The Create 3's 11 IR/cliff lidars at 20 Hz made ~20 stops per simulated second: the depot averaged 0.73× and stalled to 0.1–0.3× in bursts | `patch_create3_sim.sh` makes IR and cliff sensors switchable per launch: the TurtleBot 4 leaves both out and reads its IR off the lidar (`ir_from_lidar`: each sensor takes the nearest return in its cone), so the IR speed limits and obstacle stops still work; the maze leaves out cliffs (no drops). The depot holds real time with no stalls |
| Sim time went out at every physics step: `/clock` at 316 Hz to 58 subscriptions, and joint states at 291 Hz, each turned into TF. Idle nodes sat at ~12% CPU each | `clock_relay` passes Gazebo's clock on at 100 Hz, joint states run at 62 Hz: WSL CPU 49% → 35%, the bridges 177% → 69%, the Laya node 54% → 28% |
| WSLg's display can crash (once here, during a run); Gazebo then dies with "no Qt platform plugin" until WSL restarts | `sim_session.sh` checks for the display first and says to run `wsl --shutdown` |
| Nav2's action server appears long before Nav2 accepts goals | the test waits for `bt_navigator` to be lifecycle-active |
| Docking is only accepted while the robot can see the dock | "return to the dock" turns to face it first |
| A script's background jobs start with SIGINT ignored, so the test scripts hung on exit | the launch runs in its own session, stopped with SIGTERM, then SIGKILL after 20 s |
| Odometry moves ("go 2 meters forward", "move 1 meter to the left") were route goals, and Nav2 (or the maze's route driver) curved and stopped early: 1.79 m for 2, a 0.26 m sideways swing, 0.77 m for "1 meter to the left" | turn in place, then a straight drive that steers back onto its line: 1.99 m, 0.00 m off |
| create3_sim's IMU plugin is Gazebo Classic only, so under Harmonic the robot has no gyro, and wheel odometry over-reads turning by ~7%: "turn left 90 degrees" measured 84° | the worlds load Gazebo's IMU system (setup patches the maze; the TurtleBot 4 launch writes a patched depot), `/imu` is bridged, and the node dead-reckons with gyro heading and wheel distance, as a real Create 3 does: 89.5°. Nav2 goals go in SLAM's `map` frame, which agrees with that pose |
| Laya kept two ~2 GB checkpoints in GPU memory (three when a probe named no model), leaving Gazebo under 2 GB of an 8 GB card; the first request after Gazebo crowded it out took 17 s | `serve_laya` preloads only task understanding and loads the driving fine-tune on first use; 3.6 GB in use at idle instead of 6.1 GB |
| create3_sim's maze steps physics at 1 kHz, its main cost: 0.58× real time with the Gazebo window open. Shadows made no difference | setup sets 333 Hz, as the TurtleBot 4 worlds use for the same Create 3 base: 0.82× |

Two of these I first took for create3_sim bugs and worked around: its ground-truth topics
never publishing, and the Dock action never reporting success. Both were the lost datagrams,
and both work unmodified now.

Twice, the Gazebo window failed to open ("D3D12: Removing Device", then "Failed to create
OpenGL context") and took the simulation down with it. The same launch worked on the next
tries, as did Gazebo's own examples; start again if it happens. If WSLg itself went down with
it, the launcher says so: run `wsl --shutdown`, then start again.

`scripts/wsl/motion_check.py` measures commands against create3_sim's ground-truth pose
while a simulation runs. Depot results after the fixes:

| Command | Result |
|---|---|
| drive forward 1 meter / go 2 meters forward | 0.99 m / 1.99 m forward, 0.00 m sideways |
| turn left 90 degrees / turn right 90 degrees / turn around | 89.5° / −89.5° / 179.4° |
| move 1 meter to the left | 89.4° turn, then 0.99 m straight |
| go 1 meter backward | 179.4° turn, then 0.99 m straight |
| back up 10 centimeters | 0.09 m reversed, straight |
| go to where the pallets are stored, then come back to where you started | reached the pallet seen 3.3 m away; back within 0.14 m of the start |

Results with all of the above, through the PowerShell wrappers:

- **Create 3, maze:** "go to the east hall, then come back to the dock" ran all five steps
  (undock, route to the east hall, route back, face the dock, dock) and ended docked.
- **TurtleBot 4, depot:** Nav2 was active 7–10 s after start. "go 2 meters forward" finished.
  For "go to where the pallets are stored", the camera spotted pallets 3–4 m away and Nav2
  drove to the best sighting (OWLv2 score 0.59). "come back to where you started" then drove
  back to the start.

## What Laya does, and what code does

Measured with `tools/eval_laya.py` against the local server:

| Decision | Who | Measured |
|---|---|---|
| Skill, place, object, direction, pace | Laya `typed-decisions`, plus literal yes/no questions combined in code | 35 tuning phrasings: 97% (0 wrong actions); holdout 1: 90% when first scored, 95% now (0 wrong); holdout 2: **75% and 1 wrong action** when first scored untuned, 95% (0 wrong) now, though no longer clean (rules were written after seeing its misses); 17 depot phrasings (tuned): 100% |
| Which object, travel or handling | code reads object names ("pallet", "shelves", ...) and handling verbs ("pick", "carry", "move the box"); Laya for the rest | n/a |
| Replace or queue while busy | Laya, per-question threshold | 8/8 |
| Numbers, units, clause splitting | code (regex) | n/a |
| Route | code (A* on maze.sdf) or Nav2 | n/a |
| Motion every 0.2–0.5 s | code rules by default; or Laya | zero-shot Laya 13–33% (chance); **fine-tuned 97.8%** on 400 held-out states, 100% acceptable on 67 hand-labeled situations |
| Where the object is | OWLv2 boxes → code (bearing from intrinsics, range from depth) | n/a |
| Speed limits, bump/cliff reflex | code, 20 Hz | always on |

"Wrong action" means the robot would have acted on a misreading. Every other miss was
caught by a confidence bar, and the robot asked you to rephrase. Several of the literal
questions exist because Laya's skill answer read words literally. Direction words inside
place names pulled it toward "rotate" ("take me to the bottom **left** corner", 78%), and
object words pulled manipulation requests toward "find" ("pick up the red cup"). Plainer
questions ("does it name a destination?", "turn in place or move along the floor?", "does it
ask to handle an object?") fix those, and each plan records which rule fired. When Laya is
unsure of the action but rates one place likelier than all the alternatives together
(p ≥ 0.5), the robot goes there: "go back to your starting point" scored skill *stop* at 0.18
but place *start* at p 0.82, and "go back to the pallets" picks the remembered sighting. Laya
also only ever sees 9 places, the most its calibration covers alongside "no place".

Words that name things are read by code, like numbers. Laya took "drive to the pallet before
you" for a forklift, so an object named in the text settles the object answer. Its "does it
ask to handle an object?" answer put driving *to* shelves or a door at 0.51–0.57, just over
its 0.5 bar (real handling requests: 0.61–0.86), and "move the box" at 0.47, under it (so the
robot drove 0.5 m). So "go
to …" with no handling verb counts as travel, and a handling verb ("pick", "carry", "move the
box") as handling, whatever the score. Sides are read from the words as well: "before you",
"ahead" and "in front of you" mean ahead, while "behind the shelves" or "in front of the
door" name a reference, not a side. Fresh phrasings still find gaps. The principled next
step for language is a fine-tune like the driving one.

## Fine-tuning Laya to drive

```powershell
python tools\make_driving_data.py --rows 3000 --out data\driving_train.jsonl --seed 1
python tools\make_driving_data.py --rows 400 --out data\driving_eval.jsonl --seed 2
# stop the GPU Laya server first: training needs the VRAM
.venv-laya\Scripts\python.exe tools\train_driving.py --base <hf-cache>\models--convaiinnovations--laya\snapshots\<rev>\typed-decisions
```

The data is the code driver's own decisions on randomized situations, rendered in exactly the
run-time state format. `train_driving.py` wraps Laya's trainer to train the top 12 of 28
encoder layers plus the decision head with cross-entropy. It took 6 epochs and about 40
minutes on an RTX 4060 laptop GPU, and moved held-out accuracy from 32% to 97.9%
(calibration error 0.013). Two other setups failed on 8 GB:

- **Full fine-tune:** spilled into shared memory and ran at about 1 item/s.
- **Head-only (`--freeze-encoder`):** reached only 45%.
- **Top 6 layers with Laya's default loss:** reached 57%.

## Evaluate and test

```powershell
python tools\eval_laya.py --model typed-decisions --suite tasks -v        # also: tasks-holdout, tasks-holdout2, tasks-depot
python tools\eval_laya.py --model create3-driving --suite heldout
cd create3_laya; python -m unittest discover -s test                     # 74 unit tests
python sim2d\test_sim.py                                                  # 4 simulator tests
.venv-laya\Scripts\python.exe launcher\test_launcher.py                   # 7 launcher tests
```

## Layout

```
create3_laya/                   ROS 2 package (ament_python); the core has no ROS imports
  create3_laya/create3.py       Create 3 constants and sensor models from create3_sim
  create3_laya/world.py         SDF loader         planner.py   A* + route following
  create3_laya/pilot.py         labelled state, code driver, Laya driver, safety layer
  create3_laya/tasks.py         plain English -> steps (Laya + regex + combination rules)
  create3_laya/mission.py       runs steps; dock/undock/Nav2/find via host handlers
  create3_laya/vision.py        detection geometry, semantic memory, find behaviour
  create3_laya/laya_client.py   laya-serve client  mock_laya.py  offline stand-in
  create3_laya/laya_node.py     ROS 2 node         ros_extras.py Nav2 + camera (TurtleBot 4)
  create3_laya/task_cli.py      `laya_task`        ros_convert.py message conversions
  create3_laya/clock_relay.py   Gazebo's clock to ROS /clock at 100 Hz
  launch/ laya_create3.launch.py, laya_tb4.launch.py      config/ maze_places.json
launcher/                       app.py (Tk window), backends.py (servers, live 2D sim, WSL session),
                                theme.py (light and dark looks)
sim2d/                          2D simulator, replay viewer, vendored maze.sdf (BSD-3, NOTICE)
vision/serve_vision.py          local OWLv2 detection server
tools/                          eval harness, driving-data generator, train_driving.py
scripts/                        serve_laya / serve_vision / gazebo_test / tb4_test, wsl/ setup
                                and sim_session.sh (the launcher's Gazebo session),
                                motion_check.py (commands vs. ground truth),
                                patch_create3_sim.sh (switchable sensors, joint-state rate)
```

## Limits

- Places are a fixed catalog per world, plus `start`, `dock`, and things the camera has seen.
  Laya chooses among them and can't name new ones.
- The bare Create 3 has no camera, so "find" needs the TurtleBot 4. Without lidar, its
  odometry drifts, which is why simulation runs use ground truth.
- Speed in WSL on an RTX 4060 laptop (Ryzen 7 8845HS), measured by Gazebo:

  | | before | now, 1× asked | fastest asked for |
  |---|---|---|---|
  | TurtleBot 4 depot, window open | 0.73× idle, 0.71× searching, stalling in bursts | 1.00×, no stalls | ~2× (2× asked), ~2.3× at most |
  | TurtleBot 4 depot, no window | | 1.00× | 3.5× (4× asked) |
  | Create 3 maze, window open | ~0.8× | 1.00× | ~2× on average, with stalls (its 7 IR sensors still render) |

  Rendering runs on the NVIDIA GPU through D3D12 (both Gazebo processes load NVIDIA's
  driver), Laya and OWLv2 on CUDA. Anything else on the GPU slows Gazebo directly: with a game
  running, the depot fell to 0.008×. See the Gazebo table for what was slowing it.
- License: `TODO` in package.xml. Pick one before sharing. `sim2d/worlds/maze.sdf` is
  iRobot's (BSD-3-Clause).
