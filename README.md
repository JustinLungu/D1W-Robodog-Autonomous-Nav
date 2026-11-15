# D1W-Robodog-Autonomous-Nav

Autonomous navigation experiments for the AgiBot D1 wheeled robot dog: camera-based object detection, visual servoing approach, and posture control via the high-level SDK.

This README takes you **from zero (one robot in a box) to the robot standing up under your code**.
Follow it top to bottom the first time.

> **Robot model:** AgiBot D1 *wheeled* version. In the SDK this is **`zsl-1w`**.
> Ignore everything in the SDK labelled `zsl-1` (that's the legged version, with a different library).

---

## 0. What you need

| Item | Notes |
|---|---|
| AgiBot D1 wheeled robot + charger | Charge it to > 50 % before you start |
| Handheld remote (joystick) | For driving by hand. It does **not** work while one of our programs is connected (§6): the laptop is the kill switch then |
| Laptop with **Ubuntu 22.04, x86_64** | The SDK's Python module is built for **Python 3.10** only, which is Ubuntu 22.04's default |
| Clear floor space | At least 3 × 3 m, nothing fragile |
| *(Optional)* Ethernet cable | Lower lag than Wi-Fi. See [§8](#8-optional-ethernet-instead-of-wi-fi) |

---

## 1. Laptop: install dependencies

```bash
sudo apt update
sudo apt install -y build-essential cmake git curl \
    libeigen3-dev libboost-all-dev ffmpeg
```

The SDK needs **Boost 1.74** exactly. That's what Ubuntu 22.04 ships. With an older Boost, `initRobot()` can hang.

Install [uv](https://docs.astral.sh/uv/) (the Python package manager this repo uses):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Check the toolchain versions:
```bash
uv --version
cmake --version     # >= 3.8
gcc --version       # >= 11
```

> The SDK's Python module only loads on **Python 3.10**. The repo pins that (`.python-version`, `requires-python = "==3.10.*"`), and uv uses the system `python3.10` or downloads it automatically. You don't need to manage it yourself, and an active conda environment doesn't matter.

---

## 2. Get the code

Clone the vendor SDK **next to** this repo. This repo never contains the SDK itself.

```bash
mkdir -p ~/robodog && cd ~/robodog
git clone https://github.com/AgibotTech/agibot_D1_Edu-Ultra.git
git clone https://github.com/JustinLungu/D1W-Robodog-Autonomous-Nav.git
```

Expected layout:
```
~/robodog/
├── agibot_D1_Edu-Ultra/          # vendor SDK (prebuilt libs, headers, demos, docs)
└── D1W-Robodog-Autonomous-Nav/   # this repo
```

Set up the Python environment. This creates `.venv/` with Python 3.10 and the exact package versions from `uv.lock`:
```bash
cd ~/robodog/D1W-Robodog-Autonomous-Nav
uv sync
```

Run anything with `uv run`, which uses the project environment without activating it:
```bash
uv run python -c "import cv2, numpy; print('ok')"
```
To add a dependency, use `uv add <package>`, then commit both `pyproject.toml` and `uv.lock`.

---

## 3. Robot: first power-on

1. Put the robot on the floor, lying down, with space around it. Power it on.
2. Watch the **tail light**:

| Tail light | Meaning |
|---|---|
| Solid blue | Booted, ready ✅ |
| Blinking blue | Still booting. If it never stops, the network config is broken (see [§9](#9-troubleshooting)) |
| Solid green | SDK has control |
| Solid white | Lab mode (stunts allowed) |
| **Solid yellow** | **Battery < 20 %. SDK control is refused.** Charge it |
| Blinking yellow | Battery < 5 %. Charge now |
| Solid red | Joints too hot. Let it cool down |
| Blinking red | System fault. Contact AgiBot support |

3. **Test the remote first.** Stand the robot up, drive it a little, lie it down. If the remote doesn't work, don't go further.

---

## 4. Connect the laptop to the robot (Wi-Fi)

1. Join the robot's Wi-Fi. The name and password are on the **label on its right side or belly**:
   ```bash
   nmcli dev wifi connect "<ROBOT_SSID>" password "<ROBOT_WIFI_PASSWORD>"
   ```
2. Check you got an address on the robot network. All commands below detect it automatically with the same `grep`:
   ```bash
   ip -4 -br addr | grep -oP '192\.168\.234\.\d+' | head -1
   ```
   Expect one address like `192.168.234.8`. Nothing printed = not on the robot's Wi-Fi.
3. Check you can reach the robot:
   ```bash
   ping -c 3 192.168.234.1
   ```

| Interface | Robot address | Gives your laptop an address automatically? |
|---|---|---|
| Wi-Fi | `192.168.234.1` | Yes |
| Ethernet | `192.168.168.168` | No (you set a fixed address yourself) |

---

## 5. Robot: one-time configuration (over SSH)

**Why this is needed:** the robot doesn't send its data to whoever connects. It sends it to **one** address written in its config file (`target_ip`), which is `127.0.0.1` (the robot itself) out of the box. Until you point it at your laptop, every SDK program fails, our code and the vendor demo alike, with:
```
client bind success
connect time out
error async_receive_from
```
SSH user and password are both `firefly` (vendor default). Everything is run **from your laptop**, so you never have to type commands on the robot.

**5.1 Check the firmware and point the robot at your laptop** (one command):
```bash
MYIP=$(ip -4 -br addr | grep -oP '192\.168\.234\.\d+' | head -1); ssh -t firefly@192.168.234.1 "grep -oP 'motion-control_\K[^_]+' /etc/release/*[^rootfs]*.yaml; sudo cp -n /opt/export/config/sdk_config.yaml /opt/export/config/sdk_config.yaml.bak && sudo sed -i 's/^target_ip:.*/target_ip: \"$MYIP\"/' /opt/export/config/sdk_config.yaml && cat /opt/export/config/sdk_config.yaml"
```
The first time, answer `yes` to the host-key question, then enter the password. What we got:
```
/etc/release/622002862AIC.yaml:0.5.1-xgw      ← firmware version (xgw = wheeled)
target_ip: "192.168.234.8"                    ← your laptop's address
target_port: 43988
```
| Firmware you need | Why |
|---|---|
| **≥ 0.3.1** | Minimum for the `zsl-1w` SDK |
| **≥ 0.3.4** | Vendor docs list it for the remote takeover (LT+RT+A / L2+R2+A). On ours (0.5.1) the takeover did not work while the SDK was connected (§6.5) |

Ours is `0.5.1-xgw` ✅. If yours is older, ask AgiBot support for an update.
`cp -n` keeps the original config as `sdk_config.yaml.bak` (and never overwrites an existing backup).
Don't change `target_port`, and never delete `target_ip` (the robot's motion control won't start without it).

**5.2 Reboot the robot** (lying down). The setting only takes effect after a reboot. This just restarts the robot's computer, like power-cycling it: nothing is deleted, and the remote stays paired.
```bash
ssh -t firefly@192.168.234.1 "sudo reboot"
```
The connection closes. Wait for a **solid blue** tail light (about 1–2 min) and rejoin the robot's Wi-Fi if the laptop dropped off. Then check with §6.1.

> ⚠️ **Firmware updates reset this config.** Redo §5 after every update.
> ⚠️ One laptop at a time. If your laptop's address changes (e.g. another laptop took `.8`), redo §5.

---

## 6. First tests (in this order)

**The laptop is the kill switch.** While any program is connected to the robot's SDK (the vendor demo, and even our read-only `robodog-status`), the joystick does nothing, and the takeover combo (LT+RT+A / L2+R2+A) didn't work on our robot either. The joystick works again once the program has exited. So during every SDK test:

| To stop the robot | How |
|---|---|
| Teleop | **let go of the key** = stop; **SPACE** = e-stop (stop + lie down, latched); **Esc** = lie down and quit; clicking away from its window = stop |
| Stand test | **SPACE** = e-stop; **q** / Ctrl+C = lie down and quit |
| Vendor demo | **`2` ↵** = lie down, **`0` ↵** = lie down and exit |
| Program crashes / Wi-Fi drops | The robot's own watchdog should lie it down after ~3 s (tested in [TESTING.md](TESTING.md)) |

Whoever runs the laptop is the safety operator: hands on the keyboard. Keep the joystick on anyway, for after the program ends.

**Before trusting any driving, run the reliability tests in [TESTING.md](TESTING.md)**: step by step, with what you should see.

**Battery:** the robot refuses SDK control below **20 %** (yellow tail light), and our code refuses to stand below **25 %**. **Charge to ≥ 50 % before any motion test.** The vendor demo has no battery check at all.

### 6.1 Read-only status (nothing moves)
Reads the robot's state. Your laptop's address on the robot's Wi-Fi is detected automatically:
```bash
cd ~/robodog/D1W-Robodog-Autonomous-Nav && uv run robodog-status
```
The first run creates `config.machine.yaml` (this laptop's settings, ignored by git: see §7).
**Before §5** (or with a wrong `target_ip`), you get this, and nothing else happens:
```
laptop IP: 192.168.234.8
connecting: laptop 192.168.234.8:43988 -> robot 192.168.234.1
ROBOT SDK is Running, APP Command or other input is InValid!
mp_recv_cp: 192.168.234.8:43988
client bind success
connect time out
error async_receive_from
...
robodog.safety.SafetyError: could not connect: check target_ip on the robot matches laptop_ip, and that the robot was rebooted
```
**After §5**, it works:
```
laptop IP: 192.168.234.8
connecting: laptop 192.168.234.8:43988 -> robot 192.168.234.1
ROBOT SDK is Running, APP Command or other input is InValid!
mp_recv_cp: 192.168.234.8:43988
client bind success
connect success!
connected
connected=True  battery=22%  <-- LOW, charge before moving  mode=0 (limp/lying)  rpy=(+0.21, -0.01, -0.32)
```
- `ROBOT SDK is Running, APP Command or other input is InValid!` is printed by the vendor library on **every** start. It's a notice that remote/app input is ignored while the SDK is connected, not an error.
- `rpy` = roll, pitch, yaw in radians. Tilt or turn the robot by hand and run it again: the numbers follow.

**Watch mode + remote check:**
```bash
uv run robodog-status --watch
```
While it prints every 0.5 s, stand the robot up and lie it down with the remote. Note whether the remote still works, the tail light colour, and the `mode` numbers. Stop with **Ctrl+C**.

### 6.2 Camera stream (nothing moves)
```bash
ffplay -rtsp_transport tcp -fflags nobuffer -flags low_delay rtsp://192.168.234.1:8554/test
```
You should see the front camera. Press `q` to close.

### 6.3 Vendor demo: stand up and lie down only
Checks the robot obeys SDK commands using AgiBot's own code, before ours. If this works and a later test fails, the problem is in our code.

What to know about their demo ([highlevel_demo.py](../agibot_D1_Edu-Ultra/demo/zsl-1w/python/examples/highlevel_demo.py)):
- It **stands the robot up by itself** about 3 s after starting, before showing a menu.
- It prints `Successfully initialized robot connection.` **even when it isn't connected.** If the robot doesn't stand within ~5 s, it isn't connected (go back to §5).
- No battery check, no e-stop key. **Ctrl+C skips its clean exit**, so quit with `0`.
- **Don't use `highlevel_demo_ultra-w.py`.** It loads the legged robot's library and calls `jump()` and `backflip()`.

Put your laptop's address into their demo (local edit; undo later with `git checkout -- highlevel_demo.py`):
```bash
cd ~/robodog/agibot_D1_Edu-Ultra/demo/zsl-1w/python/examples && MYIP=$(ip -4 -br addr | grep -oP '192\.168\.234\.\d+' | head -1) && sed -i "s/app.initRobot(\"127.0.0.1\", 43988, \"127.0.0.1\")/app.initRobot(\"$MYIP\", 43988, \"192.168.234.1\")/" highlevel_demo.py && grep -n initRobot highlevel_demo.py
```
Expect: `38:        app.initRobot("192.168.234.8", 43988, "192.168.234.1")`

Run it (robot lying down, 2 m clear, remote in hand):
```bash
python3 highlevel_demo.py
```
| You do | Expect |
|---|---|
| (start) | `Successfully initialized...`, `Robot is standing up...` → **robot stands up** |
| wait ~10 s at the menu, type nothing | Note: does it stay standing? (The demo sends nothing while waiting) |
| `2` ↵ | `Executing: Lie Down` → lies down |
| `1` ↵ | Stands up |
| `2` ↵ | Lies down |
| `0` ↵ | `Exiting demo. Robot will lie down.` → lies down, goes limp, script ends (~6 s) |

**Only use `1`, `2` and `0`.** 3–8 drive, 9 makes it go limp, 10 is a trick.
Afterwards, stand up and lie down once with the remote to check it's back in control.

### 6.4 Our stand-up test
The first motion through **our** code ([stand_test.py](src/robodog/cli/stand_test.py)). It stands up through `SafeDog`, holds still for 60 s while printing the status every second, then lies down. It never drives. For the first 3 s after standing up SafeDog sends nothing (the stand-up is still in progress); after that it only sends `move(0, 0, 0)` to keep it standing.

Turn motion on **for this laptop** (and back off afterwards):
```bash
sed -i 's/^allow_motion: false/allow_motion: true/' config.machine.yaml && grep allow_motion config.machine.yaml
```
Run it (robot lying down, 2 m clear):
```bash
uv run robodog-stand-test --allow-motion
```
It connects, prints the battery, and **asks for ENTER before standing** (Ctrl+C there aborts). While it runs:

| Key | Effect |
|---|---|
| **SPACE** | E-stop: speed 0, then lies down; latched (script stops sending anything else) |
| **q** | Lie down and quit |
| Ctrl+C | Lie down and quit |

Turn motion back off:
```bash
sed -i 's/^allow_motion: true/allow_motion: false/' config.machine.yaml && grep allow_motion config.machine.yaml
```

### 6.4b Keyboard driving (`robodog-teleop`)
Drive the robot from the laptop. It moves **only while a key is held**: let go and it stops immediately. Starts ramp up over a fraction of a second; stops never do. Everything goes through `SafeDog` (speed caps, dead-man switch, battery, e-stop).

Try the controls without the robot first:
```bash
uv run robodog-teleop --fake
```
On the robot (motion on in `config.machine.yaml`, see 6.4; 2 m clear):
```bash
uv run robodog-teleop --allow-motion
```
A small window opens. **Click it**: keys only count while it has focus, and clicking away stops the robot.

| Hold | | Click | |
|---|---|---|---|
| **W** / ↑ | forward | **R** | stand up |
| **S** / ↓ | backward | **F** | lie down |
| **A** / ← | turn left | **1 2 3** | speed level: 33 % / 66 % / 100 % of the caps |
| **D** / → | turn right | **SPACE** | **E-stop**: stops and lies down, latched until restart |
| | | **H** | shake hands (like the vendor demo's `10`): ~4 s trick, then stands up again; movement keys ignored ~7 s |
| **Q** / **E** | sideways left / right | **Esc** | lie down and quit (closing the window does the same) |

Speed level 1 is the default. The caps are `max_vx` / `max_vy` / `max_yaw` in `config.local.yaml`.
After **R**, the movement keys are ignored for **3 s** while the robot stands up (the window says `standing up...`); then it ramps up from 0 as usual.

**Recording a drive** for later tests: add `--record` (and optionally a `--note`):
```bash
uv run robodog-teleop --allow-motion --record --note "hallway, chair legs, glass door"
```
Each run gets its own folder (git-ignored):

| File | Contents |
|---|---|
| `recordings/<date_time>/video.mkv` | Robot camera, saved as sent (no re-encoding) |
| `commands.csv` | ~30 rows/s: laptop time, keys held, speed asked for and actually sent, standing, battery, mode, fault, events (`stand`, `estop`, `focus_lost`...) |
| `meta.json` | Start time, camera URL, speed caps, your note, `video_start_unix` |

To match them: laptop time of a video frame = `video_start_unix` + the frame's time in `video.mkv`. Compare that with `t_unix` in the CSV. That's when the frame *arrived* on the laptop; the camera's own delay (~0.2–0.5 s) is not subtracted. The window shows `● REC mm:ss`, or `REC: VIDEO STOPPED` if the camera stream dropped (see `ffmpeg.log` in the folder; the commands keep being recorded). Use `--camera /dev/video0` to record a webcam instead, e.g. with `--fake`.

### 6.5 Results on our robot
| Check | Result |
|---|---|
| Firmware | `0.5.1-xgw` |
| SDK connection after §5 | ✅ `connect success!` |
| `mode` while lying | `0` |
| Joystick while an SDK program is connected (even read-only) | ❌ ignored; tail light stays **blue**; works again right after the program exits |
| Takeover combo during SDK control | ❌ didn't work → the laptop is the kill switch |
| Vendor demo: stand, forward, back, turn, handshake, lie down, exit | ✅ all as expected |
| `mode` while standing / moving | _todo_ |
| Reliability tests ([TESTING.md](TESTING.md)) | _todo: record results there_ |

### 6.6 Depth from the camera (Depth Anything V2, experimental)

A live view of what [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) estimates from the robot's RGB camera. It's **vision only**: it never connects to the SDK and is **not** a safety feature. Use it to see how well monocular depth works on your floors and obstacles.

Install the vision extra **once**. Pick the line that matches the laptop:
```bash
uv sync --extra vision-cpu    # no NVIDIA GPU: CPU-only PyTorch (~200 MB)
uv sync --extra vision-gpu    # NVIDIA GPU: CUDA PyTorch (~3 GB)
```
The first run downloads the model weights (~100 MB, to `~/.cache/huggingface`).

```bash
uv run robodog-depth                 # robot camera: rtsp://<robot_ip>:8554/test
uv run robodog-depth --source 0      # laptop webcam, to try it without the robot
uv run robodog-depth --size 252      # smaller model input: faster, less detail
uv run robodog-depth --model base    # bigger model, GPU recommended
uv run robodog-depth --width 960     # smaller window (default max 1280 px wide); you can also drag to resize
uv run robodog-depth --headless --frames 20   # no window, print timings only
```
The window shows the camera image and the depth image side by side: **bright/yellow = close, dark/purple = far**. Press `q` to quit. The values are **relative**, with no units, and they're rescaled every frame. They show what is in front of what, not distances in metres.

The overlay shows the model's FPS, inference time and *age* (how old the frame was when its depth appeared). The camera's own network delay comes on top of that age.

Rough speed of the `small` model (default input: 308 px on CPU, 518 px on GPU):

| Hardware | Input | Time per frame |
|---|---|---|
| Laptop CPU (8 threads) | 518 px | ~2 s |
| Laptop CPU (8 threads) | 308 px | ~0.4 s (~2 FPS) |
| Small NVIDIA GPU | 518 px | expect ~20–50 ms |

> **GPU laptop:** check that `nvidia-smi` works first. Current PyTorch CUDA builds don't support very old cards (GTX 9xx/10xx). If `robodog-depth` reports `cpu` or errors on such a card, it needs an older PyTorch build.

> If `uv run pytest` fails with `No module named 'lark'`, your shell has ROS sourced, and pytest is loading ROS's plugin. Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest` instead.

> **A plain `uv sync` removes the vision extra** (torch, transformers). Always sync with the extra: `uv sync --extra vision-cpu` (or `vision-gpu`).

### 6.7 Go to a clicked goal (`robodog-goto`)
Click a spot on the floor in the robot's camera view; **hold G** and the robot drives there. Let go and it stops (the goal is kept). Everything goes through `SafeDog`, and all the teleop keys still work in the same window.

How it works:
1. **Click → floor point.** From the camera's height, tilt and lens ([camera_model.py](src/robodog/camera_model.py)) the clicked pixel becomes e.g. "1.51 m ahead, 0.51 m left". Clicks at or above the horizon are refused (not floor).
2. **The goal is stored in the robot's own position frame** (`getPosition()` + heading), so it stays put while the robot moves, also after it has slid out of view under the robot's nose.
3. **Hold G:** turn on the spot if the goal is more than 25° off, otherwise drive and steer toward it, slow down near it ([goto.py](src/robodog/goto.py)). It **stops short, facing the goal**: the robot's middle stops `stop_before_m` (0.55 m) before the clicked point, so with the lens 0.40 m in front of the middle, the **nose ends ~15 cm in front of the object** instead of driving over it. `stop_before_m: 0` drives the middle onto the point (within `arrive_m`, 10 cm).
4. **The picture arrives late** (~0.3 s: encoding + Wi-Fi + decoding) while the robot reacts at once. So go-to keeps the last 3 s of robot poses, and a click is turned into a goal using **where the robot was when that picture was taken** (`camera.latency_s` ago), not where it is now. Otherwise a click while the robot turns lands the goal far off.
5. **The window redraws the goal every frame** from the pose at the picture's time, so the ring stays on the object in the (late) video. If the ring slides off the real spot, the camera delay, the geometry or the odometry is off. **Tune the delay** with **[** / **]** (−/+ 0.05 s): turn the robot on the spot (A/D) with a goal set, and adjust until the ring stays on the object instead of swinging ahead and back. Put the value in `camera.latency_s`.
6. **Cyan arcs on the floor** mark 0.5 / 1 / 1.5 / 2 / 3 m **from the point under the camera** (how you measure with a tape): if the camera values are right, a tape measure agrees with them. All distances shown in the window count from there too; internally the robot steers its middle (0.40 m behind the lens).

| Key | |
|---|---|
| **click the floor** | set the goal |
| **hold G** | go to the goal (let go = stop) |
| **hold W/S A/D Q/E** | drive by hand, like teleop (overrides G) |
| **R** / **F** / **H** | stand / lie down / handshake |
| **C** | clear the goal |
| **[** / **]** | camera delay −/+ 0.05 s (see 5. above) |
| **K** | calibrate: click the bases of objects at the `--calib-distances` (default `1.5,2.5,3.0` m, measured on the floor from the point under the lens), in that order |
| **SPACE** / **Esc** | e-stop / lie down and quit |

Limits (`goto:` in [config.local.yaml](config.local.yaml)): goals over **3 m** refused, **0.15 m/s** and **0.3 rad/s** max, **20 s** driving timeout, stops if the position estimate jumps (> 0.5 m in one step).

**Out of view = driving blind.** The camera can't see the floor closer than ~1.5 m, so the last part is driven from a position estimate alone. Two safety/fallback features for that:
- **Overshoot guard:** go-to counts the forward driving it has *commanded* (speed × time). If that exceeds what the goal needed × 1.25 + 0.4 m, it stops (`[overshoot guard]`), whatever the position estimate says.
- **`goto.odometry`:** `"robot"` = the robot's own estimate (`getPosition()`); `"commanded"` = our count (commanded speed × time along the IMU heading). While G is held, the terminal prints both every 0.5 s (`commanded .. m, robot estimate .. m`), and the panel shows them too. If the robot's estimate lags or stays at 0 while it clearly drives, switch to `"commanded"`. In `"commanded"` mode, hand driving (WASD) clears the goal, because the count only knows go-to's own commands.

**Try it without the robot first:** a simulated robot on a drawn floor:
```bash
uv run --extra vision-gpu robodog-goto --fake
```
R, click inside the cyan arcs, hold G: the ring moves as the simulated robot turns and drives to it.

**On the robot**, in this order (step-by-step in [TESTING.md](TESTING.md), stage 4):
1. **View only** (nothing moves): `uv run --extra vision-gpu robodog-goto`. Click the floor: it shows the distance.
2. **Measure** with the robot standing: the lens height above the floor (`camera.height_m`) and how far the lens is in front of the robot's centre (`camera.forward_offset_m`). Put both in `config.local.yaml`. **All distances are from the robot's centre** (middle of the body): that's the point it turns around and the one its position estimate tracks.
3. **Calibrate:** put objects at **1.5, 2.5 and 3.0 m** (measured on the floor from the point **under the camera lens**; one can be off to the side). Press **K** and click each object's base in that order. It fits the **tilt, field of view and lens type together** (one point can only fix the tilt, and AgiBot's lens spec turned out not to match our camera) and applies them right away: check the cyan arcs against the objects. The terminal prints `lens`, `hfov_deg` and `pitch_deg`; put them in `config.local.yaml`. Our camera looks **up** (negative pitch), so the floor is only visible from ~1–1.5 m ahead.
4. **Check:** bottles at 1, 2 and 2.5 m, click each: the shown distance should be within ~10 %. Then set `camera.measured: true`. **G only drives once that's true.**
5. **Drive:** `uv run --extra vision-gpu robodog-goto --allow-motion` (motion on in `config.machine.yaml`). R, click a spot 1 m ahead, hold G.


### 6.8 Litter detection on recordings (offline, experimental)
Finds litter on the floor in a recorded video and draws boxes around it. It uses [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-tiny), a *zero-shot* detector: you describe what to look for in words, no training needed. Needs the vision extra (§6.6). The first run downloads the model (~700 MB).

```bash
uv run robodog-litter recordings/<run>/video.mkv                       # whole video
uv run robodog-litter recordings/<run>/video.mkv --start 20 --end 40   # try settings on a short piece first
uv run robodog-litter recordings/<run>/video.mkv --show                # watch it while it runs
uv run robodog-litter recordings/<run>/video.mkv --classes "piece of paper, can, plastic bottle"
```
Results go to `recordings/<run>/litter/`: `annotated.mp4` (the analysed frames with boxes) and `detections.csv` (video time, laptop time, class, score, box). Use the laptop time to match detections with `commands.csv` from the same run.

Tuned for the robot's low camera:
- **Floor only.** Only the lower half of the image is searched (`--search-from 0.5`), and a box must end below the dashed line (`--min-bottom 0.65`). This skips bottles and cups on tables.
- **Small things only.** Boxes bigger than 3 % of the frame are dropped (`--max-area 0.03`); big hits were sleeping bags, jackets and mats.
- **Prompt wording matters.** "piece of paper" found the floor scraps far better than "paper". Too many boxes: raise `--threshold` (default 0.25). Missing things: lower it.

Speed: ~5–8 s per frame on a laptop CPU, so it analyses 1 frame per second of video by default (an 87 s video takes ~8–10 min). On a GPU it's well under 1 s per frame, and the default becomes 5 frames per second.

This is a starting point, not a final detector. The next step is to fix and confirm its boxes on our own frames, then train a small, fast detector (YOLO) on them that can run live.

### 6.9 Obstacle check from the camera: forward block (experimental)
Detects things in the robot's path, using only the RGB camera and Depth Anything (§6.6). With `--obstacles` in teleop or go-to, **forward driving is blocked** while something is ahead: SafeDog sets forward speed to 0 whoever asks for it (keys or go-to). **Backward, turning and sideways still work**, to get away. It's a camera-only check with blind spots (see *Known limits*), so keep the e-stop at hand.

```bash
uv run robodog-obstacles                                        # live view: see what it detects, tune it
uv run robodog-obstacles --source recordings/<run>/video.mkv    # the same on a recording
uv run robodog-teleop --allow-motion --obstacles                # teleop with the forward block + camera view
uv run robodog-goto --allow-motion --obstacles                  # go-to with the forward block (also with --trash)
```
**How the block works** ([teleop.py](src/robodog/teleop.py) `ForwardGuard`, [safety.py](src/robodog/safety.py) `forward_guard`): every control tick (30 per second) SafeDog asks the guard, and while it gives a reason, the forward part of every command is 0. **When it isn't sure, it blocks:** an obstacle (held 1 s after the last sighting), no picture yet (forward is blocked until the depth model runs), a picture older than 1.5 s, or a crashed check. The terminal logs `forward BLOCKED: ...` / `forward allowed again`. Teleop shows the camera picture with the marks and a red **FORWARD BLOCKED** bar; go-to **pauses** (the goal is kept, its timeout clock stops) and continues when the path is clear. Go-to runs the check on its own camera stream (no second stream).

**Going to trash** (`robodog-goto --trash --obstacles`): the piece itself is in the robot's path, so the check leaves out the selected piece's box (YOLO's, or CSRT's while YOLO lost it), grown by `obstacles.target_margin` (3 % of the picture) on every side. Those pixels are drawn **green** ("the target") instead of red. Anything else in the path, even right next to the piece, still blocks. If the piece isn't tracked at that moment, nothing is left out and it blocks as usual (safe side). With `--record`, the start and end of each warning go into `commands.csv` (`obstacle_warn near=…`, `obstacle_clear`). If the camera picture is older than 1.5 s the window says so instead of claiming the path is clear.

**How it decides** ([obstacles.py](src/robodog/vision/obstacles.py)):
1. Depth Anything only gives *relative* depth, and the floor right in front is the nearest thing in the picture. So each frame it fits a flat **floor plane** through the floor pixels. That cancels the model's unknown scale and offset, and the robot's pitching.
2. Whatever is clearly **closer than the floor would be at that spot** sticks up from the floor: an obstacle.
3. Its depth says which floor row it stands on, i.e. how far away it is. It counts only if it's inside the **robot-sized window at that distance**: between the path's left and right edge, and no higher than the robot (the robot drives under a table top, but not through its legs).
   The **yellow box** is that window at the warning line, and it's a hard limit: anything left of, right of or above it is ignored, so what you see in the viewer is exactly what decides. Its bottom stays open: below it means nearer than the warning line.
4. It warns if such a thing stands **on or before the warning line** (`warn_row`).
5. If no floor is visible at all, something is covering it: reported as **BLOCKED**, which also warns.

**Settings** are in the `obstacles:` section of [config.local.yaml](config.local.yaml), calibrated for the robot **standing** (see below). Every setting can be tried from the command line first, e.g. `--warn-row 0.98`:

| Setting | Meaning | Standing value |
|---|---|---|
| `path_left`, `path_right` | Where the robot's path (wheel edges + margin) leaves the picture at the bottom | 0.34, 0.64 |
| `vanish_x` | Where the robot's centre line meets the horizon | 0.488 |
| `height_ratio` | Robot top (+ margin) above the camera, relative to the camera's height above the floor. Higher things are driven under | 0.37 |
| `warn_row` | Warning distance as an image row; bigger = closer, above 1.0 = closer than the nearest visible floor | 1.06 = 0.8 m |

**Calibration** (2026-11-15, from three app screenshots in `recordings/calibration_screenshots1/`: a measuring tape laid from just in front of the right wheel, 4.3 m straight ahead, parallel to a floor line through the robot's middle):
- **Lens:** the vendor spec gives HFOV 122°, VFOV 70° (and DFOV 122°), which only fits a fisheye ("equidistant") lens: ~626 px focal length at 1334 px wide.
- **Fitted per posture** from where the two lines meet (the horizon) and where the 4.3 m tape end appears:

| | Camera height | Tilt | Nearest visible floor | Tape (wheel edge) from centre |
|---|---|---|---|---|
| Standing | 40 cm | 12° up | 0.93 m | 20 cm |
| Crawling | 23 cm | 12° up | 0.55 m | 19 cm |
| Lying | 17 cm | 15° up | 0.47 m | 19 cm |

  The wheel edge comes out at 19–20 cm in every posture, matching the vendor's 360 mm robot width, which supports the model. The camera is **centred** on the robot.
- **Margins:** 10 cm beside each wheel, 10 cm above the robot (top 42 cm, vendor spec). Each setting is the most cautious value over 0.45–3 m distance and floor-to-roof height, because the checker uses straight path edges while the fisheye bends them.
- **Warning distance 0.8 m from the camera:** about 1 s of camera + processing delay at up to 0.6 m/s, plus a little braking. Other distances (standing): 0.88 = 1.5 m, 0.93 = 1.2 m, 0.98 = 1.0 m, 1.06 = 0.8 m, 1.11 = 0.7 m, 1.19 = 0.6 m.
- **Other postures** need other values (the config file lists them); there is no automatic switching yet. Crawling's height is an estimate.
- **Uncertainty:** where exactly the tape starts relative to the camera isn't known; ±15 cm there changes the camera height by ±1.5 cm. The screenshots are from the vendor app, assumed to show the full 16:9 camera frame like the RTSP stream; check once that the yellow box looks the same in `robodog-obstacles`.

**Speed:** ~0.35 s per check on a laptop CPU (2–3 per second), plus the camera's own delay of ~0.3–0.5 s. At 0.3 m/s the robot travels 20–25 cm before a warning can appear. A GPU makes the checks several times faster.

**Should the frame be cropped before Depth Anything?** No. Measured on 13 frames against a larger reference model:

| Depth computed on | Time (CPU) | Depth error in the path area | Correct verdicts |
|---|---|---|---|
| **Full frame, 308 px, cut afterwards (used)** | 0.32 s | 3.6 % | 13 / 13 |
| Cropped first, 308 px | 0.52 s | 2.5 % | 13 / 13 |
| Cropped first, 252 px | 0.35 s | 2.9 % | 13 / 13 |
| Cropped first, 196 px | 0.23 s | 3.9 % | 11 / 13 |
| Full frame, 196 px | 0.18 s | 6.0 % | 12 / 13 |

Cropping isn't faster (the model scales by the short side, so a wide crop has *more* pixels), and although the depth inside the crop is slightly sharper, objects at medium distance come out too close once the ceiling and far walls are gone, which leads to false warnings. So the "cut" happens after the depth step, as the robot-path window described above.

**Known limits:** glass, mirrors, very thin things (chair legs, cables) and low flat things can be missed; a floor that isn't flat (ramps, steps, thick mats) confuses the floor fit. With `--record` and `--obstacles` together the camera stream is opened twice (2 × ~5 Mbit/s).

### 6.10 The robot's own person detection (just for a look)
The robot already runs a person detector on its NPU (`detection_2d_node`, YOLO11n, classes person and car). It's meant for a "follow me" feature and nothing in our code uses it. This shows its boxes on the live camera, by **listening** to its ROS 2 topic `/perception/detection2d` from the laptop: nothing is sent to the robot, nothing runs on it, no ROS install needed.

```bash
uv sync --extra vision-cpu --extra robot-bus   # once (list every extra you use, or uv sync removes it)
uv run robodog-persons                         # window: camera + the robot's boxes
uv run robodog-persons --headless              # print the boxes instead
```
Each box shows the robot's track number (one person keeps it across frames) and the confidence. The class name is always empty: the vendor doesn't fill it in. It runs at 10 Hz and misses people quite often. How it connects: [robotbus.py](src/robodog/robotbus.py).

### 6.11 Trash detection and tracking (`robodog-trash`)
Run the TACO-trained YOLO detector and ByteTrack on a recorded video, or view the live camera. This only detects and tracks trash; it does not move the robot.

```bash
uv sync --extra vision-cpu                 # or vision-gpu on an NVIDIA laptop
uv run robodog-trash recordings/<run>/video.mkv
uv run robodog-trash recordings/<run>/video.mkv --headless --save annotated.mp4 --csv detections.csv
```
For a recording made while the robot was crawling, add `--posture crawl` so distance estimates use the lower camera height. The default posture is `stand`. The first run may download the model weights.

Each tracked piece gets a box labelled `#ID class confidence distance`. The distance is measured on the floor from the point under the camera lens, from the box's bottom-centre (where the piece touches the floor) and the camera geometry in `config.local.yaml` (`camera.height_m` standing, `camera.crawl_height_m` crawling). "cut off" means the box touches the picture's edge, so its foot isn't visible and no distance is given. `--csv` writes one row per box; `--start/--end/--step` pick a part of the video; SPACE pauses.

**How it works** ([vision/trash.py](src/robodog/vision/trash.py)): YOLO finds boxes in each frame; ByteTrack links them across frames into tracks with a stable ID, also through a few blurry or weak frames. `trash.conf` is the confidence needed to *start* a track; an existing track is continued with weaker detections too.

**The model** is `trash.model` in `config.local.yaml`: a local `.pt` file or `hf://<repo>/<file>`. The default is a YOLO11s fine-tuned on TACO ([fabiocigaina/TACO-yolo11s](https://huggingface.co/fabiocigaina/TACO-yolo11s), mAP50 0.31): in our first test its results were poor. Any Ultralytics model works, so try others, optionally with `trash.classes` (or `--model` / `--classes` on the command line):
```bash
uv run robodog-trash <video> --model yolo11s.pt --classes "cup, bottle"                     # Ultralytics' COCO model
uv run robodog-trash <video> --model yolov8s-worldv2.pt --classes "cup, can, bottle, paper" # open vocabulary: classes = words to look for
```
The real fix is a model trained on our own frames. A `.pt` file is a Python pickle and **runs code when loaded**, so only use files you trust; the default model's SHA-256 is pinned, and a changed file is refused.

#### Going to a tracked piece (`robodog-goto --trash`)
```bash
uv run robodog-goto --allow-motion --trash                   # robot standing
uv run robodog-goto --allow-motion --trash --posture crawl   # camera at crawl height
uv run robodog-goto --fake --trash --camera <video or 0>     # try the window with no robot
```
The go-to window (§6.7) then shows the tracked boxes. **Click a box** and **hold G**:
1. **While the camera sees the piece, the goal stays locked on it.** Every detection re-measures where it is and corrects the goal ([target.py](src/robodog/target.py)), so a slightly wrong first box and drifting odometry don't add up. The panel shows `TRACKING, goal corrected Nx`. The robot drives at `trash.follow_vx` (0.10 m/s) so the tracker keeps up.
2. **When the piece leaves the view** (cut off by the picture's edge, or not seen for `trash.lost_after_s`), correcting stops and the robot **finishes blind**: straight along the last seen heading (point 4), with the distance from odometry. The panel shows `OUT OF VIEW ... finishing on odometry`. It stops as go-to always does (`goto.stop_before_m`).
3. If the tracker gives the piece a new ID, a track within `trash.reacquire_m` (0.4 m) of the goal is taken as the same piece.
4. **Centring, so the wheels pass either side of it** ([goto.py](src/robodog/goto.py), [target.py](src/robodog/target.py)):
   - **It steers by where it *sees* the piece, not by the map.** Each detection gives the piece's direction from the box's left-right position, turned into a compass heading with the robot's heading (IMU) at the moment that picture was taken. Go-to holds that heading. The estimated goal point is only used for *how far* to drive, because the robot's position estimate drifts sideways (in a simulation with 20 cm sideways drift per metre, steering by the map ended 15 cm off-centre; by the seen heading, 2 cm).
   - **It aligns before going blind.** When the box's bottom gets into the lowest 12 % of the picture (or is cut off), the robot stops and turns on the spot until it points at the piece within 1°: `GOING (aligning)`. Then it drives the last, blind stretch straight along that heading. The heading still comes from boxes cut off only at the bottom, since only their left-right position matters.
   - **Small corrections are pulsed, not dropped.** The robot ignores turn rates below 0.1 rad/s, so a gentle correction is sent as short bursts of 0.12 rad/s whose average is the wanted rate. Within 1° it's left alone, so it doesn't twitch.
5. **If YOLO loses the selected piece for a moment** (blur, odd angle, low confidence), OpenCV's **CSRT** tracker follows its pixels instead, starting from YOLO's last box ([vision/csrt.py](src/robodog/vision/csrt.py)). That box is drawn **amber with `CSRT`** and keeps correcting the goal. CSRT restarts on YOLO's box every frame YOLO still sees the piece, and its answers are dropped after `trash.csrt_max_s` (2 s) on its own, or if it says it lost the object, jumps too far in one frame, or shrinks to nothing. Then the piece counts as not seen (point 2). If YOLO finds the piece again under a new ID on top of CSRT's box, YOLO's box wins. Only the selected piece gets this; the others are YOLO only. `csrt_max_s: 0` turns it off. Needs `opencv-contrib-python`, which is the project's OpenCV (`uv sync` installs it).

Each detection is placed using the pose the robot had when that frame was taken (`camera.latency_s` ago; tune it with **[** / **]**, §6.7), so the robot's own movement in between cancels out. Clicking the floor instead of a box still sets an ordinary fixed goal.

`--posture` only sets the **camera height used for distances**; it does not make the robot crawl. The SDK's `crawl()` had no effect on our robot (tested 2026-11-15), so the robot has to be in that posture by other means, and `camera.crawl_height_m` (0.20, a first guess) should be measured. The camera's tilt is assumed the same in both postures.

---

## 7. Safety layer (`robodog` package)

All our code talks to the robot through **`SafeDog`** ([src/robodog/safety.py](src/robodog/safety.py)), never through the raw SDK.

| Protection | What it does |
|---|---|
| **Motion off by default** | Nothing moves unless `allow_motion: true` is set in **this laptop's** `config.machine.yaml` **and** the code passes `allow_motion=True` (`--allow-motion`) |
| **Speed caps** | Clamped to `max_vx` / `max_vy` / `max_yaw` (default 0.3 m/s, 0.2 m/s, 0.5 rad/s). Values below the SDK minimums become exactly 0 |
| **Dead-man switch** | If your code stops calling `set_velocity()` for `cmd_timeout` (0.3 s), the speed drops to 0 |
| **Keepalive loop** | Sends `move()` at `rate_hz` (30 Hz) while standing, so the robot's 3 s watchdog doesn't trip by accident |
| **Stand-up pause** | For 3 s after `stand()` (`STAND_SETTLE_S`) nothing is sent and `set_velocity()` is refused, so the stand-up isn't interrupted (same wait as the vendor demo). `settled()` says when it's done; teleop shows `standing up...` meanwhile |
| **Battery** | Refuses to stand below `min_battery` (25 %). Stops and lies down if the battery drops below that mid-run |
| **Lost connection / SDK error** | Stops, tries to lie down, and latches a fault. Nothing else is sent until you restart |
| **E-stop** | `estop()` sets speed 0, waits 0.3 s, crouches (`lieDown()`), and 3 s later settles fully down (`passive()`). If lying down fails it goes straight to `passive()`. Latched. Scripts bind it to **SPACE** |
| **Lying down fully** | On this robot `lieDown()` only **crouches** (mode 51); **damping** (`passive()`, mode 0) then lets it settle all the way down, like the vendor demo's exit. `lie_down()`, the e-stop and `close()` all do crouch → 3 s (`LIE_DOWN_S`) → damping; standing up again within those 3 s cancels the last step. No stop command is sent to a robot that's already crouched (it rejects it: `transition to Move failed`) |
| **No stunts** | `jump` / `frontJump` / `backflip` aren't exposed at all. Only the handshake is (`shake_hand()`, copying the vendor demo: nothing sent for 4 s, then stand up again) |

Settings are split in two, so a `git pull` never overwrites someone's laptop setup:

| File | In git? | Contains |
|---|---|---|
| **[config.local.yaml](config.local.yaml)** | ✅ shared | Speed caps, timeouts, battery limit, robot address. Edit in place; everyone gets the change |
| **`config.machine.yaml`** | ❌ ignored | **This laptop only**: `laptop_ip` (`"auto"` = detected), `sdk_dir` (default `../agibot_D1_Edu-Ultra`), `allow_motion` (default `false`). **Created automatically** the first time you run anything |

If your SDK clone is somewhere else, change `sdk_dir` in your `config.machine.yaml`. Putting a per-laptop key in the shared file is an error, so they can't creep back in.
[src/robodog/config.py](src/robodog/config.py) only defines which keys must exist and their types. A missing key, a typo (`max_vX`) or a wrong type (`allow_motion: "false"` in quotes) stops the program with a clear error instead of being silently ignored. If you add a shared setting, add it to `config.local.yaml` and to the class in `config.py`.

Test everything **without the robot**:
```bash
uv run pytest                          # safety + config tests against a fake robot
uv run python tests/dryrun.py                  # scripted session on a fake robot; press SPACE to try the e-stop
uv run python tests/dryrun.py --low-battery    # see the battery protection kick in
uv run python tests/dryrun.py --disconnect     # see the lost-connection protection kick in
```

**Read-only check on the real robot.** It never sends a motion command, so the wheels won't turn:
```bash
uv run robodog-status            # connected / battery / mode / tilt, once
uv run robodog-status --watch    # keep printing until Ctrl+C
```
> While any SDK program is connected, the remote does nothing (even for this read-only check). Stop the script to get the remote back.

Using it in your own code:
```python
from robodog import SafeDog, load_config
from robodog.sdk import make_highlevel

cfg = load_config()
with SafeDog(make_highlevel(cfg.sdk_dir), cfg, allow_motion=True) as dog:
    dog.connect()
    dog.start()                 # keepalive / battery / connection loop
    dog.stand()
    while running:
        dog.set_velocity(vx, vy, yaw)   # call at least every 0.3 s
# leaving the `with` block stops the robot and lies it down
```

---

## 8. (Optional) Ethernet instead of Wi-Fi

Use this when the code runs on a computer mounted on the robot, or when Wi-Fi lag is a problem.

**Laptop / onboard computer.** Give it a fixed address on `192.168.168.x`. Replace `<eth-iface>` with the name shown by `ip link`:
```bash
nmcli con add type ethernet ifname <eth-iface> con-name robodog-eth \
    ipv4.method manual ipv4.addresses 192.168.168.10/24
nmcli con up robodog-eth
ping -c 3 192.168.168.168
```

**Robot** (`ssh firefly@192.168.168.168`):
1. In `/opt/export/config/sdk_config.yaml`, set `target_ip: "192.168.168.10"`.
2. In `/opt/app_launch/start_motion_control_xgw.sh` (the `_xgw` file is the wheeled version's), add this next to the other `export` lines, before the `cd /opt/export/mc/bin` line:
   ```bash
   export SDK_CLIENT_IP="192.168.168.168"
   ```
3. In `/etc/systemd/system/robot-launch.service`, add this under `[Service]`:
   ```ini
   Environment="ROBOT_NET_INTERFACES=eth0"
   ```
4. `sudo reboot`

In code, connect with `initRobot("192.168.168.10", 43988, "192.168.168.168")`.

> ⚠️ After step 2–3, if `eth0` has no IP at boot (e.g. cable unplugged), **the motion control won't start and the remote won't work** (tail light keeps blinking blue). Plug the cable in before boot, or undo the edits to go back to Wi-Fi.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `client bind success` then `connect time out` / `error async_receive_from` | The robot isn't sending to your laptop: `target_ip` doesn't match your laptop's current address, or the robot wasn't rebooted after editing. Redo §5 |
| `ROBOT SDK is Running, APP Command or other input is InValid!` | Normal. Printed by the vendor library on every start (remote/app input is ignored while the SDK is connected) |
| `battery=..% <-- LOW` / stand test says `battery ..% is below min_battery` | Charge to ≥ 50 % |
| Script hangs in `initRobot` | Wrong Boost version (needs 1.74), or the wrong library (`zsl-1` instead of `zsl-1w`) |
| `ImportError` for `mc_sdk_zsl_1w_py` | You ran plain `python3` instead of `uv run python` (wrong Python version), or `sys.path` doesn't point to `lib/zsl-1w/x86_64`. Check the SDK is cloned next to this repo |
| Robot ignores the SDK, tail light yellow | Battery < 20 %. Charge it |
| Remote doesn't work | An SDK program is connected (any of ours, or the vendor demo). Exit it (Esc, `q`, `0`, Ctrl+C) and the remote works again |
| `this laptop isn't on the robot's network` | Join the robot's Wi-Fi. (Or set a fixed `laptop_ip` in `config.machine.yaml`) |
| `SDK library folder not found` | Your SDK clone isn't at `sdk_dir` from `config.machine.yaml`. Fix the path there |
| `transition to Move failed!` / `move() returned error code 12295` | A move command reached a robot that was crouched (mode 51). Fixed: we no longer send one there. If it shows up again, note what you pressed just before |
| `... belong in config.machine.yaml` | Someone put `laptop_ip` / `sdk_dir` / `allow_motion` in the shared `config.local.yaml`. Move them to your `config.machine.yaml` |
| Tail light keeps blinking blue | Network config broken (often the Ethernet settings in §8 with no cable). Log in over Wi-Fi and undo the edits |
| Robot suddenly goes limp and lies down | It got no SDK commands for **3 s** (watchdog). Your loop froze or Wi-Fi dropped |
| Robot "does push-ups" when you press stand on the remote | `SDK_CLIENT_IP` is set to an address that doesn't exist. Fix or remove it (§8) |
| Check the motion-control port is open | On the robot: `sudo lsof -i -P -n -c mc_ctrl \| grep 43997` (should print a line) |

---

## 10. Safety rules

- **The laptop is the kill switch** while our code runs: let go of the key / SPACE / Esc (teleop), SPACE / q (stand test). The remote only works after the program has exited.
- **No battery protection under SDK control.** The robot can lose power and drop. Check `getBatteryPower()` in every script and stop below ~25 %.
- **3 s watchdog:** if the robot gets no command for 3 s, it goes limp and lies down. Send commands in a steady 20–50 Hz loop.
- **Bind `estop()` to a key in every script** (SPACE in ours): it stops and lies down, and goes limp only if lying down fails.
- Keep speeds ≤ 0.5 m/s until the code is trusted. `move()` ignores values below its minimum (vx < 0.05 m/s, yaw < 0.1 rad/s), so send exactly `0` to stop.
- Never call `jump`, `frontJump` or `backflip`, especially with anything mounted on the robot.
- `config.local.yaml` is committed and this repo is public, so never put passwords in it (e.g. the robot's Wi-Fi password).
- `allow_motion` lives in your own `config.machine.yaml` (not in git): switch it on only for a test session, off afterwards.
- Don't raise the shared speed caps until the stop tests in [TESTING.md](TESTING.md) pass.

---

## 11. Reference

- SDK repo: <https://github.com/AgibotTech/agibot_D1_Edu-Ultra>
- Wheeled API (Chinese): `agibot_D1_Edu-Ultra/docs/api_zsl-1w.md`
- Wheeled header (all available functions): `agibot_D1_Edu-Ultra/include/zsl-1w/highlevel.h`
- Deployment guide: `agibot_D1_Edu-Ultra/docs/deploy.md`, FAQ: `docs/faq.md`
