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
| Handheld remote | Your safety net. Someone holds it during every test |
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
| **≥ 0.3.4** | Remote takeover (LT+RT+A / L2+R2+A) while the SDK has control. Below this, the remote is locked out |

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

The physical remote (the joystick/controller that came with the robot) is the safety backup that doesn't depend on our code. **Keep it switched on and in your hand during every test.** It talks to the robot directly, not through your laptop.

| Controller | Take control back from the SDK: hold together |
|---|---|
| Logitech gamepad | **LT + RT + A** |
| Remote with screen | **L2 + R2 + A** |

Tail light: solid **green** = SDK (our code) has control, solid **blue** = remote has control.
While any SDK program is connected, the remote is ignored until you take over.

**Battery:** the robot refuses SDK control below **20 %** (yellow tail light), and our code refuses to stand below **25 %**. **Charge to ≥ 50 % before any motion test.** The vendor demo has no battery check at all.

### 6.1 Read-only status (nothing moves)
Puts your laptop's address into [config.local.yaml](config.local.yaml) and reads the robot's state:
```bash
cd ~/robodog/D1W-Robodog-Autonomous-Nav && MYIP=$(ip -4 -br addr | grep -oP '192\.168\.234\.\d+' | head -1) && echo "laptop IP: $MYIP" && sed -i "s/laptop_ip: \"[^\"]*\"/laptop_ip: \"$MYIP\"/" config.local.yaml && uv run robodog-status
```
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

### 6.4 Our stand-up test + remote takeover
The first motion through **our** code ([stand_test.py](src/robodog/cli/stand_test.py)). It stands up through `SafeDog`, holds still for 60 s while printing the status every second, then lies down. It never drives: SafeDog only sends `move(0, 0, 0)` to keep it standing.

Turn motion on in the config (and back off afterwards):
```bash
sed -i 's/allow_motion: false/allow_motion: true/' config.local.yaml && grep allow_motion config.local.yaml
```
Run it (robot lying down, 2 m clear, remote in hand):
```bash
uv run robodog-stand-test --allow-motion
```
It connects, prints the battery, and **asks for ENTER before standing** (Ctrl+C there aborts). While it runs:

| Key | Effect |
|---|---|
| **SPACE** | E-stop: robot goes limp immediately, script stops sending anything |
| **q** | Lie down and quit |
| Ctrl+C | Lie down and quit |

**The takeover test:** while it's standing, hold **LT+RT+A** / **L2+R2+A**. The light should turn blue and the remote should work. Watch whether the robot still reacts to the script's `move(0,0,0)` commands. If it does, `SafeDog` must learn to detect the takeover before any further motion work. Lie the robot down **with the remote**, then press `q`.

Turn motion back off:
```bash
sed -i 's/allow_motion: true/allow_motion: false/' config.local.yaml && grep allow_motion config.local.yaml
```

### 6.5 Results on our robot
| Check | Result |
|---|---|
| Firmware | `0.5.1-xgw` (takeover supported) |
| `mode` while lying | `0` |
| SDK connection after §5 | ✅ `connect success!` |
| `mode` while standing / moving | _todo (6.1 watch mode, 6.4)_ |
| Remote while `robodog-status` is connected | _todo (6.1)_ |
| Tail light while the SDK has control | _todo (6.3 / 6.4)_ |
| Stays standing while the vendor demo sends nothing? | _todo (6.3)_ |
| Takeover during our stand test, and does the robot ignore our `move(0,0,0)` afterwards? | _todo (6.4)_ |

---

## 7. Safety layer (`robodog` package)

All our code talks to the robot through **`SafeDog`** ([src/robodog/safety.py](src/robodog/safety.py)), never through the raw SDK.

| Protection | What it does |
|---|---|
| **Motion off by default** | Nothing moves unless `safety.allow_motion: true` is set in `config.local.yaml` **and** the code passes `allow_motion=True` |
| **Speed caps** | Clamped to `max_vx` / `max_vy` / `max_yaw` (default 0.3 m/s, 0.2 m/s, 0.5 rad/s). Values below the SDK minimums become exactly 0 |
| **Dead-man switch** | If your code stops calling `set_velocity()` for `cmd_timeout` (0.3 s), the speed drops to 0 |
| **Keepalive loop** | Sends `move()` at `rate_hz` (30 Hz) while standing, so the robot's 3 s watchdog doesn't trip by accident |
| **Battery** | Refuses to stand below `min_battery` (25 %). Stops and lies down if the battery drops below that mid-run |
| **Lost connection / SDK error** | Stops, tries to lie down, and latches a fault. Nothing else is sent until you restart |
| **E-stop** | `estop()` calls `passive()` (the robot goes limp) and latches. Scripts bind it to **SPACE** |
| **No stunts** | `jump` / `frontJump` / `backflip` / `shakeHand` aren't exposed at all |

All settings live in one committed file, **[config.local.yaml](config.local.yaml)**. Edit it in place; at minimum set `network.laptop_ip`.
[src/robodog/config.py](src/robodog/config.py) only defines which keys must exist and their types. A missing key, a typo (`max_vX`) or a wrong type (`allow_motion: "false"` in quotes) stops the program with a clear error instead of being silently ignored. If you add a new setting, add it to `config.local.yaml` and to the class in `config.py`.

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
> While any SDK program is connected, the remote may not respond. That's expected. Stop the script to get the remote back.

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
| Remote doesn't work | An SDK program is running and has control. Stop it, or hold **LT+RT+A** (Logitech) / **L2+R2+A** (remote with screen) to take control back (firmware ≥ 0.3.4) |
| Tail light keeps blinking blue | Network config broken (often the Ethernet settings in §8 with no cable). Log in over Wi-Fi and undo the edits |
| Robot suddenly goes limp and lies down | It got no SDK commands for **3 s** (watchdog). Your loop froze or Wi-Fi dropped |
| Robot "does push-ups" when you press stand on the remote | `SDK_CLIENT_IP` is set to an address that doesn't exist. Fix or remove it (§8) |
| Check the motion-control port is open | On the robot: `sudo lsof -i -P -n -c mc_ctrl \| grep 43997` (should print a line) |

---

## 10. Safety rules

- **Someone always holds the remote** and knows how to take control back (LT+RT+A / L2+R2+A).
- **No battery protection under SDK control.** The robot can lose power and drop. Check `getBatteryPower()` in every script and stop below ~25 %.
- **3 s watchdog:** if the robot gets no command for 3 s, it goes limp and lies down. Send commands in a steady 20–50 Hz loop.
- **Call `passive()` as an emergency stop.** Bind it to a key in every script.
- Keep speeds ≤ 0.5 m/s until the code is trusted. `move()` ignores values below its minimum (vx < 0.05 m/s, yaw < 0.1 rad/s), so send exactly `0` to stop.
- Never call `jump`, `frontJump` or `backflip`, especially with anything mounted on the robot.
- `config.local.yaml` is committed and this repo is public, so never put passwords in it (e.g. the robot's Wi-Fi password).
- Don't push `allow_motion: true` by accident. Set it only on the machine that's actually driving the robot.

---

## 11. Reference

- SDK repo: <https://github.com/AgibotTech/agibot_D1_Edu-Ultra>
- Wheeled API (Chinese): `agibot_D1_Edu-Ultra/docs/api_zsl-1w.md`
- Wheeled header (all available functions): `agibot_D1_Edu-Ultra/include/zsl-1w/highlevel.h`
- Deployment guide: `agibot_D1_Edu-Ultra/docs/deploy.md`, FAQ: `docs/faq.md`
