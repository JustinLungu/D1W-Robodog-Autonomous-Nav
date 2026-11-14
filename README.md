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
| *(Optional)* Ethernet cable | Lower lag than Wi-Fi. See [§7](#7-optional-ethernet-instead-of-wi-fi) |

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
| Blinking blue | Still booting. If it never stops, the network config is broken (see [§8](#8-troubleshooting)) |
| Solid green | SDK has control |
| Solid white | Lab mode (stunts allowed) |
| **Solid yellow** | **Battery < 20 %. SDK control is refused.** Charge it |
| Blinking yellow | Battery < 5 %. Charge now |
| Solid red | Joints too hot. Let it cool down |
| Blinking red | System fault. Contact AgiBot support |

3. **Test the remote first.** Stand the robot up, drive it a little, lie it down. If the remote doesn't work, don't go further.

---

## 4. Connect the laptop to the robot (Wi-Fi)

1. Find the robot's Wi-Fi name and password on the **label on its right side or belly**. Then join it:
   ```bash
   nmcli dev wifi connect "<ROBOT_SSID>" password "<ROBOT_WIFI_PASSWORD>"
   ```
2. Find your laptop's address on the robot network. Write it down; it's `LAPTOP_IP` from here on.
   ```bash
   ip -4 -br addr | grep 192.168.234
   ```
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

The robot only sends data to **one** address, set in its config file. You have to point it at your laptop.

1. Log in. The vendor's default user and password are both `firefly`.
   ```bash
   ssh firefly@192.168.234.1
   ```

2. **Check the firmware version** (still on the robot):
   ```bash
   grep -oP 'motion-control_\K[^_]+' /etc/release/*[^rootfs]*.yaml
   ```
   | You need (wheeled) | Why |
   |---|---|
   | Motion control **≥ 0.3.1** | Minimum for the `zsl-1w` SDK |
   | Robot software **≥ 0.3.4** (recommended) | Below this, the remote is completely locked out while the SDK is in control. From 0.3.4 you can take control back |

   If it's too old, ask AgiBot support for a firmware update.

3. **Point the SDK at your laptop.** Back up the config, then replace `LAPTOP_IP` with your address from §4:
   ```bash
   sudo cp /opt/export/config/sdk_config.yaml /opt/export/config/sdk_config.yaml.bak
   sudo sed -i 's/^target_ip:.*/target_ip: "LAPTOP_IP"/' /opt/export/config/sdk_config.yaml
   cat /opt/export/config/sdk_config.yaml
   ```
   Expected:
   ```yaml
   target_ip: "192.168.234.XX"
   target_port: 43988
   ```
   - **Don't change `target_port`.**
   - **Don't delete or comment out `target_ip`.** Without it, the robot's motion control won't start.

4. **Reboot the robot.** The change doesn't take effect until you do. This only restarts the robot's computer, like turning it off and on: nothing is deleted, and your edit stays. Do it with the robot lying down. Your SSH session will drop. (Turning it off and on with the power button works too.)
   ```bash
   sudo reboot
   ```

5. Wait for **solid blue**, rejoin its Wi-Fi, and check `LAPTOP_IP` hasn't changed. If it has, redo step 3.

> ⚠️ **Firmware updates reset this config.** Redo §5 after every update.
> ⚠️ It's one laptop at a time. To switch laptops, change `target_ip` and reboot again.

---

## 6. Smoke tests (in this order)

### 6.1 Camera stream (no motion)
```bash
ffplay -rtsp_transport tcp -fflags nobuffer -flags low_delay rtsp://192.168.234.1:8554/test
```
You should see the front camera. Press `q` to close.

### 6.2 Read-only connection test (no motion)
Run from this repo. Replace `LAPTOP_IP`:
```bash
uv run python - <<'EOF'
import os, sys, time, platform
arch = platform.machine().replace('amd64', 'x86_64').replace('arm64', 'aarch64')
sys.path.insert(0, os.path.abspath(f'../agibot_D1_Edu-Ultra/lib/zsl-1w/{arch}'))
import mc_sdk_zsl_1w_py as sdk

app = sdk.HighLevel()
app.initRobot("LAPTOP_IP", 43988, "192.168.234.1")
time.sleep(1)
print("connected:", app.checkConnect())
print("battery %:", app.getBatteryPower())
print("ctrl mode:", app.getCurrentCtrlmode(), "(0=limp, 1=standing, 3=moving)")
print("rpy:", app.getRPY())
EOF
```
`connected: True` and a believable battery number mean the network setup is right.
`connect time out` means go to [§8](#8-troubleshooting).

### 6.3 First motion: stand up and lie down
Clear the area. Remote in hand.
```bash
uv run python - <<'EOF'
import os, sys, time, platform
arch = platform.machine().replace('amd64', 'x86_64').replace('arm64', 'aarch64')
sys.path.insert(0, os.path.abspath(f'../agibot_D1_Edu-Ultra/lib/zsl-1w/{arch}'))
import mc_sdk_zsl_1w_py as sdk

app = sdk.HighLevel()
app.initRobot("LAPTOP_IP", 43988, "192.168.234.1")
time.sleep(1)
assert app.getBatteryPower() > 25, "charge the robot first"
print("stand:", app.standUp());  time.sleep(4)
print("lie:",   app.lieDown());  time.sleep(4)
EOF
```

### 6.4 Vendor demo (optional)
The vendor's interactive demo is at `../agibot_D1_Edu-Ultra/demo/zsl-1w/python/examples/highlevel_demo.py`.
Change its `initRobot("127.0.0.1", 43988, "127.0.0.1")` line to `initRobot("LAPTOP_IP", 43988, "192.168.234.1")` first.
**Don't use `highlevel_demo_ultra-w.py`.** It loads the legged robot's library and contains `jump()` and `backflip()`.

---

## 6.5 Safety layer (`robodog` package)

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
uv run robodog-dryrun                  # scripted session on a fake robot; press SPACE to try the e-stop
uv run robodog-dryrun --low-battery    # see the battery protection kick in
uv run robodog-dryrun --disconnect     # see the lost-connection protection kick in
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

## 7. (Optional) Ethernet instead of Wi-Fi

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

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `connect time out` | `target_ip` on the robot doesn't match your laptop's current address, or the robot wasn't rebooted after editing. Redo §5 |
| Script hangs in `initRobot` | Wrong Boost version (needs 1.74), or the wrong library (`zsl-1` instead of `zsl-1w`) |
| `ImportError` for `mc_sdk_zsl_1w_py` | You ran plain `python3` instead of `uv run python` (wrong Python version), or `sys.path` doesn't point to `lib/zsl-1w/x86_64`. Check the SDK is cloned next to this repo |
| Robot ignores the SDK, tail light yellow | Battery < 20 %. Charge it |
| Remote doesn't work | An SDK program is running and has control. Stop it, or hold **LT+RT+A** (Logitech) / **L2+R2+A** (remote with screen) to take control back (firmware ≥ 0.3.4) |
| Tail light keeps blinking blue | Network config broken (often the Ethernet settings in §7 with no cable). Log in over Wi-Fi and undo the edits |
| Robot suddenly goes limp and lies down | It got no SDK commands for **3 s** (watchdog). Your loop froze or Wi-Fi dropped |
| Robot "does push-ups" when you press stand on the remote | `SDK_CLIENT_IP` is set to an address that doesn't exist. Fix or remove it (§7) |
| Check the motion-control port is open | On the robot: `sudo lsof -i -P -n -c mc_ctrl \| grep 43997` (should print a line) |

---

## 9. Safety rules

- **Someone always holds the remote** and knows how to take control back (LT+RT+A / L2+R2+A).
- **No battery protection under SDK control.** The robot can lose power and drop. Check `getBatteryPower()` in every script and stop below ~25 %.
- **3 s watchdog:** if the robot gets no command for 3 s, it goes limp and lies down. Send commands in a steady 20–50 Hz loop.
- **Call `passive()` as an emergency stop.** Bind it to a key in every script.
- Keep speeds ≤ 0.5 m/s until the code is trusted. `move()` ignores values below its minimum (vx < 0.05 m/s, yaw < 0.1 rad/s), so send exactly `0` to stop.
- Never call `jump`, `frontJump` or `backflip`, especially with anything mounted on the robot.
- `config.local.yaml` is committed and this repo is public, so never put passwords in it (e.g. the robot's Wi-Fi password).
- Don't push `allow_motion: true` by accident. Set it only on the machine that's actually driving the robot.

---

## 10. Reference

- SDK repo: <https://github.com/AgibotTech/agibot_D1_Edu-Ultra>
- Wheeled API (Chinese): `agibot_D1_Edu-Ultra/docs/api_zsl-1w.md`
- Wheeled header (all available functions): `agibot_D1_Edu-Ultra/include/zsl-1w/highlevel.h`
- Deployment guide: `agibot_D1_Edu-Ultra/docs/deploy.md`, FAQ: `docs/faq.md`
