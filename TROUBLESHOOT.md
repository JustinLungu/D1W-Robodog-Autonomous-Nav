# Troubleshooting

Run everything from the repo folder on your laptop, with the laptop on the robot's Wi-Fi:
```bash
cd ~/Hackathon/D1W-Robodog-Autonomous-Nav && source .venv/bin/activate
```
SSH to the robot: user and password are both `firefly` (vendor default). `sudo` may ask for it again.

---

## 1. Switching laptops (do this every time someone else used the robot)

The robot sends its data to **one** laptop only: the address in `target_ip` on the robot. Whoever
used it last has it pointed at their laptop, so on yours everything fails with `connect time out`.

**0. The other laptop closes every robot program** (teleop, go-to, status, viewers).

**1. Check where the robot sends to, and your laptop's address:**
```bash
ssh firefly@192.168.234.1 "grep target_ip /opt/export/config/sdk_config.yaml"; ip -4 -br addr | grep -oP '192\.168\.234\.\d+' | head -1
```
- **The same address twice** → it's already yours: skip to step 4.
- **Different** → steps 2 and 3.

**2. Point the robot at your laptop:**
```bash
MYIP=$(ip -4 -br addr | grep -oP '192\.168\.234\.\d+' | head -1); ssh -t firefly@192.168.234.1 "sudo sed -i 's/^target_ip:.*/target_ip: \"$MYIP\"/' /opt/export/config/sdk_config.yaml && grep target_ip /opt/export/config/sdk_config.yaml"
```
**Expect:** `target_ip: "192.168.234.<your number>"`.

**3. Reboot the robot** (it must be lying down):
```bash
ssh -t firefly@192.168.234.1 "sudo reboot"
```
`client_loop: send disconnect: Broken pipe` is normal (the robot dropped the connection to reboot).
Wait for the **solid blue** light (~1–2 min). Rejoin the robot's Wi-Fi if your laptop dropped off.

**4. Check your address didn't change, then the connection:**
```bash
ip -4 -br addr | grep -oP '192\.168\.234\.\d+'; uv run --no-sync --extra vision-gpu robodog-status
```
**Expect:** the same address as in step 2, and `connect success!` with a battery reading (≥ 25 %).
If the address changed, do steps 2–3 again (or see section 2).

**5. Motion on for this laptop** (off again at the end, see section 9):
```bash
sed -i 's/^allow_motion: false/allow_motion: true/' config.machine.yaml
```

---

## 2. `connect time out` although it was working before

- **Someone else's laptop has it** → section 1.
- **Your laptop got a new address** on the robot's Wi-Fi (it's handed out when you connect, and
  can change after reconnecting) → section 1, steps 1–4.
- **To stop the address from changing:** give your laptop a fixed one on the robot's network.
  Find the Wi-Fi connection's name:
  ```bash
  nmcli -t -f NAME,DEVICE connection show --active
  ```
  Then (replace the name; pick a number nobody else uses, e.g. `.10` for you, `.7` for a teammate):
  ```bash
  nmcli connection modify "<robot-wifi-name>" ipv4.method manual ipv4.addresses 192.168.234.10/24 ipv4.gateway 192.168.234.1 && nmcli connection up "<robot-wifi-name>"
  ```
- **Two of our programs at once** (e.g. teleop and go-to): only one can be connected. Close one.

## 3. No internet on the robot's Wi-Fi

The robot's Wi-Fi has no internet. So:
- Use `uv run --no-sync ...` (otherwise `uv` tries to download and times out with
  `Failed to fetch ... hatchling`).
- Before loading models: `export HF_HUB_OFFLINE=1` (otherwise "loading the trash model..." hangs
  while it tries to check online for a newer version).
- Do `git pull`, `uv sync --extra vision-gpu` and first-time model downloads (person model:
  section 8) **on normal Wi-Fi**.

## 4. Picture problems

- **Lots of red `[h264 ...] error while decoding` lines:** a few are harmless (lost Wi-Fi packets).
  A constant stream = two programs reading the camera at once (close one), or a weak signal (move
  the laptop closer to the robot).
- **Picture lags ~0.5 s:** make sure you're on a version with `DECODE_THREADS = 1` in
  `src/robodog/vision/camera.py`.
- **Viewer window only shows half** (e.g. only the depth map): it's wider than the screen; drag it,
  or start it with `--width 900`.

## 5. The trash tracker's CSRT stopped working / `TrackerCSRT` errors

Both OpenCV packages got installed (happens after `uv sync` on an old branch). Fix (works offline):
```bash
uv sync --offline --extra vision-gpu --reinstall-package opencv-contrib-python
```
Check: `uv run --no-sync python -c "import cv2; print(cv2.TrackerCSRT_create)"` prints a function.

## 6. The robot lies down by itself / won't stand

- **Battery below 25 %:** SafeDog refuses to stand and lies the robot down. Charge.
- **No commands for 3 s** (a program froze, Wi-Fi dropped): the robot's own watchdog lays it down.
  That's a safety feature. Restart the program.
- **E-stop (SPACE) was pressed:** it's latched. Restart the program to move again.
- `warning: move() returned error code 12295 (... 3.x s after standUp)`: the robot wasn't done
  standing up yet. Harmless (that one command is ignored).

## 7. Robot config files we changed (and how to undo)

- `target_ip` in `/opt/export/config/sdk_config.yaml`: section 1 (the robot's original backup is
  `sdk_config.yaml.bak` in the same folder).
- `body_height_ratio` test (didn't help, should be back at `1.0`). Check:
  ```bash
  ssh firefly@192.168.234.1 "grep -n body_height_ratio /opt/export/config/xg_wheel-user-parameters.yaml"
  ```
  If it isn't `1.0`, restore the original and reboot:
  ```bash
  ssh -t firefly@192.168.234.1 "sudo cp /opt/export/config/xg_wheel-user-parameters.yaml.orig /opt/export/config/xg_wheel-user-parameters.yaml && sudo reboot"
  ```

## 8. Person check (`--people`)

- **First time:** download the model on normal Wi-Fi (~5 MB, into `models/`):
  ```bash
  uv run --no-sync --extra vision-gpu python -c "from robodog.vision.people import PersonDetector; PersonDetector('models/yolo11n.pt', 0.4, 640)"
  ```
- `--people needs --obstacles`: add `--obstacles` (it stops for a person in the red area).
- **Robot won't move and says `STOPPED: person check: waiting for the first picture`:** normal for
  the first seconds while the models load. If it stays: the camera isn't delivering (section 4).

## 9. End of every session

```bash
sed -i 's/^allow_motion: true/allow_motion: false/' config.machine.yaml
```
Lie the robot down (F or Esc) before closing anything. If a teammate takes over next, they start
at section 1.
