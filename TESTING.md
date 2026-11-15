# Robot reliability tests

Step-by-step checks that the robot stops when it should, **before** we trust it to drive by itself.
Run them **in order**. Run each test **3 times** (once is luck, three times is a pattern).
**Stop at the first failure** and look at it before going further.

Why these tests: while one of our programs is connected, the joystick does nothing (README §6). So the
ways to stop the robot are the laptop (SPACE / Esc / q / Ctrl+C, or letting go of the key in teleop) and,
if the laptop or Wi-Fi fails, **the robot's own watchdog** (vendor: no SDK data for 3 s → damping mode,
lies down). That watchdog has not been tested yet.

Tools used: `robodog-stand-test` (stands still, never drives) for stage 1, `robodog-teleop` (keyboard
driving, moves only while a key is held) for stage 2.

---

## Before every session

**People:** one person at the laptop (hands on the keyboard), one person watching the robot with the
joystick switched on (it only works after the program has exited).

**Space:** 2 m clear around the robot, nothing fragile, nobody standing in front of it.

**Two terminals, both in the repo:**
```bash
cd ~/Hackathon/D1W-Robodog-Autonomous-Nav
```
Terminal 1 runs the test. Terminal 2 is for "breaking" things (killing the program, switching Wi-Fi off).

**Laptop on the robot's Wi-Fi, then check connection and battery:**
```bash
uv run --extra vision-gpu robodog-status
```
**See:** `connecting: laptop 192.168.234.x:43988 -> robot 192.168.234.1`, `connect success!`, `battery=..%`.
- `this laptop isn't on the robot's network` → join the robot's Wi-Fi.
- `connect time out` → see below.
- Battery below 50 % → charge first.

Your laptop's address is detected automatically (`laptop_ip: "auto"` in `config.machine.yaml`,
the per-laptop file that's created on first run and ignored by git).

**If the status check says `connect time out`:** the robot only sends its data to **one** laptop, the
address in its `target_ip`. Most likely it's still set to a teammate's laptop.

1. See which laptop the robot is sending to (password `firefly`), and this laptop's address:
   ```bash
   ssh firefly@192.168.234.1 "grep target_ip /opt/export/config/sdk_config.yaml"; ip -4 -br addr | grep -oP '192\.168\.234\.\d+' | head -1
   ```
2. **They differ** → point the robot at this laptop, then reboot it:
   ```bash
   MYIP=$(ip -4 -br addr | grep -oP '192\.168\.234\.\d+' | head -1); ssh -t firefly@192.168.234.1 "sudo sed -i 's/^target_ip:.*/target_ip: \"$MYIP\"/' /opt/export/config/sdk_config.yaml && cat /opt/export/config/sdk_config.yaml"
   ```
   ```bash
   ssh -t firefly@192.168.234.1 "sudo reboot"
   ```
   Wait for the **solid blue** tail light (about 1–2 min), rejoin the robot's Wi-Fi, run the status check again.
3. **They're the same** → something on this laptop may still hold the connection (an old demo or test that didn't exit):
   ```bash
   ss -ulpn | grep 43988
   ```
   - Prints a line with a program name → stop it, then retry the status check:
     ```bash
     pkill -f highlevel_demo; pkill -f robodog
     ```
   - Prints nothing → the robot may have just booted: wait for solid blue and retry. Still failing → reboot the robot once.

**Turn motion on for this laptop** (only for the session; it's off again at the end):
```bash
sed -i 's/^allow_motion: false/allow_motion: true/' config.machine.yaml && grep allow_motion config.machine.yaml
```
**See:** `allow_motion: true`

**If a terminal stops showing what you type** after a test (a killed program can leave it in a raw mode):
```bash
reset
```

---

## Stage 1: Robot standing still

### Test 1: Joystick and app e-stop while the SDK is connected (vendor demo)
**Goal:** find out whether the Retroid can stop or take over the robot while an SDK program is connected.

1. Check the AgiBot app version on the Retroid (the docs need **≥ v2.4.5** for the control switch). Write it down.
2. Terminal 1:
   ```bash
   cd ~/Hackathon/agibot_D1_Edu-Ultra/demo/zsl-1w/python/examples && python3 highlevel_demo.py
   ```
3. **See:** `Robot is standing up...`, the robot stands. Note the **tail light colour**.
4. On the Retroid, hold **L2 + R2 + A** for 3 s. Try to drive.
5. Hold **L2 + R2 + B** for 3 s (on the Retroid, B is the bottom button: AgiBot's "A" may mean that position). Try to drive.
6. In the app, press the **red emergency-stop button** (the manual: "press the red emergency stop button to trigger the emergency stop operation").
7. Terminal 1 (if the robot is still standing): `0` ↵.
8. **See:** robot lies down, goes limp, demo exits. Joystick works again.
9. `cd ~/Hackathon/D1W-Robodog-Autonomous-Nav` to get back to the repo.

**Pass:** any of steps 4–6 stops or takes over the robot. **A fail is OK**: just record which did what. If none work, the laptop stays the only stop during SDK control.

### Test 2: Stand up and quit (q)
1. Terminal 1:
   ```bash
   uv run --extra vision-gpu robodog-stand-test --allow-motion
   ```
2. **See:** `battery=..%`, then `The robot will STAND UP... ENTER = go`. Press **ENTER**.
3. **See:** the robot stands; status lines every second with `mode=..` and `sent=move(...)`. For the first 3 s `sent=` stays `None`: SafeDog doesn't send anything while the robot is still standing up. Write down the `mode` number once it's standing.
4. Press **q**.
5. **See:** `lying down...`, the robot lies down, the program ends.
6. Stand up and lie down once with the joystick.

**Pass:** robot lies down gently; program exits; joystick works right after.

### Test 3: E-stop while standing (SPACE)
1. Terminal 1: same command as test 2, **ENTER**, wait until it's standing.
2. Press **SPACE**.
3. **See:** `!!! E-STOP (SPACE pressed): stopping and lying down`. The robot **crouches**, then about **3 s later settles fully down** (no collapse). If you instead see `lying down failed ... calling passive() instead`, it went limp as the fallback: note it.
4. **See:** `stopped by SafeDog: E-STOP (SPACE pressed)`, the program ends.
5. Joystick: stand up and lie down.

**Pass:** starts lying down within ~0.5 s; nothing else happens; joystick works after the program exits.

### Test 4: Program crash while standing (robot's watchdog)
**Goal:** prove the robot stops by itself if our program dies without saying goodbye.

1. Terminal 1: same command as test 2, **ENTER**, wait until it's standing.
2. Terminal 2, type this but **don't press ENTER yet**:
   ```bash
   pkill -9 -f robodog-stand-test
   ```
3. Have a stopwatch ready (phone). Press **ENTER** in terminal 2 and start the stopwatch at the same moment.
4. **See in terminal 1:** the program just disappears (`Killed`), with no `lying down...` message. That's the point: our code had no chance to stop the robot.
5. **Watch the robot:** stop the stopwatch when it starts to lie down / go limp.
6. Run `reset` in terminal 1 if it stopped echoing.
7. Joystick: does it work now? (If not, wait 10 s and try again.)

**Pass:** the robot lies down **by itself within ~3–5 s**, every time. **Fail:** still standing after 10 s → stop, lie it down with the joystick (or the vendor demo's `0`), and don't do any driving tests.

### Test 5: Wi-Fi loss while standing (robot's watchdog)
**Goal:** the real "lost connection" case.

1. Terminal 1: same command as test 2, **ENTER**, wait until it's standing.
2. Terminal 2, type but don't press ENTER yet:
   ```bash
   nmcli radio wifi off
   ```
3. Press **ENTER** in terminal 2 and start the stopwatch.
4. **See in terminal 1:** after a moment `!!! halting: connection lost` (our code notices, but can't reach the robot anymore).
5. **Watch the robot:** stop the stopwatch when it lies down / goes limp.
6. Turn Wi-Fi back on and rejoin the robot:
   ```bash
   nmcli radio wifi on
   ```
   If it doesn't rejoin by itself within ~20 s: `nmcli dev wifi connect "<ROBOT_SSID>" password "<ROBOT_WIFI_PASSWORD>"`.
7. Run the status check from "Before every session" (the address may have changed).

**Pass:** same as test 4, the robot lies down by itself within ~3–5 s.

---

## Stage 2: Moving slowly (teleop)

Use **speed level 1** (the default: a third of the caps, ~0.1 m/s forward, ~0.17 rad/s turning).
Mark the robot's start position on the floor with tape (e.g. at the front wheels), and have a tape measure ready.

**Teleop controls** (the small window must have focus: click it):

| Hold | | Click | |
|---|---|---|---|
| **W** / ↑ | forward | **R** | stand up |
| **S** / ↓ | backward | **F** | lie down |
| **A** / ← | turn left | **1 2 3** | speed level |
| **D** / → | turn right | **SPACE** | **E-stop** (stop + lie down, latched) |
| | | **H** | shake hands (then stands again) |
| **Q** / **E** | sideways | **Esc** | lie down and quit |

### Test 6: Controls without the robot
```bash
uv run --extra vision-gpu robodog-teleop --fake
```
Press R, hold W, release, press SPACE, Esc. **See** the window react (speeds shown, standing/e-stop state). Nothing on the robot moves (`--fake`).

**Pass:** you know where every key is before touching the real robot.

### Test 7: Stand, wait, lie down
1. Terminal 1:
   ```bash
   uv run --extra vision-gpu robodog-teleop --allow-motion
   ```
2. Click the window. Press **R**.
3. **See:** `standing up... (keys ignored for 3 s)`, the robot stands. Hold **W** during these 3 s: **nothing happens** (on purpose).
4. **See:** after ~3 s the message changes to `standing. hold W/A/S/D (or arrows) to drive`.
5. Press **H**. **See:** `shaking hands... (keys ignored ~7 s)`, the robot shakes hands, then stands up again by itself; after ~7 s the message is back to `standing. hold W/A/S/D`. Hold **W** during the handshake: nothing moves.
6. Press **F**. **See:** the robot crouches, then ~3 s later settles fully down (`mode 51` → `mode 0` in the window). No `transition to Move failed` in the terminal.
7. Press **Esc**. The program ends.

**Pass:** no movement during the stand-up or the handshake; handshake ends standing; lies down on F; exits on Esc.

### Test 8: Hold to drive, release to stop
1. Teleop as in test 7, **R**, wait for `standing. hold W/A/S/D`.
2. Hold **W** for **2 s**, then let go.
3. **See:** the robot creeps forward (~20 cm) and **stops the moment you let go**.
4. **Measure** how far it rolled after you let go (second person watching).
5. Repeat with **S** (backward), **A** and **D** (turning).
6. **F**, then **Esc**.

**Pass:** each key moves the right way; stops within a few cm of letting go.

### Test 9: Clicking away stops it
1. Teleop, **R**, wait. Hold **W**.
2. While holding W, **click on another window** (e.g. the terminal) with the mouse.
3. **See:** the robot stops immediately, even though W is still pressed.
4. Click the teleop window again, **F**, **Esc**.

**Pass:** stops on losing focus.

### Test 10: E-stop while driving (SPACE)
1. Teleop, **R**, wait. Hold **W**.
2. After ~1 s, press **SPACE** (keep holding W or not, doesn't matter).
3. **See:** the robot stops, then lies down; the window shows `E-STOP: stopped and lying down`; nothing moves anymore even if you press W.
4. **Measure** how far it rolled after SPACE.
5. Close the window / Esc.

**Pass:** stops within ~10 cm of where it was when SPACE was pressed, then lies down (not a collapse).

### Test 11: Quit while driving (Esc)
Same as test 10 with **Esc** instead of SPACE.
**See:** the robot stops and lies down, the program ends.

**Pass:** stops and lies down.

### Test 12: Program crash while driving (worst case)
**Goal:** how far does the robot go on its own if our program dies mid-move? This number limits how fast we may ever drive.

The teleop stops the robot as soon as its window loses focus, so you can't click into terminal 2 to kill it. Use a **timed** kill instead:
1. Teleop, **R**, wait for `standing. hold W/A/S/D`.
2. Terminal 2: type this and press **ENTER** (the kill happens 8 s later):
   ```bash
   sleep 8 && pkill -9 -f robodog-teleop
   ```
3. **Immediately** click back into the teleop window and **hold W**. Keep holding it.
4. **See:** after the 8 s the teleop window disappears.
5. **Watch and measure:** how far it rolls after the window disappeared, and how long until it stops / lies down.
6. `reset` in terminal 1 if needed. Joystick: does it work?

**Pass:** it stops by itself; write down **roll distance and time**. At level 1 (~0.1 m/s) and a ~3 s watchdog, expect up to ~30 cm.

### Test 13: Wi-Fi loss while driving (worst case)
Same as test 12, but terminal 2 runs:
```bash
sleep 8 && nmcli radio wifi off
```
Afterwards: `nmcli radio wifi on`, rejoin the robot's Wi-Fi, run the status check.

**Pass:** stops by itself; write down roll distance and time.

---

## Stage 3: Conditions

### Test 14: Wi-Fi range
Robot lying down (nothing moves). From each spot, run:
```bash
ping -c 20 192.168.234.1
```
Spots: 2 m, 5 m, 10 m from the robot, and once with a wall in between.
**See:** the last lines, `0% packet loss` and `rtt min/avg/max`.

**Pass:** 0 % loss and avg under ~50 ms wherever you plan to drive. Any loss → don't drive from there.

### Test 15: Repeatability
From the tape mark, teleop level 1: hold **W** for exactly **3 s** (count or use a stopwatch), let go, measure. Do it **3 times**.
**Pass:** the distances are similar between runs (within a few cm).

### Test 16: Low-battery refusal (whenever the battery is naturally around 20–25 %)
```bash
uv run --extra vision-gpu robodog-stand-test --allow-motion
```
**See:** after ENTER, `refused: battery ..% is below min_battery 25%; charge first`. The robot does **not** stand.

---

## End of every session
```bash
sed -i 's/^allow_motion: true/allow_motion: false/' config.machine.yaml && grep allow_motion config.machine.yaml
```
**See:** `allow_motion: false`. Then put the robot on the charger.

---

## Gate before anything autonomous (click-to-go, following, litter pickup)
- Tests 1–15 pass, 3 times each.
- The worst-case roll distance from tests 12–13 is known and small.
- Hold-to-run exists ✅ (teleop). Autonomous code needs its own equivalent (e.g. hold a key to let it drive).

---

## Results log
Copy a row per run. Times in seconds, distances in metres.

| Date | Test | Run | Result (pass/fail) | Numbers (time / distance / mode / light) | Notes |
|---|---|---|---|---|---|
| | 1 joystick / app e-stop | 1 | | app version: ; light: ; L2+R2+A: ; L2+R2+B: ; red button: | |
| | 2 stand + q | 1 | | mode standing: | |
| | 3 SPACE standing | 1 | | | |
| | 4 crash standing | 1 | | time to lie down: | joystick after? |
| | 5 Wi-Fi standing | 1 | | time to lie down: | |
| | 6 teleop --fake | — | | | |
| | 7 stand / wait / lie | 1 | | | |
| | 8 hold / release | 1 | | roll after release: | |
| | 9 focus loss | 1 | | | |
| | 10 SPACE moving | 1 | | roll after SPACE: | |
| | 11 Esc moving | 1 | | | |
| | 12 crash moving | 1 | | roll: ; time: | |
| | 13 Wi-Fi moving | 1 | | roll: ; time: | |
| | 14 range | — | | 2 m: ; 5 m: ; 10 m: ; wall: | |
| | 15 repeat (3 s hold) | 1–3 | | | |
| | 16 low battery | 1 | | | |
