"""Send the vendor phone app's commands to the robot over UDP (what the remote / app does).

Why: the SDK has no command for the robot's real low crawl. SDK crawl() starts a tall
stepping gait (RL_GaitWalk in the motion controller's log), while the app's crawl button
starts RL_crawl: the body drops from ~0.385 m to ~0.25 m and stays low while driving.
The app sends small JSON messages to the robot's `dog_task` program, UDP port 8081.
Found by reading the robot's logs (/tmp/log/dog_task.log, motion_control.stdout) while
using the remote, then confirmed by sending the crawl command from a laptop (2026-11-15).
Undocumented by the vendor: see README section "Crawling".

    {"type": "heartbeat"}          the app sends this ~10x a second; we send a few first
    {"cmd": 106, "type": "cmd"}    crawl (low body)   -> RL_crawl, control_mode "stand_down"
    {"cmd": 122, "type": "cmd"}    stand up (tall)    -> "stand_up", then "rl"
    {"cmd": 90,  "type": "cmd"}    dump (lie down)    -> "emergency"   (not used by our code)

Nothing comes back; check the effect with the body height (getPosition z).
"""

from __future__ import annotations

import json
import socket
import time

APP_PORT = 8081
CMD_CRAWL = 106
CMD_STAND = 122
CMD_DUMP = 90


class AppLink:
    def __init__(self, robot_ip: str, laptop_ip: str = "", port: int = APP_PORT):
        self.dst = (robot_ip, port)
        self.laptop_ip = laptop_ip

    def command(self, code: int, repeat: int = 3):
        """Send one app command (a few heartbeats first, then the command `repeat` times, like the app).

        UDP: no answer and no delivery guarantee; raises OSError only if the laptop can't send at all.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            if self.laptop_ip:
                s.bind((self.laptop_ip, 0))   # leave on the robot's Wi-Fi, not via another interface
            for _ in range(3):
                s.sendto(json.dumps({"type": "heartbeat"}).encode(), self.dst)
                time.sleep(0.02)
            for _ in range(repeat):
                s.sendto(json.dumps({"cmd": int(code), "type": "cmd"}).encode(), self.dst)
                time.sleep(0.005)
