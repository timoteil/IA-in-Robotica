#!/usr/bin/env python3
"""Live 3D plot of one OptiTrack rigid body (default ID 2).

Motive streams over multicast. This joins that stream on the Mac's Wi-Fi
address and draws the body. Motive's Y axis is up; the plot uses that as
the vertical axis.

The NatNet SDK client is not used. It dies on the first frame when the
version handshake has not finished, and the Motive command port does not
answer on this network. Rigid bodies are read from the multicast data port.

Quit any other Motive client first (it holds UDP port 1511). Close the
window to stop.

    python3 plot_rigid_body.py 192.168.0.164
"""

import argparse
import socket
import threading
import time
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from mocap_extpos_lab import DATA_PORT, MULTICAST_GROUP, MotiveListener

BODY_ID = 2
TRAIL_POINTS = 400
TRAIL_HZ = 30.0
MIN_SPAN_M = 1.0
STALE_S = 0.5


def default_client_ip():
    """IP of the interface used for the default route."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]
    except OSError:
        return "192.168.8.249"
    finally:
        probe.close()


class PoseBuffer:
    def __init__(self, body_id):
        self.body_id = body_id
        self._lock = threading.Lock()
        self.latest = None
        self.last_t = 0.0
        self._last_store = 0.0
        self.trail = deque(maxlen=TRAIL_POINTS)

    def on_rigid_body(self, rb_id, pos, _rot):
        if int(rb_id) != self.body_id:
            return
        xyz = (float(pos[0]), float(pos[1]), float(pos[2]))
        now = time.monotonic()
        with self._lock:
            self.latest = xyz
            self.last_t = now
            if now - self._last_store >= 1.0 / TRAIL_HZ:
                self.trail.append(xyz)
                self._last_store = now

    def snapshot(self):
        with self._lock:
            return self.latest, list(self.trail), self.last_t


def motive_to_plot(x, y, z):
    """Motive is Y-up. Matplotlib's vertical axis is Z."""
    return x, z, y


def parse_args():
    parser = argparse.ArgumentParser(description="Live 3D plot of a NatNet rigid body.")
    parser.add_argument("server", nargs="?", help="Motive PC IP address")
    parser.add_argument("--client", default=None, help="This Mac's IP on the mocap network")
    parser.add_argument("--body-id", type=int, default=BODY_ID, help="Rigid body ID (default 2)")
    return parser.parse_args()


def main():
    args = parse_args()
    server_ip = args.server
    if not server_ip:
        server_ip = input("Motive server IP: ").strip()
    if not server_ip:
        raise SystemExit("A Motive server IP is required.")

    client_ip = args.client or default_client_ip()
    poses = PoseBuffer(args.body_id)

    print(f"Listening for rigid body {args.body_id} on {MULTICAST_GROUP}:{DATA_PORT}.")
    print(f"  Client {client_ip}    Server {server_ip}")
    motive = MotiveListener(client_ip, poses.on_rigid_body)
    motive.start()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and poses.snapshot()[0] is None:
        time.sleep(0.05)
    if poses.snapshot()[0] is None:
        motive.shutdown()
        if motive.datagram_count == 0:
            raise SystemExit(
                "No UDP packets arrived on port 1511.\n"
                "In Motive set Transmission Type to Multicast. "
                "Enable Broadcast Frame Data, Local Interface "
                f"{server_ip}, data port 1511, multicast address {MULTICAST_GROUP}."
            )
        if motive.packet_count == 0:
            raise SystemExit(
                f"Received {motive.datagram_count} UDP packets from {motive.last_addr}, "
                "but none contained a rigid body.\n"
                f"First packet id {motive.first_id}, {motive.first_len} bytes: {motive.first_hex}"
            )
        ids = ", ".join(str(body_id) for body_id in motive.seen_ids)
        raise SystemExit(
            f"Motive is streaming rigid bodies {ids}, not {args.body_id}. "
            "Re-run with --body-id and one of those IDs."
        )
    print("Receiving. Close the plot window to stop.")

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    fig.canvas.manager.set_window_title(f"Rigid body {args.body_id}")
    trail_line, = ax.plot([], [], [], color="tab:blue", linewidth=1.5)
    point, = ax.plot([0.0], [0.0], [0.0], "o", color="tab:red", markersize=8)
    ax.set_xlabel("Motive X (m)")
    ax.set_ylabel("Motive Z (m)")
    ax.set_zlabel("Motive Y, up (m)")
    ax.set_box_aspect((1, 1, 1))

    stopped = False

    def shutdown():
        nonlocal stopped
        if stopped:
            return
        stopped = True
        motive.shutdown()

    def update(_frame):
        latest, trail, last_t = poses.snapshot()
        fresh = latest is not None and (time.monotonic() - last_t) <= STALE_S
        if latest is None:
            ax.set_title(f"Waiting for rigid body {args.body_id}")
            return trail_line, point

        px, py, pz = motive_to_plot(*latest)
        if trail:
            xs, ys, zs = zip(*(motive_to_plot(*p) for p in trail))
        else:
            xs, ys, zs = [px], [py], [pz]
        trail_line.set_data(xs, ys)
        trail_line.set_3d_properties(zs)
        point.set_data([px], [py])
        point.set_3d_properties([pz])
        point.set_color("tab:red" if fresh else "0.6")

        span = MIN_SPAN_M
        if len(xs) > 1:
            span = max(span, max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        half = 0.5 * span * 1.3
        ax.set_xlim(px - half, px + half)
        ax.set_ylim(py - half, py + half)
        ax.set_zlim(pz - half, pz + half)

        state = "live" if fresh else "stale"
        ax.set_title(
            f"Rigid body {args.body_id} ({state})\n"
            f"X {latest[0]:+.3f}   Y {latest[1]:+.3f}   Z {latest[2]:+.3f} m"
        )
        return trail_line, point

    anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False, blit=False)
    fig.canvas.mpl_connect("close_event", lambda _event: shutdown())
    try:
        plt.show()
    finally:
        shutdown()
        # Keep the animation alive until the window closes.
        del anim


if __name__ == "__main__":
    main()
