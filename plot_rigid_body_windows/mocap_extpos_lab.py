#!/usr/bin/env python3
"""Motive multicast listener used by plot_rigid_body.py.

This is the listener from mocap_extpos_lab.py. The Crazyflie flight code is
not included, because the live plot does not talk to a drone.
"""

import inspect
import math
import socket
import struct
import threading

MULTICAST_GROUP = "239.255.42.99"
DATA_PORT = 1511
NAT_FRAMEOFDATA = 7
# NatNet 3+ labeled marker: id, xyz, size, param, residual.
LABELED_MARKER_BYTES = 26


def _accepts_markers(callback):
    try:
        parameters = list(inspect.signature(callback).parameters.values())
    except (TypeError, ValueError):
        return False
    if any(item.kind == inspect.Parameter.VAR_POSITIONAL for item in parameters):
        return True
    positional = [
        item
        for item in parameters
        if item.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    return len(positional) >= 4


class MotiveListener:
    """Receive Motive frames and pass each rigid body to a callback."""

    def __init__(self, client_ip, on_rigid_body):
        self.client_ip = client_ip
        self.on_rigid_body = on_rigid_body
        self._stop = threading.Event()
        self._sock = None
        self._thread = None
        self._use_sizes = None
        self._announced = False
        self.packet_count = 0
        self.datagram_count = 0
        self.last_addr = None
        self.first_id = None
        self.first_len = 0
        self.first_hex = ""
        self.seen_ids = []
        self._pass_markers = _accepts_markers(on_rigid_body)

    def start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", DATA_PORT))
            membership = struct.pack(
                "4s4s",
                socket.inet_aton(MULTICAST_GROUP),
                socket.inet_aton(self.client_ip),
            )
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
        except OSError as exc:
            sock.close()
            raise SystemExit(
                f"Could not listen on UDP {DATA_PORT} ({exc}). "
                "Close any other NatNet client first."
            ) from exc
        sock.settimeout(0.5)
        self._sock = sock
        self._thread = threading.Thread(target=self._receive, name="motive-listen", daemon=True)
        self._thread.start()

    def shutdown(self):
        self._stop.set()
        if self._sock is not None:
            self._sock.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _receive(self):
        while not self._stop.is_set():
            try:
                packet, addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            self.datagram_count += 1
            self.last_addr = addr[0]
            if self.first_id is None and len(packet) >= 2:
                self.first_id = int.from_bytes(packet[0:2], byteorder="little", signed=True)
                self.first_len = len(packet)
                self.first_hex = packet[:32].hex()
            decoded = decode_frame(packet, self._use_sizes)
            if decoded is None:
                continue
            layout, bodies, markers = decoded
            if self._use_sizes is None:
                self._use_sizes = layout
            if not self._announced:
                self._announced = True
                ids = ", ".join(str(body_id) for body_id, _pos, _rot in bodies) or "none"
                size_txt = "4.1+ size fields" if layout else "NatNet 3 / 4.0"
                print(
                    f"Motive stream from {addr[0]} ({size_txt}). "
                    f"Rigid bodies: {ids}. Labeled markers: {len(markers)}"
                )
            self.packet_count += 1
            for body_id, pos, rot in bodies:
                if body_id not in self.seen_ids:
                    self.seen_ids.append(body_id)
                if self._pass_markers:
                    self.on_rigid_body(body_id, pos, rot, markers)
                else:
                    self.on_rigid_body(body_id, pos, rot)


def decode_frame(packet, locked_layout):
    """Return (use_size_fields, bodies, markers) or None if this is not a frame."""
    if len(packet) < 8:
        return None
    message_id = int.from_bytes(packet[0:2], byteorder="little", signed=True)
    if message_id != NAT_FRAMEOFDATA:
        return None
    payload = packet[4:]
    layouts = (locked_layout,) if locked_layout is not None else (True, False)
    for use_sizes in layouts:
        try:
            bodies, markers = _walk_frame(payload, use_sizes)
        except (ValueError, struct.error, IndexError):
            continue
        if _bodies_look_real(bodies):
            return use_sizes, bodies, markers
    return None


def _walk_frame(data, use_sizes):
    offset = 0
    _frame, offset = _i32(data, offset)
    offset = _skip_marker_sets(data, offset, use_sizes)
    offset = _skip_other_markers(data, offset, use_sizes)
    bodies, offset = _read_rigid_bodies(data, offset, use_sizes)
    try:
        markers = _read_labeled_markers(data, offset, use_sizes)
    except (ValueError, struct.error, IndexError):
        markers = []
    return bodies, markers


def _skip_marker_sets(data, offset, use_sizes):
    marker_sets, offset = _i32(data, offset)
    offset = _skip_size(data, offset, use_sizes)
    if not 0 <= marker_sets <= 1000:
        raise ValueError("marker set count")
    for _ in range(marker_sets):
        end = data.index(0, offset)
        offset = end + 1
        count, offset = _i32(data, offset)
        if not 0 <= count <= 10000:
            raise ValueError("marker count")
        offset = _skip(data, offset, 12 * count)
    return offset


def _skip_other_markers(data, offset, use_sizes):
    others, offset = _i32(data, offset)
    offset = _skip_size(data, offset, use_sizes)
    if not 0 <= others <= 100000:
        raise ValueError("other marker count")
    return _skip(data, offset, 12 * others)


def _read_rigid_bodies(data, offset, use_sizes):
    count, offset = _i32(data, offset)
    offset = _skip_size(data, offset, use_sizes)
    if not 0 <= count <= 1000:
        raise ValueError("rigid body count")
    bodies = []
    for _ in range(count):
        body_id, offset = _i32(data, offset)
        pos = struct.unpack_from("<fff", data, offset)
        offset += 12
        rot = struct.unpack_from("<ffff", data, offset)
        offset += 16
        offset = _skip(data, offset, 6)
        bodies.append((body_id, pos, rot))
    return bodies, offset


def _skip_sized_block(data, offset, use_sizes):
    count, offset = _i32(data, offset)
    if use_sizes:
        size, offset = _i32(data, offset)
        if size < 0:
            raise ValueError("section size")
        return _skip(data, offset, size)
    if count != 0:
        raise ValueError("non-empty section without a size")
    return offset


def _read_labeled_markers(data, offset, use_sizes):
    offset = _skip_sized_block(data, offset, use_sizes)
    if use_sizes:
        offset = _skip_sized_block(data, offset, use_sizes)
    count, offset = _i32(data, offset)
    offset = _skip_size(data, offset, use_sizes)
    if not 0 <= count <= 10000:
        raise ValueError("labeled marker count")
    markers = []
    for _ in range(count):
        if offset + LABELED_MARKER_BYTES > len(data):
            raise ValueError("short labeled marker")
        full_id, offset = _i32(data, offset)
        pos = struct.unpack_from("<fff", data, offset)
        offset += 12
        offset = _skip(data, offset, 4)
        param, = struct.unpack_from("<h", data, offset)
        offset += 2
        offset = _skip(data, offset, 4)
        model_id = (full_id >> 16) & 0xFFFF
        marker_id = full_id & 0xFFFF
        markers.append((full_id, model_id, marker_id, pos, (param & 0x01) != 0))
    if not _markers_look_real(markers):
        raise ValueError("labeled markers")
    return markers


def _bodies_look_real(bodies):
    if not bodies:
        return False
    for body_id, pos, rot in bodies:
        if not 0 <= body_id < 100000:
            return False
        if any(not math.isfinite(value) or abs(value) > 100.0 for value in pos):
            return False
        norm = sum(value * value for value in rot)
        if not 0.5 <= norm <= 1.5:
            return False
    return True


def _markers_look_real(markers):
    for _full_id, model_id, marker_id, pos, _occluded in markers:
        if model_id < 0 or marker_id < 0:
            return False
        if any(not math.isfinite(value) or abs(value) > 100.0 for value in pos):
            return False
    return True


def _i32(data, offset):
    if offset + 4 > len(data):
        raise ValueError("short packet")
    return int.from_bytes(data[offset:offset + 4], byteorder="little", signed=True), offset + 4


def _skip_size(data, offset, use_sizes):
    if not use_sizes:
        return offset
    _size, offset = _i32(data, offset)
    return offset


def _skip(data, offset, nbytes):
    end = offset + nbytes
    if end > len(data):
        raise ValueError("short packet")
    return end
