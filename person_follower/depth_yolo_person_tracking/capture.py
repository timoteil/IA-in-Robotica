#!/usr/bin/env python3
"""Save Crazyflie AI-deck WiFi frames as timestamped image files.

Connect the laptop to the AI-deck access point, then:

    python3 capture.py
    python3 capture.py --address 0xE7E7E7E7E7
    python3 capture.py --save-dir /path/to/session

JPEG frames are written as received. Raw Bayer frames are written as PNG.
Each file is named with the capture time in Unix milliseconds, for example
1712345678901.jpg.
"""

from __future__ import annotations

import argparse
import socket
import struct
import time
import zlib
from pathlib import Path

AIDECK_MAGIC = 0xBC
DEFAULT_IP = "192.168.4.1"
DEFAULT_PORT = 5000
DEFAULT_ADDRESS = "0xE7E7E7E7E7"


def radio_address(text: str) -> str:
    """Return the 5-byte Crazyflie address as 10 hex characters, no 0x prefix."""
    raw = text.strip()
    if raw.lower().startswith("0x"):
        raw = raw[2:]
    raw = raw.upper()
    if len(raw) != 10 or any(char not in "0123456789ABCDEF" for char in raw):
        raise SystemExit(f"Bad Crazyflie address {text!r}. Expected 0xE7E7E7E7E7.")
    return raw


def crazyflie_uri(address: str) -> str:
    """Radio URI for this drone. Channel 80 and 2M match the lab Crazyflies."""
    return f"radio://0/80/2M/{radio_address(address)}"


class AideckClient:
    """Blocking reader for the Bitcraze wifi-img-streamer TCP protocol."""

    def __init__(self, ip: str, port: int, timeout_sec: float) -> None:
        self._ip = ip
        self._port = port
        self._timeout_sec = timeout_sec
        self._socket: socket.socket | None = None

    @property
    def connected(self) -> bool:
        return self._socket is not None

    def connect(self) -> None:
        self.close()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(self._timeout_sec)
        sock.connect((self._ip, self._port))
        self._socket = sock

    def close(self) -> None:
        if self._socket is None:
            return
        try:
            self._socket.close()
        except OSError:
            pass
        self._socket = None

    def read_frame(self) -> tuple[int, int, int, bytes]:
        """Return width, height, format (0 = Bayer, otherwise JPEG), and payload."""
        packet_info = self._rx_bytes(4)
        length, _routing, _function = struct.unpack("<HBB", packet_info)
        header = self._rx_bytes(length - 2)
        if len(header) < 11:
            raise ValueError(f"AI-deck image header is {len(header)} bytes, expected at least 11")

        magic, width, height, _depth, fmt, size = struct.unpack_from("<BHHBBI", header)
        if magic != AIDECK_MAGIC:
            raise ValueError(f"Unexpected AI-deck magic byte: 0x{magic:02X}")
        if size < 0:
            raise ValueError(f"Unexpected image size: {size}")

        payload = bytearray()
        while len(payload) < size:
            chunk_info = self._rx_bytes(4)
            chunk_len, _dst, _src = struct.unpack("<HBB", chunk_info)
            payload.extend(self._rx_bytes(chunk_len - 2))
        return width, height, fmt, bytes(payload[:size])

    def _rx_bytes(self, size: int) -> bytearray:
        if self._socket is None:
            raise RuntimeError("Not connected to AI-deck")
        if size < 0:
            raise ValueError(f"Negative read size: {size}")

        data = bytearray()
        while len(data) < size:
            chunk = self._socket.recv(size - len(data))
            if not chunk:
                raise ConnectionError("AI-deck socket closed while reading")
            data.extend(chunk)
        return data


def write_gray_png(path: Path, width: int, height: int, pixels: bytes) -> None:
    expected = width * height
    if len(pixels) != expected:
        raise ValueError(f"Bayer frame is {len(pixels)} bytes, expected {expected}")

    raw = bytearray()
    for row in range(height):
        raw.append(0)
        start = row * width
        raw.extend(pixels[start : start + width])

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr)
    png += _png_chunk(b"IDAT", zlib.compress(bytes(raw), level=1))
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def save_frame(save_dir: Path, width: int, height: int, fmt: int, payload: bytes, last_stamp_ms: int) -> tuple[Path, int]:
    stamp_ms = max(int(time.time() * 1000), last_stamp_ms + 1)
    if fmt == 0:
        path = save_dir / f"{stamp_ms}.png"
        write_gray_png(path, width, height, payload)
    else:
        path = save_dir / f"{stamp_ms}.jpg"
        path.write_bytes(payload)
    return path, stamp_ms


def capture(
    ip: str,
    port: int,
    save_dir: Path,
    timeout_sec: float,
    reconnect_sec: float,
    count: int,
    address: str,
) -> None:
    uri = crazyflie_uri(address)
    save_dir.mkdir(parents=True, exist_ok=True)
    client = AideckClient(ip, port, timeout_sec)
    saved = 0
    last_stamp_ms = 0
    window_start = time.perf_counter()
    window_count = 0

    print(f"Drone {uri}", flush=True)
    print(f"Saving frames from {ip}:{port} to {save_dir}", flush=True)
    try:
        while count <= 0 or saved < count:
            try:
                if not client.connected:
                    print(f"Connecting to AI-deck at {ip}:{port}...", flush=True)
                    client.connect()
                    print("Connected.", flush=True)

                width, height, fmt, payload = client.read_frame()
                path, last_stamp_ms = save_frame(save_dir, width, height, fmt, payload, last_stamp_ms)
                saved += 1
                window_count += 1

                elapsed = time.perf_counter() - window_start
                if elapsed >= 2.0:
                    kind = "bayer" if fmt == 0 else "jpeg"
                    print(
                        f"{window_count / elapsed:.1f} Hz  saved={saved}  "
                        f"last={path.name}  {width}x{height} {kind}",
                        flush=True,
                    )
                    window_start = time.perf_counter()
                    window_count = 0
            except (ConnectionError, OSError, TimeoutError, ValueError) as exc:
                print(f"Stream error: {exc}", flush=True)
                client.close()
                if count > 0 and saved >= count:
                    break
                time.sleep(reconnect_sec)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        client.close()
        print(f"Saved {saved} frame(s) in {save_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save Crazyflie AI-deck images with millisecond timestamps.")
    parser.add_argument("--ip", default=DEFAULT_IP, help=f"AI-deck IP (default: {DEFAULT_IP})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"AI-deck TCP port (default: {DEFAULT_PORT})")
    parser.add_argument(
        "--address",
        default=DEFAULT_ADDRESS,
        help="Crazyflie radio address of the drone (default: 0xE7E7E7E7E7)",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("recordings"),
        help="Folder for timestamped images (default: ./recordings)",
    )
    parser.add_argument("--timeout", type=float, default=5.0, help="Socket timeout in seconds")
    parser.add_argument("--reconnect", type=float, default=2.0, help="Delay before reconnecting, in seconds")
    parser.add_argument("--count", type=int, default=0, help="Stop after this many frames (default: run until Ctrl-C)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture(
        ip=args.ip,
        port=args.port,
        save_dir=args.save_dir,
        timeout_sec=args.timeout,
        reconnect_sec=args.reconnect,
        count=args.count,
        address=args.address,
    )


if __name__ == "__main__":
    main()
