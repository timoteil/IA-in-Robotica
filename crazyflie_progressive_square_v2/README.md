# Crazyflie progressive square

Four stages of a 1 m square on a real Crazyflie, using OptiTrack for position.
This folder is the whole project. It does not need Webots or any other repository.

Each stage is one function, `step()`, and that function is the control cycle in the laboratory notes. A later stage keeps the earlier cycle and adds the next block.

| Script | `step()` |
| --- | --- |
| `step1_position.py` | Equations (3) and (4). Position setpoint. Full stop at every corner. |
| `step2_small_yaw.py` | Stage 1 velocity, then equations (5) through (9). A small yaw toward the next side near each corner. |
| `step3_cruise.py` | Stage 2 yaw, then equations (10) and (11). Cruise speed. Middle corners are passed. The last corner still stops. |
| `step4_acceleration.py` | Stage 3, then equations (12) through (14). An acceleration limit. The status line prints the tilt that limit asks for. |

`Ctrl+C` lands, then disarms. Close the Crazyflie client before you run a script. The radio allows one link, and that link carries both OptiTrack and the setpoints.

## Install

Python 3.10 or newer, a Crazyradio, and a Crazyflie with a firmware that supports arming and hover setpoints.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS the Crazyradio also needs libusb. The [Bitcraze USB permission instructions](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/installation/install/) cover Windows, macOS, and Linux.

## Motive

In Motive, stream rigid bodies over multicast:

- Transmission type: Multicast
- Data port: `1511`
- Multicast address: `239.255.42.99`

The laptop running these scripts must be on that network. Pass Motive's IP and this computer's IP on that network. `--body-id` is the rigid-body id in Motive. `--address` is the Crazyflie radio address, for example `0xE7E7E7E7E7`.

## Calibrate once per drone

Motors stay off. Slide the drone by hand: up, toward the nose, then to its left. The nose slide is the wall the drone will fly toward.

```bash
python3 calibrate_axes.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7
```

That writes `axis_calibration_4.json` in this folder. It is local to the lab and is not meant to be committed. Repeat the calibration if the marker rigid body is rebuilt.

## Fly

Put the drone on the floor, nose aimed at the calibration wall, with a clear square in front of it. `+X` is toward that wall. `+Y` is to the left of the nose. The square starts at the takeoff point, 1 m on a side, and climbs 0.5 m. `--side` and `--climb` change those two numbers.

```bash
python3 step1_position.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7
python3 step2_small_yaw.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7
python3 step3_cruise.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7
python3 step4_acceleration.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7
```

Each script asks you to hold the drone still and press Enter so the current pose becomes forward. Then it waits until the onboard estimate matches OptiTrack, and asks for another Enter before arming.

Fly stage 1 before the later stages. Gains live in `gains.py`. Change one number at a time, after the stage that uses it is already flying.

## What each file does

- `step1_position.py` through `step4_acceleration.py` are the programs you run. Each one contains `step()`, the whole control cycle for that stage.
- `runner.py` stores the waypoint index, the arrival timer, the yaw setpoint, and the last velocity, and calls `step()`.
- `geometry.py` turns world velocity into the body frame.
- `fly_square.py` is the shared flight loop: takeoff, the square, and landing.
- `mocap.py` sends the OptiTrack pose to the Crazyflie and stores the axis map.
- `motive.py` reads Motive's multicast packets.
- `calibrate_axes.py` measures that axis map.
