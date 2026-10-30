# Crazyflie progressive square on Windows

Four stages of a 1 m square on a real Crazyflie, using OptiTrack for position.
This folder is the whole project. It does not need Webots or any other repository.

Each stage is one function, `step()`, and that function is the control cycle in the laboratory notes. A later stage keeps the earlier cycle and adds the next block.

| Script | `step()` |
| --- | --- |
| `step1_position.py` | Equations (3) and (4). Position setpoint. Full stop at every corner. |
| `step2_small_yaw.py` | Stage 1 velocity, then equations (5) through (9). A small yaw toward the next side near each corner. |
| `step3_cruise.py` | Stage 2 yaw, then equations (10) and (11). Cruise speed. Middle corners are passed. The last corner still stops. |
| `step4_acceleration.py` | Stage 3, then equations (12) through (14). An acceleration limit. The status line prints the tilt that limit asks for. |

`Ctrl+C` in the terminal lands, then disarms. Close the Crazyflie client before you run a script. The radio allows one link, and that link carries both OptiTrack and the setpoints.

## Install

You need Windows 10 or 11, Python 3.10 or newer, a Crazyradio, and a Crazyflie with firmware that supports arming and hover setpoints.

### 1. Python

Install Python from [python.org](https://www.python.org/downloads/windows/). During setup, check **Add python.exe to PATH**.

Python 3.12 is the straightforward choice. Python 3.13 also works, but [Bitcraze requires Visual Studio](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/installation/install/) with the **Desktop development with C++** workload for that version.

Open PowerShell and check that `python` is the install you just made, not the Microsoft Store stub:

```powershell
python --version
```

If that opens the Store, or prints a version you did not install, turn off the `python.exe` and `python3.exe` app execution aliases under **Settings → Apps → Advanced app settings → App execution aliases**, then open a new PowerShell window.

### 2. This folder

Copy `crazyflie_progressive_square_v2` onto the PC, or clone the repository that contains it. In PowerShell, change into that folder. The prompt's path is the folder the commands below use.

```powershell
cd C:\path\to\crazyflie_progressive_square_v2
python -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The prompt then starts with `(.venv)`. Open a new terminal later and run `.\.venv\Scripts\Activate.ps1` again before `pip` or a flight script. Command Prompt uses `.venv\Scripts\activate.bat` instead of the PowerShell script.

If PowerShell refuses to run `Activate.ps1`, allow local scripts for your user, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 3. Crazyradio driver

Windows does not ship the driver the Python library uses. Install it with [Zadig](https://zadig.akeo.ie/), following the [Bitcraze Windows USB driver page](https://www.bitcraze.io/documentation/repository/crazyradio-firmware/master/building/usbwindows/).

1. Plug in the Crazyradio. Dismiss any Windows driver dialog that appears.
2. Run Zadig.
3. Select the Crazyradio in the device list. If it is missing, open **Options → List All Devices** and select it there.
4. Choose the **libusb-win32** driver. The Bitcraze page calls this driver libusb.
5. Click **Install Driver**, or **Replace Driver** if another driver is already installed.

In Device Manager the dongle should show up as a libusb device. If the radio is seen and then drops, install the **libusbK** driver the same way. Bitcraze documents that fallback on the same page.

## Motive

In Motive, stream rigid bodies over multicast:

- Transmission type: Multicast
- Data port: `1511`
- Multicast address: `239.255.42.99`

The laptop running these scripts must be on that network. Pass Motive's IP and this computer's IP on that network. `--body-id` is the rigid-body id in Motive. `--address` is the Crazyflie radio address, for example `0xE7E7E7E7E7`.

`--client` is the IPv4 address of the adapter on the OptiTrack network. With both Wi-Fi and Ethernet connected, that is the Ethernet address, not the Wi-Fi address. Find it with:

```powershell
ipconfig
```

The first time a script listens, Windows Firewall may ask to allow Python. Allow it on private networks. The script binds UDP port `1511`. Close any other NatNet client on this PC first, including another copy of these scripts.

## Calibrate once per drone

Motors stay off. Slide the drone by hand: up, toward the nose, then to its left. The nose slide is the wall the drone will fly toward.

```powershell
python calibrate_axes.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7
```

That writes `axis_calibration_4.json` in this folder. It is local to the lab and is not meant to be committed. Repeat the calibration if the marker rigid body is rebuilt.

## Fly

Put the drone on the floor, nose aimed at the calibration wall, with a clear square in front of it. `+X` is toward that wall. `+Y` is to the left of the nose. The square starts at the takeoff point, 1 m on a side, and climbs 0.5 m. `--side` and `--climb` change those two numbers.

```powershell
python step1_position.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7
python step2_small_yaw.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7
python step3_cruise.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7
python step4_acceleration.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7
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
