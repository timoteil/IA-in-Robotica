# YOLO26n depth webcam (Windows)

Live metric depth from a webcam, a 3D point cloud of that depth, and person tracking that combines a YOLOv5n box with the depth map. All three scripts run on Windows with a normal desktop webcam. Flying a Crazyflie is optional and is documented at the end.

| Script | What it does |
| --- | --- |
| `webcam_depth.py` | Camera image next to a metric depth image. Depth is in meters. |
| `webcam_pointcloud.py` | Same depth view, plus a point cloud you can orbit. |
| `person_track.py` | Finds one person, reads range from the depth map, and draws the track. `--source webcam` does not fly. |

Weights already live in this folder:

- `yolo26n-depth.pt` (about 12 MB) is the depth model.
- `yolov5n.pt` (about 4 MB) is the person detector used only by `person_track.py`.

`person_track.py` also imports `capture.py` and `follow_me.py` from the sibling folder `aideck_capture`. Keep that folder next to this one. The two webcam scripts do not need it.

```
RoboticsSpecialization-UPenn-AerialRobotics\
    aideck_capture\
        capture.py
        follow_me.py
    depth_yolo_person_tracking\
        webcam_depth.py
        webcam_pointcloud.py
        person_track.py
        requirements.txt
        yolo26n-depth.pt
        yolov5n.pt
```

## What to install first

1. **64-bit Python 3.12** from [python.org](https://www.python.org/downloads/windows/). During setup, enable **Add python.exe to PATH** and **pip**. Use the python.org installer, not the Microsoft Store stub. Python 3.11 also works. This project was run on 3.12.
2. **A webcam**, and Windows permission for desktop apps to use it: Settings, Privacy and security, Camera, then allow camera access and allow desktop apps.
3. **Git for Windows**, only if you will run `person_track.py`. The first launch clones YOLOv5 v7.0 through `torch.hub`. [git-scm.com](https://git-scm.com/download/win). After install, open a new terminal and check `git --version`.
4. **An NVIDIA GPU is optional.** The scripts run on CPU. A CUDA GPU makes depth much faster. Check a GPU with `nvidia-smi` in a terminal. If that command is missing, use the CPU install below.

## Create the environment

Open **PowerShell**. Go to this folder (change the path if yours differs):

```powershell
cd C:\path\to\RoboticsSpecialization-UPenn-AerialRobotics\depth_yolo_person_tracking
py -3.12 -m venv .venv
```

If PowerShell refuses to run the activate script, allow local scripts once for your user, then activate:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Command Prompt uses a different activate file:

```bat
.venv\Scripts\activate.bat
```

The prompt should start with `(.venv)`. Upgrade pip inside that environment:

```powershell
python -m pip install --upgrade pip
```

## Install the Python packages

Stay in the activated environment, in this folder.

### CPU only

Install the CPU build of PyTorch first, then the rest. Pip will keep that PyTorch because it already satisfies `requirements.txt`.

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install seaborn
```

### NVIDIA GPU

Install PyTorch with the CUDA build that matches your driver. Copy the current **pip** command from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) (Windows, Pip, Python, CUDA), run it, then:

```powershell
pip install -r requirements.txt
pip install seaborn
```

If you skip the PyTorch site and only run `pip install -r requirements.txt`, pip installs the default PyTorch wheel from PyPI. That wheel includes CUDA and also runs on CPU. The download is large.

`seaborn` is not in `requirements.txt`. `webcam_depth.py` and `webcam_pointcloud.py` do not need it. `person_track.py` loads YOLOv5 v7 with auto-install turned off, and that repo imports `seaborn`. Install it so the detector import does not fail.

## Check the install

```powershell
python -c "import torch, cv2; from ultralytics import YOLO; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('opencv', cv2.__version__)"
```

You want a torch version printed, `opencv` 4.8 or newer, and `cuda False` on a CPU install or `cuda True` when the NVIDIA wheel matches the driver.

Confirm the weights are real files, not empty copies:

```powershell
Get-Item .\yolo26n-depth.pt, .\yolov5n.pt | Select-Object Name, Length
```

`yolo26n-depth.pt` is about 12 MB. `yolov5n.pt` is about 4 MB. A file of a few hundred bytes is a broken copy. Replace it from the original project.

## Run

Run these from this folder with the virtual environment activated. The OpenCV windows must open on the Windows desktop. A camera window will not appear over SSH.

PyTorch and OpenCV both ship OpenMP. On Windows that can abort the process before a window opens, with `OMP: Error #15`. Set this once in the same terminal, then start a script:

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

Command Prompt:

```bat
set KMP_DUPLICATE_LIB_OK=TRUE
```

`person_track.py` sets the same variable itself. The two webcam scripts do not.

### Depth only

```powershell
python webcam_depth.py
```

The window shows the camera, the depth image, and a color bar. Near is dark, far is bright, on a fixed meter scale. The center crosshair prints distance in meters.

- `q` or `Esc` quits.
- `[` shrinks the far end of the color scale. `]` grows it.

Another camera:

```powershell
python webcam_depth.py --camera 1
```

CPU, or GPU 0:

```powershell
python webcam_depth.py --device cpu
python webcam_depth.py --device 0
```

`--imgsz` defaults to `768`, the size these depth weights were trained at. On CPU, `384` is faster and less sharp:

```powershell
python webcam_depth.py --device cpu --imgsz 384
```

### Depth and point cloud

```powershell
python webcam_pointcloud.py
```

Two windows. The depth window still uses `[` and `]`. The point-cloud window:

| Key or mouse | Action |
| --- | --- |
| Left drag | Orbit |
| Right drag | Pan |
| Scroll | Zoom |
| `j` / `l` | Orbit left / right |
| `i` / `k` | Orbit up / down |
| `-` / `=` | Zoom out / in |
| `r` | Reset the view on the current cloud |
| Space | Pause live updates |
| `q` or `Esc` | Quit |

The cloud is one frame at a time, in meters: X right, Y up, Z forward, origin at the camera. It is not a fused map. Horizontal field of view defaults to 70 degrees. Pass `--fov 60` or `--fx`, `--fy`, `--cx`, `--cy` if you know the lens.

Click a window before pressing keys. Windows sends the key to whichever window is focused.

### Person tracking on the webcam

The first run needs internet and Git. `torch.hub` clones `ultralytics/yolov5` tag `v7.0` into `%USERPROFILE%\.cache\torch\hub`. Later runs use that cache.

```powershell
python person_track.py --source webcam
```

Three windows: the track preview, a bird's-eye plot of the person on the floor (camera X to the right, camera Z forward), and the point cloud. `q` quits. This command does not connect to a drone. `--fly` is rejected together with `--source webcam`.

`--device auto` uses CUDA when PyTorch can see a GPU, otherwise CPU. The depth image size defaults to `384` here (the depth-only scripts default to `768`).

```powershell
python person_track.py --source webcam --device cpu --camera 0
```

## Optional: AI-deck stream and flying

`--source aideck` reads the Crazyflie AI-deck Wi-Fi stream at `192.168.4.1` port `5000`. Join that access point first. The sibling `aideck_capture` folder must sit next to this folder.

```powershell
python person_track.py --source aideck
```

`--fly` arms a Crazyflie and sends position setpoints. It requires the AI-deck source, a Crazyradio, and `cflib`. Install the radio library in the same virtual environment:

```powershell
pip install cflib
```

On Windows the Crazyradio needs a WinUSB driver. Bitcraze documents that with [Zadig](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/installation/usb_driver/). Install the driver before `--fly`. Close cfclient first. Only one program can hold the radio.

```powershell
python person_track.py --source aideck --fly --uri radio://0/80/2M/E7E7E7E7E7
```

`q` or Ctrl-C lands and disarms. Do this only with a drone you are allowed to fly, in a clear space, with the Flow deck the script checks for.

## If something fails

**`py` is not recognized.** Reinstall Python from python.org and enable the py launcher, or call `python -m venv .venv` if `python --version` already prints 3.11 or 3.12.

**Activate.ps1 cannot be loaded.** Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use Command Prompt and `.venv\Scripts\activate.bat`.

**`Could not open camera 0`.** Allow desktop apps to use the camera in Windows privacy settings. Close Zoom, Teams, or the Camera app, then try `--camera 1` and `--camera 2`.

**The process exits immediately with `OMP: Error #15`.** Set `KMP_DUPLICATE_LIB_OK` as shown in the Run section, then start the script again in that same terminal.

**`Weights not found`.** Run from `depth_yolo_person_tracking`, not from the repository root. Both `.pt` files must be in this folder at the sizes above.

**`No module named 'capture'` or `No module named 'follow_me'`.** `person_track.py` looks in the parent directory for `aideck_capture`. Copy or clone the whole repository so the two folders stay siblings.

**`git` is not recognized, or the YOLOv5 clone fails.** Install Git for Windows, open a new terminal, activate `.venv`, and run `person_track.py` again while online.

**`No module named 'seaborn'` (or pandas, IPython).** With the environment activated: `pip install seaborn pandas "ipython>=8"`.

**`cuda` is False on an NVIDIA PC.** The CPU wheel is installed, or the CUDA wheel does not match the driver. Check `nvidia-smi`, then reinstall PyTorch using the command from pytorch.org for that CUDA version. Activate `.venv` first so you replace the environment's torch, not another Python.

**DLL load failed while importing torch or cv2.** Install the latest [Microsoft Visual C++ Redistributable (x64)](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist), then reopen the terminal.

**Depth is slow.** Expect a low frame rate on CPU at `--imgsz 768`. Use `--device 0` on an NVIDIA GPU, or `--imgsz 384` on CPU.
