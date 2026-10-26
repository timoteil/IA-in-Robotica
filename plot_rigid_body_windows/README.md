# Live rigid-body plot on Windows

Copy this folder to the Windows PC. The only third-party package is matplotlib. Pip also installs the libraries matplotlib needs, including numpy.

Install Python 3.11 or 3.12 from https://www.python.org/downloads/ and check **Add python.exe to PATH**. That installer includes Tk, which matplotlib uses for the plot window.

A virtual environment is tied to the machine that creates it. Create it on the Windows PC. Do not copy a `.venv` folder from the Mac.

## Command Prompt

Open Command Prompt, then run these commands. Replace the folder path with wherever you copied this folder, and replace the Motive IP with the Motive PC address.

```bat
cd /d C:\path\to\plot_rigid_body_windows
py -3 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python plot_rigid_body.py 192.168.0.164
```

`setup.bat` runs the venv and install commands for you:

```bat
cd /d C:\path\to\plot_rigid_body_windows
setup.bat
.venv\Scripts\activate.bat
python plot_rigid_body.py 192.168.0.164
```

## PowerShell

```powershell
cd C:\path\to\plot_rigid_body_windows
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python plot_rigid_body.py 192.168.0.164
```

If activation is blocked, run this once in that window, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

`setup.ps1` runs the venv and install commands for you:

```powershell
cd C:\path\to\plot_rigid_body_windows
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\Activate.ps1
python plot_rigid_body.py 192.168.0.164
```

Unblock Python in Windows Firewall

1. Click the Windows Start button (or press Windows key)

2. Type PowerShell

3. Right-click on Windows PowerShell and choose "Run as administrator".

4. In that administrator window, paste and run 

```powershell
SetNetFirewallRule -DisplayName "python.exe" -Action Allow
```


## Motive

In Motive, set Transmission Type to Multicast. Turn on Broadcast Frame Data. Data port `1511`, multicast address `239.255.42.99`. Quit any other NatNet client first. It holds UDP port 1511.

When Windows asks, allow Python on the private network so it can receive UDP.

If this PC has more than one network adapter, pass the address of the adapter on the Motive network:

```bat
python plot_rigid_body.py 192.168.0.164 --client 192.168.0.10
```

The default rigid body is ID 2. Use `--body-id` for a different one. Close the plot window to stop.
