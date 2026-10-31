@echo off
rem Start the desktop app from the per-user installation made by install.ps1.
rem Pass --workspace D:\path to use a different local workspace.
start "" "%LOCALAPPDATA%\MicroSegWorkbench\venv\Scripts\pythonw.exe" -m microseg_workbench.gui.app %*
