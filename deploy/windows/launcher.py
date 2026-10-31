"""Entry point for a packaged (PyInstaller) build of the desktop app.

The frozen executable starts the GUI; started as ``MicroSegWorkbench.exe --cli <msw arguments>`` it runs the
command line instead (this is how the app launches training in a child process when frozen).
"""

import multiprocessing
import sys

from microseg_workbench.gui.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
