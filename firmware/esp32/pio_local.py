"""Project-local PlatformIO launcher with bounded network read timeouts."""
import os
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[2]
os.environ['PLATFORMIO_CORE_DIR']=str(root/'.pio-esp32')
os.environ['PLATFORMIO_SETTING_ENABLE_TELEMETRY']='No'

import platformio.http
from platformio.__main__ import main

# The default has no read deadline. A stalled package mirror must fail so that
# its retry/mirror handling can run; this does not alter installed source files.
platformio.http.__default_requests_timeout__=(10,30)

if __name__=='__main__':
    raise SystemExit(main(['pio',*sys.argv[1:]]))
