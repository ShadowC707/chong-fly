# Chong-Fly ESP32 application foundation

Current target: `app/` (ESP-IDF). Board-neutral application logic: `foundation/`.
Two ground peers: Raspberry Pi runs the pinned model; notebook carries GPW70.
Start with the [colleague handover and one-command offline check](../../docs/HARDWARE_COLLEAGUE_UK.md).
Configuration, tasks, packet tables, tests and pending adapters:
[ESP32_FOUNDATION_UK.md](../../docs/ESP32_FOUNDATION_UK.md).
Default Wi-Fi is off, sensors are invalid and FC output has no implementation.
Build only (does not upload):

```powershell
& .venv-esp32-build/Scripts/python.exe firmware/esp32/pio_local.py run --project-dir firmware/esp32/app --jobs 2
& firmware/esp32/build_host.ps1 -Foundation -OutDir .test-tmp-foundation-host
```

Legacy core and USB diagnostic follow below; USB Host is not required by the
current notebook manual-input architecture.

Contract and limitations: [ESP32_WIRE_V1_UK.md](../../docs/ESP32_WIRE_V1_UK.md).

`receiver_core.hpp` contains a fixed-memory C++17 receiver and COMMAND decoder.
It deliberately has no board entrypoint, GPIO mapping, FC output or invented
fallback. The platform must supply trusted local mode/reset actions, persistent
boot epochs, a monotonic microsecond clock and independently scheduled polls.
The new `app/` supplies the NVS epoch, local console, monotonic clock and tasks.
Its scheduling and persistence failure behavior are not hardware-qualified yet.

Windows host build (MSVC x64 + Windows SDK):

```powershell
& firmware/esp32/build_host.ps1 -OutDir .test-tmp-esp32-build
& venv/Scripts/python.exe -m pytest tests/test_esp32_host.py tests/test_hardware_wire.py tests/test_hardware_mock.py -q --basetemp .test-tmp-esp32-tests
```

The host harness uses bounded text input for testing. It is not the device
wire protocol. `foundation/wire.hpp` now also encodes SENSOR/RECEIPT in C++;
the original receiver harness continues to decode COMMAND only.
Golden packets are shared under `tests/fixtures/`.
The separate `usb_probe/` target uses PlatformIO Core 6.2.0,
`espressif32@6.12.0` and ESP-IDF 5.5.0. It is a USB input diagnostic;
it does not run the receiver core or send RC commands. Setup, verification and
limitations: [USB_PROBE_FIRMWARE_UK.md](../../docs/hardware/USB_PROBE_FIRMWARE_UK.md).

Optional read-only ROM diagnostics use a separate `.venv-esp32` environment with
`requirements-host-tools.txt` (esptool 5.6.0). It is independent of the model's
Python environment. The connected board was detected on COM14; ports can change.
The successful ROM `flash-id` observation is recorded in
`docs/hardware/esp32_chip_probe_v1.json`. It did not write/erase firmware or eFuses.

Build tooling is isolated in `.venv-esp32-build` with `requirements-build.txt`.
The project-local launcher stores the SDK under ignored `.pio-esp32/` and bounds
package download timeouts. Build without uploading:

```powershell
& .venv-esp32-build/Scripts/python.exe firmware/esp32/pio_local.py run --project-dir firmware/esp32/usb_probe --jobs 2
```

`usb_probe_core.hpp` and `host/usb_probe_host.cpp` test descriptor selection and
a candidate Xbox 360 packet layout on Windows. Their fixtures are synthetic.
No GPW70 packet or radio-loss behavior is inferred from those tests.
