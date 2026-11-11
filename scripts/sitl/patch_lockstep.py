"""Apply a reviewed, version-guarded host-only harness to the isolated image.

No PID, ANGLE, mixer, rate curves or flight failsafe algorithms are replaced.
Writes the exact source diff for auditing alongside the binary.
"""
from pathlib import Path
import subprocess

ROOT=Path('/opt/betaflight')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()=='79065c96ba0bb5cdc675e67d7093e05dab8b330e'


def edit(path,old,new):
    p=ROOT/path
    text=p.read_text()
    if text.count(old)!=1:
        raise RuntimeError(f'Patch context differs: {path}: {old[:70]}')
    p.write_text(text.replace(old,new))


sitl='src/platform/SIMULATOR/sitl.c'
edit(sitl,'static fdm_packet fdmPkt;', '''static uint64_t harnessUs;
static uint32_t harnessMotorUpdates;
static fdm_packet fdmPkt;''')
# Skip wall-derived clock and the cross-thread mutex handoff, retaining the
# original conversions to virtual sensors/quaternion/pressure exactly.
edit(sitl,'    if (realtime_now > last_realtime + 500*1e3) {','    if (false && realtime_now > last_realtime + 500*1e3) {')
edit(sitl,'    if (deltaSim < 0.02 && deltaSim > 0) {','    if (false && deltaSim < 0.02 && deltaSim > 0) {')
edit(sitl,'    pthread_mutex_unlock(&updateLock); // can send PWM output now\n\n#if defined(SIMULATOR_GYROPID_SYNC)',
     '    // Main-thread lockstep owns sensor/motor state; no foreign unlock.\n\n#if defined(SIMULATOR_GYROPID_SYNC)')
edit(sitl,'    ret = pthread_create(&tcpWorker, NULL, tcpThread, NULL);','''    dyad_init(); // Before serial ports can allocate dyad streams.
    dyad_setUpdateTimeout(0);
    return; // Harness owns IO on the main thread; no legacy UDP/TCP workers.
    ret = pthread_create(&tcpWorker, NULL, tcpThread, NULL);''')
edit(sitl,'uint64_t micros64(void)\n{','uint64_t micros64(void)\n{\n    return harnessUs;')
edit(sitl,'uint64_t millis64(void)\n{','uint64_t millis64(void)\n{\n    return harnessUs / 1000;')
edit(sitl,'    microsleep(us / simRate);','    harnessUs += us;')
edit(sitl,'void delay(uint32_t ms)\n{','void delay(uint32_t ms)\n{\n    harnessUs += (uint64_t)ms * 1000;\n    return;')
edit(sitl,'void pwmCompleteMotorUpdate(void)\n{','void pwmCompleteMotorUpdate(void)\n{\n    harnessMotorUpdates++;\n    return; // Reply only after the complete requested interval.')
# No worker threads exist in this build; save/reset must not join invalid IDs.
edit(sitl,'void systemReset(void)\n{','void systemReset(void)\n{\n    exit(0);')
edit(sitl,'void systemResetToBootloader(bootloaderRequestType_e requestType)\n{',
     'void systemResetToBootloader(bootloaderRequestType_e requestType)\n{\n    UNUSED(requestType);\n    exit(0);')
content=Path('/opt/harness/lockstep.c').read_text()
content=content.replace('static uint64_t harnessUs;\n','').replace('static uint32_t harnessMotorUpdates;\n','')
p=ROOT/sitl
p.write_text(p.read_text()+'\n'+content)
edit('src/main/main.c','void FAST_CODE run(void)\n{',
     'void chongHarnessRun(void);\nvoid FAST_CODE run(void)\n{\n    chongHarnessRun();\n    return;')
# A frozen clock cannot busy-spin to a future boundary. Return to harness,
# which advances exactly 10 us; only dispatch gyro once its deadline is due.
edit('src/main/scheduler/scheduler.c',
     'if (schedLoopRemainingCycles < schedLoopStartCycles) {',
     'if (schedLoopRemainingCycles <= 0) { // CHONG_LOCKSTEP: no wall-clock spin')
diff=subprocess.check_output(['git','diff','--','src'],cwd=ROOT)
(ROOT/'chong-lockstep.patch').write_bytes(diff)
