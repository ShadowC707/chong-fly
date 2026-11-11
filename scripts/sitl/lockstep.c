/* Experimental host-only harness, included by patched sitl.c. GPL-3.0-or-later.
 * Private local ABI v1, not MSP, ESP32, or upstream Betaflight protocol.
 * All firmware tasks, UART and virtual sensors execute on the main thread.
 */
#include "scheduler/scheduler.h"

static uint64_t harnessUs;
static uint32_t harnessMotorUpdates;
static bool harnessRcNew;

static uint8_t harnessRxStatus(rxRuntimeState_t *state)
{
    UNUSED(state);
    if (!harnessRcNew) return RX_FRAME_PENDING;
    harnessRcNew = false;
    return RX_FRAME_COMPLETE;
}

void chongHarnessRun(void)
{
    udpLink_t link;
    if (udpInit(&link, "127.0.0.1", 9005, true)) exit(2);
    uint8_t packet[197], cachedRequest[196], response[40] = {0};
    uint64_t lastSeq = 0;
    bool cached = false;
    rxRuntimeState.channelCount = 16;
    rxRuntimeState.rcReadRawFn = readRCSITL;
    rxRuntimeState.rcFrameStatusFn = harnessRxStatus;
    rxRuntimeState.rxProvider = RX_PROVIDER_UDP;
    dyad_setUpdateTimeout(0);
    printf("CHONG_LOCKSTEP_READY v1 clock_us=%llu\n", (unsigned long long)harnessUs);
    fflush(stdout);
    while (workerRunning) {
        int size = udpRecv(&link, packet, sizeof(packet), 100);
        if (size < 0) continue;
        if (size != 196 || memcmp(packet, "CFS1", 4)) continue;
        uint64_t seq;
        uint32_t duration, rcNew, status = 0;
        memcpy(&seq, packet+4, 8);
        memcpy(&duration, packet+12, 4);
        memcpy(&rcNew, packet+16, 4);
        if (cached && seq == lastSeq && !memcmp(packet, cachedRequest,196)) {
            udpSend(&link,response,sizeof(response));
            continue;
        }
        if (seq != lastSeq+1 || duration == 0 || duration > 4000 || duration%10 || rcNew>1) {
            uint8_t rejected[40] = {0};
            status = 1;
            memcpy(rejected,&seq,8); memcpy(rejected+8,&harnessUs,8);
            memcpy(rejected+20,&status,4);
            udpSend(&link,rejected,sizeof(rejected));
            continue;
        }
        fdm_packet state;
        memcpy(&state,packet+20,sizeof(state));
        if (rcNew) memcpy(rcPkt.channels,packet+164,32);
        harnessRcNew = rcNew;
        updateState(&state);
        uint64_t end = harnessUs + duration;
        uint32_t before = harnessMotorUpdates;
        /* Ten microseconds is the explicit scheduler time quantum, not CPU
         * execution latency. All task execution cost is idealized as zero. */
        while (harnessUs < end) {
            harnessUs += 10;
            dyad_update();
            scheduler();
        }
        uint32_t count = harnessMotorUpdates-before;
        memcpy(response,&seq,8); memcpy(response+8,&harnessUs,8);
        memcpy(response+16,&count,4); memcpy(response+20,&status,4);
        for (int i=0;i<4;i++) {
            float value = motorsPwm[i]/1000.0f; // native QUADX, no Gazebo permutation
            memcpy(response+24+i*4,&value,4);
        }
        lastSeq=seq; cached=true;
        memcpy(cachedRequest,packet,196);
        udpSend(&link,response,sizeof(response));
    }
}
