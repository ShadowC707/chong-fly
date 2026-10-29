// Single state owner / externally locked application. No OS or I/O in this file.
#pragma once
#include "manual.hpp"

namespace chong::foundation {
template<std::size_t N> class SampleSlot {
    Measurement<N> stored_{};
    bool seen_=false,healthy_=true;
public:
    bool update(const Measurement<N>& m,std::uint64_t now,bool normalized=false) {
        if (!measurement_ok(m,now,normalized)) { healthy_=false; return false; }
        if (seen_) {
            if (m.sample_id==stored_.sample_id && m.capture_us==stored_.capture_us
                    && m.values==stored_.values && m.valid==stored_.valid && m.quality==stored_.quality) return healthy_;
            if (m.sample_id<=stored_.sample_id || m.capture_us<=stored_.capture_us) { healthy_=false; return false; }
        }
        stored_=m; seen_=healthy_=true; return true;
    }
    Measurement<N> snapshot() const { auto m=stored_; m.valid=m.valid && healthy_; return m; }
};
struct SensorCache {
    SampleSlot<2> flow;
    SampleSlot<64> tof;
    SampleSlot<3> altitude;
    SampleSlot<2> attitude;
    Sensors snapshot() const { return {flow.snapshot(),tof.snapshot(),altitude.snapshot(),attitude.snapshot()}; }
};
class Runtime {
    bool clock_set_=false,clock_fault_=false;
    bool frame_set_=false,platform_fault_=false;
    std::uint32_t frame_sequence_=0;
    std::uint64_t clock_=0,frame_created_=0;
    const bool bench_auto_allowed_;
public:
    Receiver receiver;
    ManualGate manual;
    SensorCache sensors;
    const char* local_reason="manual_default";
    explicit Runtime(std::uint64_t epoch,bool bench_auto_allowed=false,std::uint32_t manual_timeout=100000)
        :bench_auto_allowed_(bench_auto_allowed),receiver(epoch),manual(manual_timeout) {}
    void poll(std::uint64_t now) {
        if (clock_set_ && now<clock_) { clock_fault_=true; manual.invalidate("clock_regressed"); }
        clock_=now; clock_set_=true;
        receiver.poll(now); manual.poll(now);
        if ((manual.tripped || clock_fault_) && receiver.mode==Mode::Auto) receiver.set_mode(Mode::Manual,now);
    }
    bool reset(std::uint64_t now) {
        poll(now);
        if (clock_fault_ || !receiver.reset(now)) return false;
        manual.reset(receiver.session()); frame_set_=platform_fault_=false; local_reason="local_reset"; return true;
    }
    bool mode(Mode requested,std::uint64_t now) {
        poll(now);
        if (clock_fault_) return false;
        if (requested==Mode::Auto && (!bench_auto_allowed_ || !manual.active || manual.tripped || receiver.tripped || platform_fault_)) {
            local_reason="auto_not_authorized"; return false;
        }
        receiver.set_mode(requested,now); local_reason="local_mode"; return true;
    }
    Selection selection() const {
        if (clock_fault_) return {};
        if (receiver.mode==Mode::Auto) {
            if (receiver.active && !receiver.tripped && !platform_fault_ && manual.active && !manual.tripped)
                return {Source::Auto,receiver.pwm};
            return {};
        }
        return manual.active && !manual.tripped ? Selection{Source::Manual,manual.pwm} : Selection{};
    }
    bool frame(std::uint32_t seq,std::uint64_t now,Frame& out) {
        const bool replay=frame_set_ && seq<=frame_sequence_;
        poll(now); receiver.announce(seq,now);
        if (clock_fault_ || replay || !valid_session(receiver.session()) || receiver.latest()!=seq) return false;
        frame_set_=true; frame_sequence_=seq; frame_created_=now;
        manual.announce(seq,now);
        out.session=receiver.session(); out.sequence=seq; out.created_us=now; out.mode=receiver.mode;
        out.tripped=receiver.tripped || platform_fault_; out.stop=manual.active ? manual.stop : true;
        out.manual_valid=manual.active && !manual.tripped; out.manual=manual.pwm;
        out.selected=selection(); out.sensors=sensors.snapshot(); return true;
    }
    void receive_manual(const std::uint8_t* p,std::size_t n,std::uint64_t now) {
        poll(now); if (clock_fault_) return;
        bool takeover=false; manual.receive(p,n,now,takeover);
        if (takeover || manual.tripped) receiver.set_mode(Mode::Manual,now);
    }
    bool receive_pi(const std::uint8_t* p,std::size_t n,std::uint64_t now,Receipt& out) {
        poll(now); if (clock_fault_) return false;
        Command command; const bool decoded=decode_command(p,n,command);
        bool rejected=false; const char* platform_reason="lease_rejected";
        if (decoded && command.session==receiver.session() && receiver.mode==Mode::Auto) {
            if (!receiver.tripped && frame_set_ && command.sequence==frame_sequence_
                    && now>=frame_created_ && now-frame_created_>=FRAME_PERIOD_US) {
                platform_fault_=true; platform_reason="sequence_rejected"; local_reason="frame_deadline_missed";
            }
            rejected=platform_fault_;
        }
        // A stalled publisher must not turn a nominal 20-ms response window
        // into the 40-ms active-lease hold time. The reference core is unchanged.
        if (!rejected) receiver.receive(p,n,now);
        if (!decoded) return false;
        out.session=command.session; out.sequence=command.sequence; out.now=now;
        out.reason=rejected ? platform_reason : receiver.event_reason;
        out.tripped=receiver.tripped || platform_fault_; out.mode=receiver.mode;
        out.accepted=!rejected && receiver.accepted; out.accepted_pwm=receiver.pwm; out.selected=selection();
        if (out.selected.source==Source::Auto) {
            out.selected_session=receiver.session(); out.selected_sequence=receiver.active_sequence;
        }
        return true;
    }
    void disconnect(std::uint64_t now) {
        poll(now); manual.invalidate("link_lost"); receiver.set_mode(Mode::Manual,now);
        local_reason="link_lost_requires_reset";
    }
    bool platform_fault() const { return platform_fault_; }
};
} // namespace chong::foundation
