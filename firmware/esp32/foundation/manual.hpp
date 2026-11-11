#pragma once
#include "wire.hpp"

namespace chong::foundation {
struct ManualInput {
    Session session{};
    std::uint32_t sequence=0;
    bool valid=false, takeover=false, stop=true;
    Pwm pwm{};
};
inline bool decode_manual(const std::uint8_t* p,std::size_t n,ManualInput& out) {
    if (n!=67 || std::memcmp(p,"CFMI",4)!=0 || p[4]!=1 || p[5]!=2 || read_le(p+6,2)!=n
            || !read_le(p+8,8) || !read_le(p+16,8) || crc32(p,n-4)!=read_le(p+n-4,4)
            || p[28]>1 || p[29]>1 || p[30]>1) return false;
    ManualInput m;
    for (unsigned i=0;i<16;++i) m.session[i]=p[8+i];
    m.sequence=static_cast<std::uint32_t>(read_le(p+24,4));
    m.valid=p[28]!=0; m.takeover=p[29]!=0; m.stop=p[30]!=0;
    if (m.valid) {
        for (unsigned i=0;i<4;++i) {
            const auto bits=read_le(p+31+8*i,8); std::memcpy(&m.pwm[i],&bits,8);
        }
        if (!valid_pwm(m.pwm)) return false;
    } else for (unsigned i=31;i<63;++i) if (p[i]!=0) return false;
    out=m; return true;
}
inline void encode_challenge(const Frame& f,std::array<std::uint8_t,40>& bytes) {
    Writer w{bytes.data()}; w.header("CFMI",1,40,f.session,f.sequence); w.integer(f.created_us,8); w.finish();
}
class ManualGate {
    Session session_{};
    std::uint32_t challenge_=0,received_=0;
    std::uint64_t sent_=0,deadline_=0;
    const std::uint32_t timeout_;
    bool challenge_set_=false,received_set_=false;
public:
    bool active=false,tripped=false,stop=true;
    Pwm pwm{};
    const char* reason="awaiting_manual";
    explicit ManualGate(std::uint32_t timeout=100000):timeout_(timeout) {}
    void invalidate(const char* why) { active=false; tripped=true; reason=why; }
    void reset(const Session& s) {
        session_=s; active=tripped=challenge_set_=received_set_=false;
        stop=true; sent_=deadline_=0; reason="awaiting_manual";
    }
    void poll(std::uint64_t now) { if (active && now>=deadline_) invalidate("manual_expired"); }
    void announce(std::uint32_t seq,std::uint64_t now) { challenge_=seq; sent_=now; challenge_set_=true; }
    bool receive(const std::uint8_t* p,std::size_t n,std::uint64_t now,bool& takeover) {
        poll(now); takeover=false;
        ManualInput input;
        if (!decode_manual(p,n,input)) { invalidate("manual_malformed"); return false; }
        if (input.session!=session_) { reason="manual_session_mismatch"; return false; }
        if (!challenge_set_ || input.sequence!=challenge_ || (received_set_ && input.sequence<=received_)) {
            reason="manual_sequence_rejected"; return false;
        }
        if (tripped || now<sent_ || now-sent_>=FRAME_PERIOD_US || timeout_==0 || timeout_>100000
                || sent_>UINT64_MAX-timeout_) { reason="manual_rejected"; return false; }
        received_=input.sequence; received_set_=true;
        takeover=input.takeover;
        if (!input.valid) { invalidate("manual_unavailable"); return false; }
        active=true; pwm=input.pwm; stop=input.stop; deadline_=sent_+timeout_; reason="manual_accepted";
        return true;
    }
};
} // namespace chong::foundation
