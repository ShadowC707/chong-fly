// Portable receiver core. No ESP-IDF, UART, radio, motor or fallback driver.
#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace chong {
using Pwm = std::array<double,4>;
using Session = std::array<std::uint8_t,16>;
enum class Mode { Manual, Shadow, Auto };
struct Command {
    Session session{};
    std::uint32_t sequence=0, ttl_us=0;
    bool has_lease=false;
    Pwm pwm{};
};
inline std::uint64_t read_le(const std::uint8_t* p, unsigned bytes) {
    std::uint64_t value=0;
    for (unsigned i=0;i<bytes;++i) value|=std::uint64_t(p[i])<<(8*i);
    return value;
}
inline std::uint32_t crc32(const std::uint8_t* p, std::size_t n) {
    std::uint32_t crc=0xffffffffu;
    for (std::size_t i=0;i<n;++i) {
        crc^=p[i];
        for (int bit=0;bit<8;++bit) crc=(crc>>1)^(0xedb88320u & (0u-(crc&1u)));
    }
    return ~crc;
}
inline bool decode_command(const std::uint8_t* p, std::size_t n, Command& out) {
    static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559,
                  "Wire v1 requires IEEE binary64");
    if (n!=69 || std::memcmp(p,"CFHW",4)!=0 || p[4]!=1 || p[5]!=2 || read_le(p+6,2)!=n
            || read_le(p+8,8)==0 || read_le(p+16,8)==0
            || crc32(p,n-4)!=read_le(p+n-4,4) || p[28]>1) return false;
    Command candidate;
    for (unsigned i=0;i<16;++i) candidate.session[i]=p[8+i];
    candidate.sequence=static_cast<std::uint32_t>(read_le(p+24,4));
    candidate.has_lease=p[28]==1;
    candidate.ttl_us=static_cast<std::uint32_t>(read_le(p+29,4));
    if (!candidate.has_lease) {
        for (unsigned i=29;i<65;++i) if (p[i]!=0) return false;
    } else {
        if (candidate.ttl_us==0 || candidate.ttl_us>40000) return false;
        for (unsigned i=0;i<4;++i) {
            const std::uint64_t bits=read_le(p+33+8*i,8);
            std::memcpy(&candidate.pwm[i],&bits,8); // bits, never a packed C struct
            if (!std::isfinite(candidate.pwm[i]) || candidate.pwm[i]<1000 || candidate.pwm[i]>2000)
                return false;
        }
    }
    out=candidate;
    return true;
}

class Receiver {
    const std::uint64_t boot_epoch_;
    std::uint64_t generation_, clock_=0, sent_=0, deadline_=0;
    bool clock_set_=false, session_set_=false, latest_set_=false, received_set_=false;
    std::uint32_t latest_=0, received_=0;
    Session session_{};
    void trip(const char* why) { tripped=true; active=false; reason=event_reason=why; }
    bool time(std::uint64_t now) {
        if (clock_set_ && now<clock_) { trip("invalid_receiver_clock"); return false; }
        clock_=now; clock_set_=true; return true;
    }
public:
    Mode mode=Mode::Manual;
    bool tripped=false, active=false, accepted=false;
    const char* reason="reset_required";
    const char* event_reason="reset_required";
    std::uint32_t active_sequence=0;
    Pwm pwm{};
    explicit Receiver(std::uint64_t boot_epoch, std::uint64_t generation=0)
        : boot_epoch_(boot_epoch), generation_(generation) {}
    const Session& session() const { return session_; }
    std::uint32_t latest() const { return latest_; }
    std::uint64_t deadline() const { return deadline_; }
    void poll(std::uint64_t now) {
        accepted=false;
        event_reason=reason;
        if (!time(now)) return;
        if (active && (mode!=Mode::Auto || now>=deadline_))
            trip(mode==Mode::Auto ? "watchdog_expired" : "authority_revoked");
    }
    bool reset(std::uint64_t now) {
        poll(now);
        if (clock_!=now) return false;
        if (mode!=Mode::Manual) { event_reason="reset_requires_manual"; return false; }
        if (!boot_epoch_ || generation_==UINT64_MAX) { trip("session_exhausted"); return false; }
        ++generation_;
        for (unsigned i=0;i<8;++i) {
            session_[i]=static_cast<std::uint8_t>(boot_epoch_>>(8*i));
            session_[8+i]=static_cast<std::uint8_t>(generation_>>(8*i));
        }
        session_set_=true; latest_set_=received_set_=false;
        latest_=received_=0; sent_=deadline_=0;
        active=accepted=tripped=false; reason=event_reason="awaiting_command";
        return true;
    }
    void set_mode(Mode requested,std::uint64_t now) {
        poll(now);
        if (clock_!=now) return;
        if (mode==Mode::Auto && requested!=Mode::Auto) trip("authority_revoked");
        mode=requested;
    }
    void announce(std::uint32_t sequence,std::uint64_t now) {
        poll(now);
        if (clock_!=now) return;
        if (!session_set_ || (latest_set_ && sequence<=latest_)) {
            trip("invalid_sensor_sequence"); return;
        }
        latest_=sequence; latest_set_=true; sent_=now;
    }
    void receive(const std::uint8_t* bytes,std::size_t length,std::uint64_t now) {
        poll(now); // mandatory watchdog priority even without an earlier output poll
        if (clock_!=now) return;
        Command c;
        if (!decode_command(bytes,length,c)) { trip("malformed_packet"); return; }
        if (!session_set_ || c.session!=session_) { event_reason="session_mismatch"; return; }
        if (!latest_set_ || c.sequence!=latest_ || (received_set_ && c.sequence<=received_)) {
            trip("sequence_rejected");
        } else if (!c.has_lease) {
            if (active) trip("sender_revoked");
            else if (!tripped) reason=event_reason="awaiting_command";
        } else if (tripped || mode!=Mode::Auto || now<sent_ || now-sent_>=c.ttl_us
                || sent_>UINT64_MAX-c.ttl_us) {
            trip("lease_rejected");
        } else {
            pwm=c.pwm; active=accepted=true; active_sequence=c.sequence;
            reason=event_reason="accepted"; deadline_=sent_+c.ttl_us;
        }
        if (!received_set_ || c.sequence>received_) received_=c.sequence;
        received_set_=true;
    }
};
} // namespace chong
