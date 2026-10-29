// Explicit CFHW v1 telemetry encoders. No struct memcpy or dynamic allocation.
#pragma once
#include "../receiver_core.hpp"

namespace chong::foundation {
inline constexpr std::uint32_t FRAME_PERIOD_US=20000;
template<std::size_t N> struct Measurement {
    std::uint32_t sample_id=0;
    std::uint64_t capture_us=0;
    bool valid=false;
    std::uint8_t quality=0;
    std::array<double,N> values{};
};
struct Sensors {
    Measurement<2> flow;
    Measurement<64> tof;
    Measurement<3> altitude;
    Measurement<2> attitude;
};
enum class Source : std::uint8_t { None, Manual, Auto };
struct Selection { Source source=Source::None; Pwm pwm{}; };
struct Frame {
    Session session{};
    std::uint32_t sequence=0;
    std::uint64_t created_us=0;
    Mode mode=Mode::Manual;
    bool stop=true, tripped=false, manual_valid=false;
    Pwm manual{};
    Selection selected;
    Sensors sensors;
};
struct Receipt {
    Session session{}, selected_session{};
    std::uint32_t sequence=0, selected_sequence=0;
    std::uint64_t now=0;
    const char* reason="awaiting_command";
    bool tripped=false, accepted=false;
    Mode mode=Mode::Manual;
    Pwm accepted_pwm{};
    Selection selected;
};
inline bool valid_session(const Session& s) { return read_le(s.data(),8)!=0 && read_le(s.data()+8,8)!=0; }
inline bool valid_pwm(const Pwm& p) {
    for (double x:p) if (!std::isfinite(x) || x<1000 || x>2000) return false;
    return true;
}
struct Writer {
    std::uint8_t* p;
    std::size_t offset=0;
    void integer(std::uint64_t value,unsigned count) {
        for (unsigned i=0;i<count;++i) p[offset++]=static_cast<std::uint8_t>(value>>(8*i));
    }
    void number(double value) {
        std::uint64_t bits=0; std::memcpy(&bits,&value,8); integer(bits,8);
    }
    void pwm(const Pwm& values,bool present) { for (double x:values) number(present ? x : 0.); }
    void session(const Session& s) { for (auto b:s) integer(b,1); }
    void header(const char* magic,std::uint8_t kind,std::uint16_t size,const Session& sid,std::uint32_t seq) {
        for (unsigned i=0;i<4;++i) integer(static_cast<std::uint8_t>(magic[i]),1);
        integer(1,1); integer(kind,1); integer(size,2); session(sid); integer(seq,4);
    }
    void finish() { const auto crc=crc32(p,offset); integer(crc,4); }
};
template<std::size_t N> inline bool measurement_ok(const Measurement<N>& m,std::uint64_t now,bool normalized=false) {
    if (m.capture_us>now) return false;
    for (double v:m.values) if (!std::isfinite(v) || (normalized && (v<0 || v>1))) return false;
    return true;
}
template<std::size_t N> inline void put_measurement(Writer& w,const Measurement<N>& m) {
    w.integer(m.sample_id,4); w.integer(m.capture_us,8); w.integer(m.valid,1); w.integer(m.quality,1);
    for (double value:m.values) w.number(value);
}
inline bool encode_sensor(const Frame& f,std::array<std::uint8_t,733>& bytes) {
    if (!valid_session(f.session) || (f.manual_valid && !valid_pwm(f.manual))
            || (f.selected.source!=Source::None && !valid_pwm(f.selected.pwm))
            || !measurement_ok(f.sensors.flow,f.created_us) || !measurement_ok(f.sensors.tof,f.created_us,true)
            || !measurement_ok(f.sensors.altitude,f.created_us) || !measurement_ok(f.sensors.attitude,f.created_us)) return false;
    Writer w{bytes.data()}; w.header("CFHW",1,733,f.session,f.sequence);
    w.integer(f.created_us,8); w.integer(static_cast<unsigned>(f.mode),1); w.integer(f.stop,1);
    w.integer(f.tripped,1); w.integer(static_cast<unsigned>(f.selected.source),1); w.integer(f.manual_valid,1);
    w.pwm(f.manual,f.manual_valid); w.pwm(f.selected.pwm,f.selected.source!=Source::None);
    put_measurement(w,f.sensors.flow); put_measurement(w,f.sensors.tof);
    put_measurement(w,f.sensors.altitude); put_measurement(w,f.sensors.attitude); w.finish();
    return w.offset==bytes.size();
}
inline unsigned reason_code(const char* reason) {
    constexpr const char* names[]={"awaiting_command","accepted","session_mismatch","sequence_rejected",
        "sender_revoked","malformed_lease","lease_rejected","watchdog_expired","authority_revoked",
        "invalid_receiver_clock","invalid_sensor_sequence","malformed_packet","reset_required",
        "reset_requires_manual","session_exhausted","invalid_mode"};
    for (unsigned i=0;i<17;++i) if (std::strcmp(names[i],reason)==0) return i;
    return 11; // no invented extension to the pinned receiver enum
}
inline bool encode_receipt(const Receipt& r,std::array<std::uint8_t,130>& bytes) {
    if (!valid_session(r.session) || (r.accepted && (r.tripped || std::strcmp(r.reason,"accepted")!=0
            || !valid_pwm(r.accepted_pwm))) || (r.selected.source!=Source::None && !valid_pwm(r.selected.pwm))
            || (r.selected.source==Source::Auto && (r.tripped || r.mode!=Mode::Auto || !valid_session(r.selected_session)))) return false;
    Writer w{bytes.data()}; w.header("CFHW",3,130,r.session,r.sequence);
    w.integer(r.now,8); w.integer(reason_code(r.reason),1); w.integer(r.tripped,1);
    w.integer(static_cast<unsigned>(r.mode),1); w.integer(r.accepted,1);
    w.integer(static_cast<unsigned>(r.selected.source),1); w.integer(0,1); // FC not_attempted
    w.session(r.selected.source==Source::Auto ? r.selected_session : Session{});
    w.integer(r.selected.source==Source::Auto ? r.selected_sequence : 0,4);
    w.pwm(r.accepted_pwm,r.accepted); w.pwm(r.selected.pwm,r.selected.source!=Source::None); w.finish();
    return w.offset==bytes.size();
}
} // namespace chong::foundation
