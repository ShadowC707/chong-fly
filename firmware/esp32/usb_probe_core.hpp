// Diagnostic-only USB descriptor selection and candidate XUSB input decoding.
// Synthetic tests use Xbox 360 USB layout; this is not a GPW70 capture or RC map.
#pragma once
#include <cstddef>
#include <cstdint>

namespace chong::usb_probe {
struct Endpoint {
    bool valid=false, found=false;
    std::uint8_t interface=0, alternate=0, address=0;
    std::uint16_t packet_size=0;
};
inline std::uint16_t le16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0]|(std::uint16_t(p[1])<<8));
}
inline Endpoint select(const std::uint8_t* p,std::size_t n) {
    Endpoint result;
    if (n<9 || n>4096 || p[0]!=9 || p[1]!=2 || le16(p+2)!=n) return result;
    bool candidate=false;
    unsigned expected=0, seen=0, inputs=0;
    Endpoint pending;
    for (std::size_t pos=9;pos<n;) {
        if (n-pos<2 || p[pos]<2 || p[pos]>n-pos) return {};
        const auto* d=p+pos;
        if (d[1]==4) {
            if (d[0]!=9 || (candidate && seen!=expected)) return {};
            if (candidate && inputs==1) {
                if (result.found) return {}; // ambiguous matching interfaces
                result=pending; result.found=true;
            }
            candidate=d[3]==0 && d[5]==0xff && d[6]==0x5d && d[7]==1;
            expected=d[4]; seen=inputs=0;
            pending={}; pending.interface=d[2]; pending.alternate=d[3];
        } else if (d[1]==5 && candidate) {
            if (d[0]!=7) return {};
            ++seen;
            if ((d[2]&0x80) && (d[3]&3)==3) {
                ++inputs;
                const auto size=le16(d+4);
                if (inputs>1 || size<20 || size>64 || (d[2]&0x0f)==0 || d[6]==0) return {};
                pending.address=d[2]; pending.packet_size=size;
            }
        }
        pos+=d[0];
    }
    if (candidate && seen!=expected) return {};
    if (candidate && inputs==1) {
        if (result.found) return {};
        result=pending; result.found=true;
    }
    result.valid=true;
    return result;
}
struct Input {
    std::uint16_t buttons=0;
    std::uint8_t lt=0, rt=0;
    std::int32_t lx=0, ly=0, rx=0, ry=0;
};
inline std::int32_t signed16(const std::uint8_t* p) {
    const auto value=le16(p);
    return value<32768 ? value : static_cast<std::int32_t>(value)-65536;
}
inline bool decode(const std::uint8_t* p,std::size_t n,Input& out) {
    if (n!=20 || p[0]!=0 || p[1]!=20) return false;
    Input candidate;
    candidate.buttons=le16(p+2); candidate.lt=p[4]; candidate.rt=p[5];
    candidate.lx=signed16(p+6); candidate.ly=signed16(p+8);
    candidate.rx=signed16(p+10); candidate.ry=signed16(p+12);
    out=candidate;
    return true;
}
} // namespace chong::usb_probe
