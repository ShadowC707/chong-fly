#pragma once
#include <array>
#include <cstdint>
#include <cstring>

namespace chong::foundation {
enum class WifiMode { Disabled, Station, AccessPoint };
struct Peer { const char* address=""; std::uint16_t port=0; };
struct PinPlan {
    int spi_sck=-1,spi_mosi=-1,spi_miso=-1,pmw_cs=-1;
    int i2c_sda=-1,i2c_scl=-1,fc_tx=-1,fc_rx=-1;
    std::array<int,8> pins() const { return {spi_sck,spi_mosi,spi_miso,pmw_cs,i2c_sda,i2c_scl,fc_tx,fc_rx}; }
};
struct AppConfig {
    WifiMode wifi=WifiMode::Disabled;
    const char* ssid="";
    const char* password="";
    Peer pi,notebook;
    std::uint16_t pi_local_port=47001,manual_local_port=47002;
    std::uint32_t manual_timeout_us=100000; // bench policy, not qualified for flight
    bool trusted_link=false,allow_bench_auto=false;
    bool enable_sensor_hardware=false,enable_fc_output=false;
    PinPlan pins;
};
inline bool unicast_ipv4(const char* text) {
    if (!text || !*text) return false;
    unsigned fields=0,value=0,digits=0,first=0; bool nonzero=false;
    for (const char* p=text;;++p) {
        if (*p>='0' && *p<='9') {
            if (++digits>3) return false;
            value=value*10+static_cast<unsigned>(*p-'0'); if (value>255) return false;
        } else if (*p=='.' || *p=='\0') {
            if (!digits || fields>=4) return false;
            if (fields==0) first=value;
            nonzero=nonzero || value!=0; ++fields; value=digits=0;
            if (*p=='\0') break;
        } else return false;
    }
    return fields==4 && nonzero && first>0 && first<224 && std::strcmp(text,"255.255.255.255")!=0;
}
inline const char* validate(const AppConfig& c) {
    if (c.manual_timeout_us<1000 || c.manual_timeout_us>100000) return "manual_timeout_bounds";
    const auto pins=c.pins.pins();
    for (std::size_t i=0;i<pins.size();++i) {
        const int pin=pins[i]; if (pin==-1) continue;
        if (pin<0 || pin>48 || (pin>=22 && pin<=25)) return "invalid_pin";
        if (pin==19 || pin==20 || (pin>=26 && pin<=37) || pin==43 || pin==44)
            return "reserved_pin"; // native USB, N16R8 memory and COM UART0
        for (std::size_t j=0;j<i;++j) if (pins[j]==pin) return "pin_conflict";
    }
    if (c.enable_sensor_hardware) return "sensor_drivers_pending";
    if (c.enable_fc_output) return "fc_adapter_pending";
    if (c.allow_bench_auto && !c.trusted_link) return "trusted_link_required";
    if (c.wifi==WifiMode::Disabled) return nullptr;
    if (!c.ssid || !c.password || std::strlen(c.ssid)<1 || std::strlen(c.ssid)>32
            || std::strlen(c.password)<8 || std::strlen(c.password)>63) return "wifi_credentials_required";
    if (!c.pi_local_port || !c.manual_local_port || c.pi_local_port==c.manual_local_port) return "local_port_conflict";
    if (!unicast_ipv4(c.pi.address) || !unicast_ipv4(c.notebook.address) || !c.pi.port || !c.notebook.port) return "peer_required";
    if (c.pi.port==c.notebook.port && std::strcmp(c.pi.address,c.notebook.address)==0) return "peer_conflict";
    return nullptr;
}
inline bool next_epoch(std::uint64_t stored,std::uint64_t& next) {
    if (stored==UINT64_MAX) return false;
    next=stored+1; return next!=0;
}
} // namespace chong::foundation
