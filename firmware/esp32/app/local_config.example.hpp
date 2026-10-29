#pragma once
#include "foundation/config.hpp"
// Copy to local_config.hpp (Git-ignored). Default build never joins/creates Wi-Fi.
// Fill static peer addresses/source ports for the actual bench network.
// Do not enable FC or real sensor flags: their adapters are deliberately pending.
inline chong::foundation::AppConfig local_board_config() {
    chong::foundation::AppConfig c;
    // c.wifi=chong::foundation::WifiMode::Station; // or AccessPoint
    // c.ssid="YOUR_BENCH_NETWORK";
    // c.password="YOUR_BENCH_PASSWORD";
    // c.pi={"192.168.1.20",48001};
    // c.notebook={"192.168.1.30",48002};
    // c.trusted_link=true; // only after separately establishing a trusted link
    // c.allow_bench_auto=true; // local console A; still no FC output
    return c;
}
