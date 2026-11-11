#include "../usb_probe_core.hpp"
#include <array>
#include <iostream>
#include <string>
int digit(char c) {
    if (c>='0' && c<='9') return c-'0';
    if (c>='a' && c<='f') return c-'a'+10;
    return -1;
}
int main(int argc,char** argv) {
    if (argc!=3) return 2;
    std::string kind=argv[1], hex=argv[2];
    std::array<std::uint8_t,4096> data{};
    if (hex.size()%2 || hex.size()>data.size()*2) return 2;
    for (std::size_t i=0;i<hex.size()/2;++i) {
        int a=digit(hex[2*i]),b=digit(hex[2*i+1]);
        if (a<0 || b<0) return 2;
        data[i]=static_cast<std::uint8_t>(a*16+b);
    }
    if (kind=="config") {
        auto e=chong::usb_probe::select(data.data(),hex.size()/2);
        std::cout<<"{\"valid\":"<<(e.valid ? "true" : "false")<<",\"found\":"<<(e.found ? "true" : "false");
        if (e.found) std::cout<<",\"interface\":"<<unsigned(e.interface)<<",\"alternate\":"<<unsigned(e.alternate)
            <<",\"endpoint\":"<<unsigned(e.address)<<",\"packet_size\":"<<e.packet_size;
        std::cout<<"}\n";
    } else if (kind=="report") {
        chong::usb_probe::Input i;
        if (!chong::usb_probe::decode(data.data(),hex.size()/2,i)) { std::cout<<"{\"valid\":false}\n"; return 0; }
        std::cout<<"{\"valid\":true,\"buttons\":"<<i.buttons<<",\"lt\":"<<unsigned(i.lt)<<",\"rt\":"<<unsigned(i.rt)
            <<",\"lx\":"<<i.lx<<",\"ly\":"<<i.ly<<",\"rx\":"<<i.rx<<",\"ry\":"<<i.ry<<"}\n";
    } else return 2;
}
