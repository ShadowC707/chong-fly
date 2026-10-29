// Bounded line-oriented harness for host tests only; not a device protocol.
#include "../receiver_core.hpp"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {
const char* mode_name(chong::Mode m) {
    return m==chong::Mode::Manual ? "MANUAL" : m==chong::Mode::Shadow ? "SHADOW" : "AUTO";
}
void pwm(const chong::Pwm& p,bool present) {
    if (!present) { std::cout<<"null"; return; }
    std::cout<<'[';
    for (unsigned i=0;i<4;++i) std::cout<<(i ? "," : "")<<std::setprecision(17)<<p[i];
    std::cout<<']';
}
void report(const chong::Receiver& r) {
    std::cout<<"{\"reason\":\""<<r.event_reason<<"\",\"tripped\":"<<(r.tripped ? "true" : "false")
             <<",\"mode\":\""<<mode_name(r.mode)<<"\",\"accepted_pwm\":";
    pwm(r.pwm,r.accepted);
    std::cout<<",\"selected_pwm\":"; pwm(r.pwm,r.active);
    std::cout<<",\"selected_session_id\":";
    if (r.active) {
        std::cout<<'"'<<std::hex<<std::setfill('0');
        for (auto byte:r.session()) std::cout<<std::setw(2)<<static_cast<unsigned>(byte);
        std::cout<<'"'<<std::dec<<std::setfill(' ');
    } else std::cout<<"null";
    std::cout<<",\"selected_sequence\":";
    if (r.active) std::cout<<r.active_sequence;
    else std::cout<<"null";
    std::cout<<"}\n";
}
int digit(char c) {
    if (c>='0' && c<='9') return c-'0';
    if (c>='a' && c<='f') return c-'a'+10;
    return -1;
}
}
int main(int argc,char** argv) {
    if (argc!=3) return 2;
    chong::Receiver receiver(std::stoull(argv[1]),std::stoull(argv[2]));
    // Input line and decoded frame have fixed upper bounds; no event history.
    std::array<char,2200> line{};
    while (std::cin.getline(line.data(),static_cast<std::streamsize>(line.size()))) {
        std::istringstream input(line.data());
        std::string op; input>>op;
        std::uint64_t now=0;
        if (op=="reset") { input>>now; receiver.reset(now); }
        else if (op=="poll") { input>>now; receiver.poll(now); }
        else if (op=="mode") {
            std::string mode; input>>mode>>now;
            if (mode!="MANUAL" && mode!="SHADOW" && mode!="AUTO") return 2;
            receiver.set_mode(mode=="MANUAL" ? chong::Mode::Manual :
                mode=="SHADOW" ? chong::Mode::Shadow : chong::Mode::Auto,now);
        } else if (op=="sensor") {
            std::uint64_t sequence=0; input>>sequence>>now;
            if (sequence>UINT32_MAX) return 2;
            receiver.announce(static_cast<std::uint32_t>(sequence),now);
        } else if (op=="receive") {
            std::string hex; input>>hex>>now;
            std::array<std::uint8_t,1024> bytes{};
            bool valid=hex.size()%2==0 && hex.size()<=bytes.size()*2;
            for (std::size_t i=0;valid && i<hex.size()/2;++i) {
                int a=digit(hex[i*2]),b=digit(hex[i*2+1]);
                if (a<0 || b<0) valid=false;
                else bytes[i]=static_cast<std::uint8_t>(16*a+b);
            }
            receiver.receive(bytes.data(),valid ? hex.size()/2 : 0,now);
        } else return 2;
        if (!input) return 2;
        report(receiver);
    }
    return std::cin.eof() ? 0 : 2;
}
