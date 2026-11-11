// Host-only bounded CLI. No device protocol, GPIO or output driver.
#include "../foundation/runtime.hpp"
#include "../foundation/config.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

using namespace chong;
using namespace chong::foundation;
namespace {
template<class Bytes> std::string hex(const Bytes& bytes) {
    std::ostringstream out; out<<std::hex<<std::setfill('0');
    for (auto b:bytes) out<<std::setw(2)<<static_cast<unsigned>(b);
    return out.str();
}
int digit(char c) { return c>='0' && c<='9' ? c-'0' : c>='a' && c<='f' ? c-'a'+10 : -1; }
template<std::size_t N> bool sample(std::istream& in,SampleSlot<N>& slot,bool normalized=false) {
    Measurement<N> m; unsigned valid=0,quality=0; std::uint64_t now=0;
    in>>m.sample_id>>m.capture_us>>valid>>quality;
    if (valid>1 || quality>255) return false;
    m.valid=valid!=0; m.quality=static_cast<std::uint8_t>(quality);
    for (double& x:m.values) in>>x;
    in>>now; if (!in) return false;
    slot.update(m,now,normalized); return true;
}
}
int main(int argc,char** argv) {
    if (argc!=3) return 2;
    Runtime runtime(std::stoull(argv[1]),std::string(argv[2])=="1");
    std::array<char,2200> line{};
    while (std::cin.getline(line.data(),static_cast<std::streamsize>(line.size()))) {
        std::istringstream in(line.data()); std::string op; in>>op;
        std::uint64_t now=0; std::string sensor_hex,challenge_hex,receipt_hex;
        if (op=="config") {
            std::string profile; in>>profile;
            AppConfig c;
            if (profile=="wifi_missing") c.wifi=WifiMode::Station;
            if (profile=="same_peer" || profile=="same_port") {
                c.wifi=WifiMode::Station; c.ssid="bench"; c.password="bench-only-key";
                c.pi={"127.0.0.1",48001}; c.notebook={"127.0.0.1",48002};
                if (profile=="same_peer") c.notebook=c.pi;
                else c.manual_local_port=c.pi_local_port;
            }
            if (profile=="sensor") c.enable_sensor_hardware=true;
            if (profile=="fc") c.enable_fc_output=true;
            if (profile=="reserved") c.pins.i2c_sda=35;
            if (profile=="pin_conflict") { c.pins.i2c_sda=4; c.pins.i2c_scl=4; }
            if (profile=="untrusted_auto") c.allow_bench_auto=true;
            if (profile=="manual_timeout") c.manual_timeout_us=100001;
            const auto error=validate(c);
            std::cout<<"{\"config_error\":";
            if (error) std::cout<<'"'<<error<<'"'; else std::cout<<"null";
            std::cout<<"}\n"; continue;
        } else if (op=="reset") { in>>now; runtime.reset(now); }
        else if (op=="poll") { in>>now; runtime.poll(now); }
        else if (op=="disconnect") { in>>now; runtime.disconnect(now); }
        else if (op=="mode") {
            std::string mode; in>>mode>>now;
            if (mode!="MANUAL" && mode!="SHADOW" && mode!="AUTO") return 2;
            runtime.mode(mode=="AUTO" ? Mode::Auto : mode=="SHADOW" ? Mode::Shadow : Mode::Manual,now);
        } else if (op=="frame") {
            std::uint32_t seq=0; in>>seq>>now; Frame frame;
            if (runtime.frame(seq,now,frame)) {
                std::array<std::uint8_t,733> sensor{}; std::array<std::uint8_t,40> challenge{};
                if (encode_sensor(frame,sensor)) sensor_hex=hex(sensor);
                encode_challenge(frame,challenge); challenge_hex=hex(challenge);
            }
        } else if (op=="manual" || op=="pi") {
            std::string text; in>>text>>now; std::array<std::uint8_t,1024> bytes{};
            bool ok=text.size()%2==0 && text.size()/2<=bytes.size();
            for (std::size_t i=0;ok && i<text.size()/2;++i) {
                const int a=digit(text[2*i]),b=digit(text[2*i+1]);
                if (a<0 || b<0) ok=false; else bytes[i]=static_cast<std::uint8_t>(16*a+b);
            }
            const auto count=ok ? text.size()/2 : 0;
            if (op=="manual") runtime.receive_manual(bytes.data(),count,now);
            else {
                Receipt receipt; std::array<std::uint8_t,130> encoded{};
                if (runtime.receive_pi(bytes.data(),count,now,receipt) && encode_receipt(receipt,encoded)) receipt_hex=hex(encoded);
            }
        } else if (op=="sample") {
            std::string channel; in>>channel; bool ok=false;
            if (channel=="flow") ok=sample(in,runtime.sensors.flow);
            else if (channel=="tof") ok=sample(in,runtime.sensors.tof,true);
            else if (channel=="altitude") ok=sample(in,runtime.sensors.altitude);
            else if (channel=="attitude") ok=sample(in,runtime.sensors.attitude);
            if (!ok) return 2;
        } else return 2;
        if (!in) return 2;
        const auto selection=runtime.selection();
        std::cout<<"{\"mode\":\""<<(runtime.receiver.mode==Mode::Manual ? "MANUAL" : runtime.receiver.mode==Mode::Shadow ? "SHADOW" : "AUTO")
            <<"\",\"source\":\""<<(selection.source==Source::None ? "none" : selection.source==Source::Manual ? "manual" : "auto")
            <<"\",\"auto_tripped\":"<<((runtime.receiver.tripped || runtime.platform_fault()) ? "true" : "false")
            <<",\"manual_tripped\":"<<(runtime.manual.tripped ? "true" : "false")
            <<",\"reason\":\""<<runtime.receiver.event_reason<<"\",\"manual_reason\":\""<<runtime.manual.reason
            <<"\",\"session_id\":\""<<hex(runtime.receiver.session())<<"\",\"selected_pwm\":";
        if (selection.source==Source::None) std::cout<<"null";
        else {
            std::cout<<'[';
            for (unsigned i=0;i<4;++i) std::cout<<(i ? "," : "")<<std::setprecision(17)<<selection.pwm[i];
            std::cout<<']';
        }
        std::cout<<",\"fc_stage\":\"not_attempted\",\"sensor_hex\":\""<<sensor_hex
            <<"\",\"challenge_hex\":\""<<challenge_hex<<"\",\"receipt_hex\":\""<<receipt_hex<<"\"}\n";
    }
    return std::cin.eof() ? 0 : 2;
}
