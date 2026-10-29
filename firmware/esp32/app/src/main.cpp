// Board foundation. FC output is absent; default Wi-Fi and sensor hardware off.
#include <array>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <optional>
#include <type_traits>
#include <unistd.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/queue.h"
#include "driver/uart.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_random.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "nvs.h"
#include "nvs_flash.h"
#include "lwip/sockets.h"
#include "foundation/config.hpp"
#include "foundation/adapters.hpp"

#if __has_include("../local_config.hpp")
#include "../local_config.hpp"
#else
inline chong::foundation::AppConfig local_board_config() { return {}; }
#endif

using namespace chong;
using namespace chong::foundation;
namespace {
constexpr const char* TAG="chong_app";
constexpr std::uint64_t FRAME_US=FRAME_PERIOD_US;
constexpr unsigned RX_BUDGET=4;
AppConfig config;
std::optional<Runtime> state;
PendingSensors sensor_backend;
DisabledFc fc;
StaticSemaphore_t mutex_memory;
SemaphoreHandle_t state_mutex=nullptr;
StaticQueue_t sensor_queue_memory;
std::array<std::uint8_t,sizeof(SensorCache)> sensor_queue_bytes{};
QueueHandle_t sensor_queue=nullptr;
std::atomic<bool> network_ready{false},contention_fault{false};
std::atomic<std::uint32_t> link_generation{0},foreign_rx{0},oversize_rx{0},untrusted_rx{0};
std::atomic<std::uint32_t> socket_errors{0},tx_errors{0},missed_frames{0},lock_timeouts{0};
std::atomic<std::uint32_t> rx_budget_limits{0};

std::uint64_t now_us() { return static_cast<std::uint64_t>(esp_timer_get_time()); }
class Guard {
    bool locked_;
public:
    Guard():locked_(xSemaphoreTake(state_mutex,pdMS_TO_TICKS(2))==pdTRUE) {
        if (!locked_) { ++lock_timeouts; contention_fault=true; }
    }
    ~Guard() { if (locked_) xSemaphoreGive(state_mutex); }
    explicit operator bool() const { return locked_; }
};

// Persist and read back a boot epoch before allocating any live session.
// Never erase an existing NVS partition on an initialization error.
std::uint64_t boot_epoch() {
    if (nvs_flash_init()!=ESP_OK) { ESP_LOGE(TAG,"NVS unavailable; sessions disabled, no erase performed"); return 0; }
    nvs_handle_t handle;
    if (nvs_open("chong_hw",NVS_READWRITE,&handle)!=ESP_OK) return 0;
    std::uint64_t stored=0,next=0,check=0;
    auto result=nvs_get_u64(handle,"epoch",&stored);
    if (result==ESP_ERR_NVS_NOT_FOUND) {
        esp_fill_random(&stored,sizeof(stored));
        stored&=UINT64_MAX>>1; // a fresh initial namespace is not a fixed epoch=1
    } else if (result!=ESP_OK) { nvs_close(handle); return 0; }
    const bool success=next_epoch(stored,next) && nvs_set_u64(handle,"epoch",next)==ESP_OK
        && nvs_commit(handle)==ESP_OK && nvs_get_u64(handle,"epoch",&check)==ESP_OK && check==next;
    nvs_close(handle); return success ? next : 0;
}

void on_network(void*,esp_event_base_t base,std::int32_t id,void*) {
    if (base==WIFI_EVENT) {
        if (id==WIFI_EVENT_STA_START) esp_wifi_connect();
        if (id==WIFI_EVENT_STA_DISCONNECTED) {
            network_ready=false; ++link_generation;
            esp_wifi_connect(); // reconnect does not reset/grant command authority
        }
        if (id==WIFI_EVENT_AP_START) network_ready=true;
        if (id==WIFI_EVENT_AP_STOP) { network_ready=false; ++link_generation; }
        if (id==WIFI_EVENT_AP_STADISCONNECTED) ++link_generation; // conservative revoke of both roles
    }
    if (base==IP_EVENT && id==IP_EVENT_STA_GOT_IP) network_ready=true;
    if (base==IP_EVENT && id==IP_EVENT_STA_LOST_IP) { network_ready=false; ++link_generation; }
}

bool start_wifi() {
    if (config.wifi==WifiMode::Disabled) { ESP_LOGI(TAG,"Wi-Fi disabled: fill app/local_config.hpp for a bench network"); return false; }
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    if (config.wifi==WifiMode::Station) esp_netif_create_default_wifi_sta();
    else esp_netif_create_default_wifi_ap();
    wifi_init_config_t init=WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM)); // credentials not persisted by Wi-Fi
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT,ESP_EVENT_ANY_ID,on_network,nullptr));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT,ESP_EVENT_ANY_ID,on_network,nullptr));
    wifi_config_t wifi{};
    if (config.wifi==WifiMode::Station) {
        std::memcpy(wifi.sta.ssid,config.ssid,std::strlen(config.ssid));
        std::memcpy(wifi.sta.password,config.password,std::strlen(config.password));
        wifi.sta.threshold.authmode=WIFI_AUTH_WPA2_PSK;
        ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
        ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA,&wifi));
    } else {
        std::memcpy(wifi.ap.ssid,config.ssid,std::strlen(config.ssid));
        std::memcpy(wifi.ap.password,config.password,std::strlen(config.password));
        wifi.ap.ssid_len=static_cast<std::uint8_t>(std::strlen(config.ssid));
        wifi.ap.authmode=WIFI_AUTH_WPA2_PSK; wifi.ap.max_connection=2;
        ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
        ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP,&wifi));
    }
    ESP_ERROR_CHECK(esp_wifi_start());
    if (config.wifi==WifiMode::Station) ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_LOGI(TAG,"Wi-Fi started; fixed peers configured; trusted_link=%u. Credentials omitted.",config.trusted_link ? 1u : 0u);
    return true;
}

struct Channel {
    int fd=-1;
    sockaddr_in peer{};
    bool open(std::uint16_t local_port,const Peer& remote) {
        fd=socket(AF_INET,SOCK_DGRAM,IPPROTO_UDP);
        if (fd<0) return false;
        const int receive_bytes=4096;
        const int flags=fcntl(fd,F_GETFL,0);
        if (flags<0 || fcntl(fd,F_SETFL,flags|O_NONBLOCK)<0
                || setsockopt(fd,SOL_SOCKET,SO_RCVBUF,&receive_bytes,sizeof(receive_bytes))<0) {
            close(fd); fd=-1; return false;
        }
        sockaddr_in local{}; local.sin_family=AF_INET; local.sin_port=htons(local_port);
        local.sin_addr.s_addr=htonl(INADDR_ANY);
        if (bind(fd,reinterpret_cast<const sockaddr*>(&local),sizeof(local))<0) { close(fd); fd=-1; return false; }
        peer.sin_family=AF_INET; peer.sin_port=htons(remote.port);
        peer.sin_addr.s_addr=inet_addr(remote.address); return true;
    }
    void send(const std::uint8_t* p,std::size_t n) const {
        if (fd<0 || sendto(fd,p,n,MSG_DONTWAIT,reinterpret_cast<const sockaddr*>(&peer),sizeof(peer))!=static_cast<ssize_t>(n)) ++tx_errors;
    }
};

void drain(Channel& channel,bool manual_channel) {
    std::array<std::uint8_t,1025> bytes{};
    unsigned examined=0;
    for (unsigned budget=0;budget<RX_BUDGET;++budget) {
        sockaddr_in sender{}; socklen_t length=sizeof(sender);
        const auto count=recvfrom(channel.fd,bytes.data(),bytes.size(),MSG_DONTWAIT,
            reinterpret_cast<sockaddr*>(&sender),&length);
        if (count<0) {
            if (errno!=EWOULDBLOCK && errno!=EAGAIN) ++socket_errors;
            break;
        }
        ++examined;
        if (sender.sin_addr.s_addr!=channel.peer.sin_addr.s_addr || sender.sin_port!=channel.peer.sin_port) {
            ++foreign_rx; continue;
        }
        if (count>1024) { ++oversize_rx; continue; }
        if (!config.trusted_link) { ++untrusted_rx; continue; }
        Receipt receipt; bool has_receipt=false;
        {
            Guard guard; if (!guard) continue;
            // Timestamp after taking the mutex: waiting must not make input younger.
            const auto now=now_us();
            if (manual_channel) state->receive_manual(bytes.data(),static_cast<std::size_t>(count),now);
            else has_receipt=state->receive_pi(bytes.data(),static_cast<std::size_t>(count),now,receipt);
        }
        if (has_receipt) {
            std::array<std::uint8_t,130> out{};
            if (encode_receipt(receipt,out)) channel.send(out.data(),out.size());
        }
    }
    if (examined==RX_BUDGET) ++rx_budget_limits; // scan limit used, not proof of kernel loss
}

void safety_task(void*) {
    std::uint32_t seen_generation=link_generation.load();
    TickType_t tick=xTaskGetTickCount();
    for (;;) {
        {
            Guard guard;
            if (guard) {
                const auto now=now_us(); const auto current=link_generation.load();
                if (current!=seen_generation || contention_fault.exchange(false)) state->disconnect(now);
                seen_generation=current; state->poll(now);
                SensorCache updated;
                if (xQueueReceive(sensor_queue,&updated,0)==pdTRUE) state->sensors=updated;
                fc.observe(state->selection(),now); // always NotAttempted; no FC UART driver exists
            }
        }
        vTaskDelayUntil(&tick,pdMS_TO_TICKS(1));
    }
}

void sensor_task(void*) {
    SensorCache cache;
    TickType_t tick=xTaskGetTickCount();
    for (;;) {
        sensor_backend.poll(cache,now_us());
        // Sensor snapshots are latest-only; capture identities stay unchanged.
        // This is not a command queue, and no command is coalesced/retried.
        xQueueOverwrite(sensor_queue,&cache);
        vTaskDelayUntil(&tick,pdMS_TO_TICKS(5));
    }
}

void io_task(void*) {
    Channel pi,notebook;
    if (!pi.open(config.pi_local_port,config.pi) || !notebook.open(config.manual_local_port,config.notebook)) {
        ++socket_errors; ++link_generation; ESP_LOGE(TAG,"UDP setup failed; authority revoked, no retry queue");
        if (pi.fd>=0) close(pi.fd);
        if (notebook.fd>=0) close(notebook.fd);
        vTaskDelete(nullptr); return;
    }
    ESP_LOGI(TAG,"UDP ready: Pi local=%u, notebook local=%u; scan budget=%u per channel",config.pi_local_port,config.manual_local_port,RX_BUDGET);
    Session seen{}; std::uint32_t sequence=0;
    std::uint64_t next_frame=now_us(); bool exhausted=false;
    for (;;) {
        if (network_ready) {
            const auto now=now_us();
            if (now>=next_frame) {
                const auto skipped=(now-next_frame)/FRAME_US;
                missed_frames.fetch_add(static_cast<std::uint32_t>(skipped));
                next_frame+=(skipped+1)*FRAME_US; // no burst of catch-up frames
                Frame frame; bool available=false;
                {
                    Guard guard;
                    if (guard) {
                        if (state->receiver.session()!=seen) { seen=state->receiver.session(); sequence=0; exhausted=false; }
                        if (!exhausted) available=state->frame(sequence,now_us(),frame);
                        if (sequence==UINT32_MAX) exhausted=true;
                        else ++sequence;
                    }
                }
                // Announce before draining commands. Encoding/socket work is
                // outside the state mutex and included in the local frame age.
                if (available) {
                    std::array<std::uint8_t,733> sensor{}; std::array<std::uint8_t,40> challenge{};
                    if (encode_sensor(frame,sensor)) pi.send(sensor.data(),sensor.size());
                    encode_challenge(frame,challenge); notebook.send(challenge.data(),challenge.size());
                }
            }
            drain(notebook,true); drain(pi,false);
        }
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

void console_task(void*) {
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM_0,256,0,0,nullptr,0));
    ESP_LOGI(TAG,"Local console: M=MANUAL S=SHADOW A=bench AUTO (config-gated) R=reset only in MANUAL");
    std::uint64_t last_log=0;
    for (;;) {
        std::array<std::uint8_t,16> input{};
        const int count=uart_read_bytes(UART_NUM_0,input.data(),input.size(),0);
        {
            Guard guard;
            if (guard) {
                for (int i=0;i<count;++i) {
                    const auto now=now_us();
                    if (input[i]=='M') state->mode(Mode::Manual,now);
                    if (input[i]=='S') state->mode(Mode::Shadow,now);
                    if (input[i]=='A') state->mode(Mode::Auto,now);
                    if (input[i]=='R') state->reset(now);
                }
            }
        }
        const auto now=now_us();
        if (now-last_log>=2000000) {
            last_log=now; unsigned mode=0,source=0,manual_fault=0,auto_fault=0;
            const char* receiver_reason="unavailable";
            const char* manual_reason="unavailable";
            const char* local_reason="unavailable";
            {
                Guard guard;
                if (guard) {
                    state->poll(now_us()); mode=static_cast<unsigned>(state->receiver.mode);
                    source=static_cast<unsigned>(state->selection().source);
                    manual_fault=state->manual.tripped; auto_fault=state->receiver.tripped || state->platform_fault();
                    receiver_reason=state->receiver.reason; manual_reason=state->manual.reason;
                    local_reason=state->local_reason;
                }
            }
            ESP_LOGI(TAG,"STATUS mode=%u source=%u manual_fault=%u auto_fault=%u wifi=%u fc=not_attempted foreign=%" PRIu32
                " oversize=%" PRIu32 " untrusted=%" PRIu32 " socket_err=%" PRIu32 " tx_err=%" PRIu32
                " missed_frames=%" PRIu32 " lock_timeouts=%" PRIu32 " rx_scan_limits=%" PRIu32 " reasons=%s/%s/%s",
                mode,source,manual_fault,auto_fault,network_ready ? 1u : 0u,foreign_rx.load(),oversize_rx.load(),
                untrusted_rx.load(),socket_errors.load(),tx_errors.load(),missed_frames.load(),lock_timeouts.load(),
                rx_budget_limits.load(),receiver_reason,manual_reason,local_reason);
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
bool launch(TaskFunction_t task,const char* name,std::uint32_t stack,UBaseType_t priority) {
    return xTaskCreate(task,name,stack,nullptr,priority,nullptr)==pdPASS;
}
} // namespace

extern "C" void app_main() {
    static_assert(std::is_trivially_copyable<SensorCache>::value,"Queue snapshots must be trivially copyable");
    ESP_LOGI(TAG,"CHONG_FOUNDATION_V1; Pi=model, notebook=manual; FC OUTPUT ABSENT");
    config=local_board_config();
    if (const auto error=validate(config)) {
        ESP_LOGE(TAG,"Configuration rejected: %s. No network/control tasks started.",error); return;
    }
    state_mutex=xSemaphoreCreateMutexStatic(&mutex_memory);
    sensor_queue=xQueueCreateStatic(1,sizeof(SensorCache),sensor_queue_bytes.data(),&sensor_queue_memory);
    if (!state_mutex || !sensor_queue) { ESP_LOGE(TAG,"Static state allocation failed"); return; }
    const auto epoch=boot_epoch();
    state.emplace(epoch,config.allow_bench_auto && config.trusted_link,config.manual_timeout_us);
    if (!state->reset(now_us())) ESP_LOGE(TAG,"No session allocated; local reset/storage repair required");
    ESP_LOGI(TAG,"Sensors=%s; FC=%s; period=20ms manual_timeout=%" PRIu32 "us (bench only)",
        sensor_backend.status(),fc.status(),config.manual_timeout_us);
    if (!launch(safety_task,"chong_safety",6144,10) || !launch(sensor_task,"chong_sensors",6144,4)
            || !launch(console_task,"chong_console",6144,3)) {
        ESP_LOGE(TAG,"Task creation failed; network remains disabled"); return;
    }
    if (epoch && start_wifi() && !launch(io_task,"chong_udp",10240,5)) {
        network_ready=false; ++link_generation; ESP_LOGE(TAG,"UDP task creation failed");
    }
}
