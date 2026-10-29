// Bench USB input diagnostic. No receiver authority, FC UART, motors or Wi-Fi.
#include <array>
#include <cinttypes>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "usb/usb_host.h"
#include "usb_probe_core.hpp"

namespace {
constexpr const char* TAG="chong_usb";
struct Probe {
    usb_host_client_handle_t client=nullptr;
    usb_device_handle_t device=nullptr;
    usb_transfer_t* transfer=nullptr;
    chong::usb_probe::Endpoint endpoint{};
    std::uint8_t pending_address=0;
    bool gone=false, flushing=false, claimed=false, in_flight=false, packet_ready=false;
    std::array<std::uint8_t,64> packet{};
    std::size_t packet_length=0;
    std::uint64_t reports=0, overwritten_logs=0;
    std::int64_t received_us=0;
    int transfer_error=0;
} probe;

void library_task(void*) {
    for (;;) {
        std::uint32_t flags=0;
        const auto err=usb_host_lib_handle_events(portMAX_DELAY,&flags);
        if (err!=ESP_OK) ESP_LOGE(TAG,"USB library error: %s",esp_err_to_name(err));
        if (flags&USB_HOST_LIB_EVENT_FLAGS_NO_CLIENTS) usb_host_device_free_all();
    }
}

void on_device(const usb_host_client_event_msg_t* event,void*) {
    if (event->event==USB_HOST_CLIENT_EVENT_NEW_DEV) {
        if (!probe.device && !probe.pending_address) probe.pending_address=event->new_dev.address;
    } else if (event->event==USB_HOST_CLIENT_EVENT_DEV_GONE && probe.device==event->dev_gone.dev_hdl) {
        probe.gone=true;
    }
}

void on_input(usb_transfer_t* transfer) {
    probe.in_flight=false;
    if (probe.gone) return;
    if (transfer->status!=USB_TRANSFER_STATUS_COMPLETED) {
        probe.transfer_error=static_cast<int>(transfer->status)+1;
        return;
    }
    const int count=transfer->actual_num_bytes;
    if (count>0 && count<=static_cast<int>(probe.packet.size())) {
        if (probe.packet_ready) ++probe.overwritten_logs;
        std::memcpy(probe.packet.data(),transfer->data_buffer,static_cast<std::size_t>(count));
        probe.packet_length=static_cast<std::size_t>(count);
        probe.packet_ready=true;
        probe.received_us=esp_timer_get_time();
        ++probe.reports;
    }
    const auto err=usb_host_transfer_submit(transfer);
    probe.in_flight=err==ESP_OK;
    if (err!=ESP_OK) probe.transfer_error=-static_cast<int>(err);
}

void hex_line(const char* label,const std::uint8_t* bytes,std::size_t n,std::size_t offset=0) {
    char text[129]{};
    if (n>64) n=64;
    for (std::size_t i=0;i<n;++i) std::snprintf(text+2*i,3,"%02x",bytes[i]);
    ESP_LOGI(TAG,"%s offset=%u hex=%s",label,static_cast<unsigned>(offset),text);
}

void open_device() {
    const auto address=probe.pending_address;
    probe.pending_address=0;
    const auto err=usb_host_device_open(probe.client,address,&probe.device);
    if (err!=ESP_OK) { ESP_LOGE(TAG,"Open failed: %s",esp_err_to_name(err)); return; }
    probe.gone=probe.flushing=false;
    const usb_device_desc_t* desc=nullptr;
    if (usb_host_get_device_descriptor(probe.device,&desc)!=ESP_OK) return;
    ESP_LOGI(TAG,"DEVICE vid=%04x pid=%04x revision=%04x",desc->idVendor,desc->idProduct,desc->bcdDevice);
    hex_line("DEVICE_DESCRIPTOR",reinterpret_cast<const std::uint8_t*>(desc),desc->bLength);
    const usb_config_desc_t* config=nullptr;
    if (usb_host_get_active_config_descriptor(probe.device,&config)!=ESP_OK) return;
    const auto total=config->wTotalLength;
    if (total<9 || total>4096) { ESP_LOGE(TAG,"Configuration outside diagnostic bound"); return; }
    const auto* bytes=reinterpret_cast<const std::uint8_t*>(config);
    for (std::size_t pos=0;pos<total;pos+=32) {
        const auto left=total-pos;
        hex_line("CONFIG",bytes+pos,left<32 ? left : 32,pos);
    }
    if (desc->idVendor!=0x045e || desc->idProduct!=0x028e) {
        ESP_LOGW(TAG,"Device logged only: this build reads input only for 045e:028e");
        return;
    }
    probe.endpoint=chong::usb_probe::select(bytes,total);
    if (!probe.endpoint.valid || !probe.endpoint.found) {
        ESP_LOGW(TAG,"No unambiguous FF/5D/01 interrupt-IN interface; descriptor captured only");
        return;
    }
    auto e=probe.endpoint;
    auto code=usb_host_interface_claim(probe.client,probe.device,e.interface,e.alternate);
    if (code!=ESP_OK) { ESP_LOGE(TAG,"Claim failed: %s",esp_err_to_name(code)); return; }
    probe.claimed=true;
    code=usb_host_transfer_alloc(e.packet_size,0,&probe.transfer);
    if (code!=ESP_OK) { ESP_LOGE(TAG,"Transfer allocation failed"); return; }
    probe.transfer->device_handle=probe.device;
    probe.transfer->bEndpointAddress=e.address;
    probe.transfer->callback=on_input;
    probe.transfer->num_bytes=e.packet_size;
    code=usb_host_transfer_submit(probe.transfer);
    probe.in_flight=code==ESP_OK;
    ESP_LOGI(TAG,"INPUT_SELECTED interface=%u endpoint=%02x max_packet=%u submit=%s",
        e.interface,e.address,e.packet_size,esp_err_to_name(code));
}

void close_gone_device() {
    if (!probe.flushing) {
        probe.flushing=true;
        probe.packet_ready=false;
        ESP_LOGW(TAG,"DEVICE_GONE: input invalid; RF freshness was never assumed");
        if (probe.claimed) {
            usb_host_endpoint_halt(probe.device,probe.endpoint.address);
            usb_host_endpoint_flush(probe.device,probe.endpoint.address);
        }
    }
    // The cancellation callback is delivered by client_handle_events. Never
    // free a transfer while the host library still owns it.
    if (probe.in_flight) return;
    if (probe.transfer) { usb_host_transfer_free(probe.transfer); probe.transfer=nullptr; }
    if (probe.claimed) {
        usb_host_interface_release(probe.client,probe.device,probe.endpoint.interface);
        probe.claimed=false;
    }
    usb_host_device_close(probe.client,probe.device);
    probe.device=nullptr;
    probe.endpoint={}; probe.gone=probe.flushing=false;
    probe.transfer_error=0;
}
} // namespace

extern "C" void app_main() {
    ESP_LOGI(TAG,"CHONG_USB_PROBE_V1 -- diagnostic only, FC output absent, RF health unknown");
    ESP_LOGI(TAG,"UART0/COM console; native USB GPIO19/20. VBUS power is not switched by this firmware.");
    usb_host_config_t host{};
    host.intr_flags=ESP_INTR_FLAG_LEVEL1;
    ESP_ERROR_CHECK(usb_host_install(&host));
    if (xTaskCreate(library_task,"usb_library",4096,nullptr,2,nullptr)!=pdPASS) abort();
    usb_host_client_config_t client{};
    client.max_num_event_msg=5;
    client.async.client_event_callback=on_device;
    ESP_ERROR_CHECK(usb_host_client_register(&client,&probe.client));
    ESP_LOGI(TAG,"USB_HOST_READY; waiting for receiver. No USB OUT gamepad commands are sent.");
    std::int64_t last_log=0,last_heartbeat=0;
    for (;;) {
        const auto err=usb_host_client_handle_events(probe.client,pdMS_TO_TICKS(20));
        if (err!=ESP_OK && err!=ESP_ERR_TIMEOUT) ESP_LOGE(TAG,"Client events: %s",esp_err_to_name(err));
        if (probe.gone) close_gone_device();
        if (!probe.device && probe.pending_address) open_device();
        const auto now=esp_timer_get_time();
        if (probe.packet_ready && now-last_log>=100000) {
            // Only the diagnostic log is latest-only and limited to 10 Hz.
            // This is not the command transport's queue policy or a control rate.
            probe.packet_ready=false; last_log=now;
            hex_line("INPUT_RAW",probe.packet.data(),probe.packet_length);
            chong::usb_probe::Input input;
            if (chong::usb_probe::decode(probe.packet.data(),probe.packet_length,input)) {
                ESP_LOGI(TAG,"CANDIDATE_XUSB usb_received_us=%" PRId64 " buttons=%04x lt=%u rt=%u lx=%" PRId32
                    " ly=%" PRId32 " rx=%" PRId32 " ry=%" PRId32,
                    probe.received_us,input.buttons,input.lt,input.rt,input.lx,input.ly,input.rx,input.ry);
            } else ESP_LOGW(TAG,"UNDECODED_INPUT length=%u",static_cast<unsigned>(probe.packet_length));
        }
        if (probe.transfer_error) {
            ESP_LOGE(TAG,"INPUT_TRANSFER_STOPPED code=%d; input unusable",probe.transfer_error);
            probe.transfer_error=0;
        }
        if (now-last_heartbeat>=2000000) {
            last_heartbeat=now;
            ESP_LOGI(TAG,"HEARTBEAT device=%u reports=%" PRIu64 " log_overwrites=%" PRIu64 " rf_health=unknown fc_output=disabled",
                probe.device ? 1u : 0u,probe.reports,probe.overwritten_logs);
        }
    }
}
