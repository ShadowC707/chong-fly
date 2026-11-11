#pragma once
#include "runtime.hpp"

namespace chong::foundation {
// A real backend owns bus state and publishes normalized values with acquisition
// identity/timestamps. Poll must not wait for a conversion; use a state machine.
class SensorBackend {
public:
    virtual ~SensorBackend()=default;
    virtual const char* status() const=0;
    virtual void poll(SensorCache& cache,std::uint64_t now)=0;
};
class PendingSensors final : public SensorBackend {
public:
    const char* status() const override { return "drivers_pending_all_invalid"; }
    void poll(SensorCache&,std::uint64_t) override {} // no fake clear space/zero attitude
};
enum class FcStage { NotAttempted };
class DisabledFc final {
public:
    const char* status() const { return "disabled_protocol_and_revocation_pending"; }
    FcStage observe(const Selection&,std::uint64_t) const { return FcStage::NotAttempted; }
};
} // namespace chong::foundation
