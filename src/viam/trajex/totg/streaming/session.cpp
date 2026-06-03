#include <viam/trajex/totg/streaming/session.hpp>

#include <stdexcept>
#include <utility>

namespace viam::trajex::totg::streaming {

session::session(path::options path_options, trajectory::options trajectory_options, types::hertz sample_rate)
    : path_options_(std::move(path_options)), trajectory_options_(std::move(trajectory_options)), sample_rate_(sample_rate) {}

void session::extend(const waypoint_accumulator& /*batch*/) {
    throw std::logic_error("viam::trajex::totg::streaming::session::extend is not yet implemented");
}

trajectory::seconds session::current_time() const noexcept {
    return trajectory::seconds{0.0};
}

std::vector<struct trajectory::sample> session::sample_next(std::size_t /*n*/) {
    throw std::logic_error("viam::trajex::totg::streaming::session::sample_next is not yet implemented");
}

std::vector<struct trajectory::sample> session::sample_at_least(trajectory::seconds /*horizon*/) {
    throw std::logic_error("viam::trajex::totg::streaming::session::sample_at_least is not yet implemented");
}

const trajectory* session::active_trajectory() const noexcept {
    return nullptr;
}

trajectory::seconds session::active_epoch() const noexcept {
    return trajectory::seconds{0.0};
}

std::size_t session::trajectory_generation_count() const noexcept {
    return 0;
}

}  // namespace viam::trajex::totg::streaming
