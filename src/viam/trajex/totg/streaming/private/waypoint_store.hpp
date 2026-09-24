#pragma once

// Persistent waypoint storage for a streaming session.
//
// A session outlives the batches handed to it, so it cannot retain the caller's
// `waypoint_accumulator`: that accumulator holds row-views into memory the caller owns and
// may reuse or destroy as soon as `extend` returns. The store keeps its own copy and
// presents it back as an accumulator, which is the form `path::create` consumes.
//
// This is a private header: header-only, no library backing, and not installed. It exists
// so the session and the pipeline benchmarks share one definition of what accumulating
// waypoints across a session costs, which lets a change to the storage strategy be
// measured rather than guessed at.

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#else
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#endif

#include <viam/trajex/totg/streaming/private/session_utils.hpp>
#include <viam/trajex/totg/waypoint_accumulator.hpp>

namespace viam::trajex::totg::streaming {

class waypoint_store {
   public:
    waypoint_store() = default;

    // Neither copyable nor movable, because the accumulator returned by `waypoints()` holds
    // views referencing `storage_` by address. Moving the store would move `storage_` and
    // leave those views dangling while still appearing valid.
    waypoint_store(const waypoint_store&) = delete;
    waypoint_store& operator=(const waypoint_store&) = delete;
    waypoint_store(waypoint_store&&) = delete;
    waypoint_store& operator=(waypoint_store&&) = delete;

    std::size_t size() const noexcept {
        return empty() ? 0 : storage_.shape(0);
    }

    std::size_t dof() const noexcept {
        return empty() ? 0 : storage_.shape(1);
    }

    bool empty() const noexcept {
        return storage_.dimension() != 2 || storage_.shape(0) == 0;
    }

    // Appends rows `[from, batch.size())`, copying them out of the batch.
    //
    // A batch arrives carrying the previous batch's final waypoint so the session can
    // verify the seam, so callers pass `from = 1` for every batch after the first to avoid
    // storing that waypoint twice.
    void append(const waypoint_accumulator& batch, std::size_t from) {
        if (from > batch.size()) {
            throw std::out_of_range("waypoint_store::append: `from` exceeds batch size");
        }
        if (from == batch.size()) {
            return;
        }
        if (!empty() && batch.dof() != dof()) {
            throw std::invalid_argument("waypoint_store::append: DOF mismatch");
        }

        storage_ = empty() ? detail::accumulator_tail_to_xarray(batch, from) : detail::concat_active_with_batch_tail(storage_, batch, from);
        invalidate_();
    }

    // Shrinks the store to its first `count` rows.
    //
    // The session builds a candidate trajectory before it knows whether that candidate can
    // be pivoted onto, so an append is provisional. Recording `size()` beforehand and
    // truncating back to it abandons the candidate without disturbing what came before.
    void truncate(std::size_t count) {
        if (count > size()) {
            throw std::out_of_range("waypoint_store::truncate: `count` exceeds stored size");
        }
        if (count == size()) {
            return;
        }

        storage_ = xt::xarray<double>{xt::view(storage_, xt::range(std::size_t{0}, count), xt::all())};
        invalidate_();
    }

    // Discards every row but the last, which remains as the seam that following batches
    // are checked and joined against.
    //
    // A rebase abandons the trajectory the session has finished emitting and starts a new
    // one from where that trajectory ended, so nothing before its final waypoint can
    // influence what comes next.
    void reset_to_last() {
        if (empty()) {
            return;
        }
        storage_ = detail::row_to_xarray(storage_, size() - 1);
        storage_.reshape({std::size_t{1}, storage_.size()});
        invalidate_();
    }

    // The stored waypoints in the form `path::create` accepts.
    //
    // The returned reference is invalidated by any subsequent mutation of the store, since
    // the accumulator's views reference the storage that mutation replaces.
    const waypoint_accumulator& waypoints() const {
        if (!accumulator_) {
            accumulator_.emplace(storage_);
        }
        return *accumulator_;
    }

    // The final stored row, copied out so it survives later mutation.
    xt::xarray<double> last() const {
        if (empty()) {
            throw std::out_of_range("waypoint_store::last: store is empty");
        }
        return detail::row_to_xarray(storage_, size() - 1);
    }

   private:
    // Dropped rather than rebuilt, because most mutations are followed by another mutation
    // rather than by a read, and rebuilding costs a view per stored waypoint.
    void invalidate_() {
        accumulator_.reset();
    }

    xt::xarray<double> storage_;

    // Built on first read after a mutation. Mutable so that `waypoints()` can stay const,
    // which is what its callers want; there is no thread-safety consideration here because
    // a session is driven from a single thread.
    mutable std::optional<waypoint_accumulator> accumulator_;
};

}  // namespace viam::trajex::totg::streaming
