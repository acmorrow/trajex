#pragma once

// Persistent waypoint storage for a streaming session.
//
// A session cannot retain the caller's `waypoint_accumulator`, which holds row-views into
// memory the caller may reuse or destroy as soon as `extend` returns. The store keeps its own
// copy and presents it back as an accumulator.
//
// Waypoints live in fixed-size chunks, each allocated separately and reached through a vector
// of pointers. A session appends on every extend, so storage that reallocated to grow would be
// O(N) per extend and O(N^2) across a session. The chunk arrays must never move: the
// accumulator below holds views referencing them by address.
//
// Header-only and not installed. The session and the pipeline benchmarks share it, so the
// benchmark measures the real storage rather than a copy that can drift.

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#if __has_include(<xtensor/views/xview.hpp>)
#include <xtensor/views/xview.hpp>
#else
#include <xtensor/xview.hpp>
#endif

#include <viam/trajex/totg/waypoint_accumulator.hpp>
#include <viam/trajex/types/xt.hpp>

namespace viam::trajex::totg::streaming {

class waypoint_store {
   public:
    // Rows per chunk. At six degrees of freedom a chunk is roughly 48 KiB, so a session holding
    // tens of thousands of waypoints costs a few dozen allocations, and a short move wastes at
    // most one chunk's worth of unused rows. Public so the tests can drive past a boundary
    // without restating the number.
    static constexpr std::size_t k_chunk_rows = 1024;

    waypoint_store() = default;

    // The accumulator in `waypoints()` holds views into the chunk arrays and into itself, so a
    // moved-from store would leave a copy whose views still looked valid.
    waypoint_store(const waypoint_store&) = delete;
    waypoint_store& operator=(const waypoint_store&) = delete;
    waypoint_store(waypoint_store&&) = delete;
    waypoint_store& operator=(waypoint_store&&) = delete;

    std::size_t size() const noexcept {
        return size_;
    }

    std::size_t dof() const noexcept {
        return dof_;
    }

    bool empty() const noexcept {
        return size_ == 0;
    }

    // Appends rows `[from, batch.size())`, copying them out of the batch.
    //
    // A batch arrives carrying the previous batch's final waypoint so the session can verify
    // the seam, so callers pass `from = 1` for every batch after the first to avoid storing
    // that waypoint twice.
    void append(const waypoint_accumulator& batch, std::size_t from) {
        if (from > batch.size()) {
            throw std::out_of_range("waypoint_store::append: `from` exceeds batch size");
        }
        if (from == batch.size()) {
            return;
        }
        if (dof_ != 0 && batch.dof() != dof_) {
            throw std::invalid_argument("waypoint_store::append: DOF mismatch");
        }
        dof_ = batch.dof();

        auto chunk_index = size_ / k_chunk_rows;
        auto offset = size_ % k_chunk_rows;
        auto* chunk = &chunk_for_write_(chunk_index);

        for (std::size_t i = from; i != batch.size(); ++i) {
            if (offset == k_chunk_rows) {
                ++chunk_index;
                offset = 0;
                chunk = &chunk_for_write_(chunk_index);
            }

            // Copied element by element rather than as `xt::view(...) = batch.at(i)`, which
            // reads better but measured 27 to 40 percent slower in every benchmark cell.
            const auto& row = batch.at(i);
            for (std::size_t joint = 0; joint != dof_; ++joint) {
                (*chunk)(offset, joint) = row(joint);
            }
            extend_accumulator_(*chunk, offset);

            ++offset;
            ++size_;
        }
    }

    // Shrinks the store to its first `count` rows.
    //
    // The session builds a candidate trajectory before it knows whether that candidate can
    // be pivoted onto, so an append is provisional. Recording `size()` beforehand and
    // truncating back to it abandons the candidate without disturbing what came before.
    //
    // Emptied chunks are kept, not released. Appending and truncating is the session's ordinary
    // rhythm, so freeing a chunk the moment it empties just reallocates it on the next batch.
    //
    // Emptying releases the chunks and forgets the DOF, which is the one case where keeping
    // them would be wrong. A chunk is shaped to the width it was allocated at, so a retry at a
    // different width would index it with the wrong stride and write off the end. Only a failed
    // first extend empties the store, so the rhythm the paragraph above describes is unaffected.
    void truncate(std::size_t count) {
        if (count > size_) {
            throw std::out_of_range("waypoint_store::truncate: `count` exceeds stored size");
        }

        while (size_ != count) {
            accumulator_->pop_back();
            --size_;
        }
        if (size_ == 0) {
            accumulator_.reset();
            chunks_.clear();
            dof_ = 0;
        }
    }

    // Discards every row but the last, which remains as the seam that following batches are
    // checked and joined against.
    //
    // A rebase abandons the trajectory the session has finished emitting and starts a new
    // one from where that trajectory ended, so nothing before its final waypoint can
    // influence what comes next.
    void reset_to_last() {
        if (size_ <= 1) {
            return;
        }

        // Copied out before anything is overwritten, since the surviving waypoint is about
        // to be written over row zero and may currently live there.
        const auto surviving = last();

        accumulator_.reset();
        size_ = 0;

        auto& first = *chunks_.front();
        xt::view(first, 0, xt::all()) = surviving;
        size_ = 1;
        extend_accumulator_(first, 0);
    }

    // The stored waypoints in the form `path::create` accepts.
    //
    // The returned reference is invalidated by any subsequent mutation of the store, which
    // may append to or pop from the accumulator it refers to.
    const waypoint_accumulator& waypoints() const {
        if (!accumulator_) {
            throw std::out_of_range("waypoint_store::waypoints: store is empty");
        }
        return *accumulator_;
    }

    // The final stored row, copied out so it survives later mutation.
    xvector<> last() const {
        if (empty()) {
            throw std::out_of_range("waypoint_store::last: store is empty");
        }
        const auto index = size_ - 1;
        return xvector<>{xt::view(chunk_at_(index / k_chunk_rows), index % k_chunk_rows, xt::all())};
    }

   private:
    // Only the pointer vector reallocates as chunks are added; the arrays it points at stay
    // where they were allocated, which is what the accumulator's views require. See the top of
    // this file.
    using chunk_list = std::vector<std::unique_ptr<xmatrix<>>>;

    // Every chunk but the one currently being filled is exactly full, so a row's position
    // follows from its index alone and chunks need carry no fill count of their own.
    xmatrix<>& allocate_chunk_() {
        chunks_.push_back(std::make_unique<xmatrix<>>(xmatrix<>::from_shape(std::vector<std::size_t>{k_chunk_rows, dof_})));
        return *chunks_.back();
    }

    xmatrix<>& chunk_for_write_(std::size_t index) {
        while (chunks_.size() <= index) {
            allocate_chunk_();
        }
        return *chunks_[index];
    }

    const xmatrix<>& chunk_at_(std::size_t index) const {
        return *chunks_[index];
    }

    // Grown a row at a time as they are written rather than rebuilt on read, since the session
    // reads it on every extend and rebuilding costs a view per stored waypoint.
    void extend_accumulator_(const xmatrix<>& chunk, std::size_t offset) {
        const auto row = xt::adapt(&chunk(offset, 0), dof_, xt::no_ownership());
        if (accumulator_) {
            accumulator_->add_waypoint(row);
        } else {
            accumulator_.emplace(row);
        }
    }

    chunk_list chunks_;
    std::size_t size_ = 0;
    std::size_t dof_ = 0;
    std::optional<waypoint_accumulator> accumulator_;
};

}  // namespace viam::trajex::totg::streaming
