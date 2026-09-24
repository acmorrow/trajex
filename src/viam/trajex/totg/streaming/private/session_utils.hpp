#pragma once

// Materialization helpers for the streaming session.
//
// These copy accumulator row-views and 2D-xarray slices into owned xarrays. The session
// needs stable storage independent of caller-provided accumulators, because
// waypoint_accumulator holds row-views into source arrays it does not own.
//
// This is a private header: header-only, no library backing, and not installed. It exists
// so the pipeline benchmarks can measure this marshalling without duplicating it and
// letting the copy drift from the original. See RSDK-14499.

#include <cstddef>
#include <vector>

#if __has_include(<xtensor/containers/xarray.hpp>)
#include <xtensor/containers/xarray.hpp>
#else
#include <xtensor/xarray.hpp>
#endif

#include <viam/trajex/totg/waypoint_accumulator.hpp>

namespace viam::trajex::totg::streaming::detail {

inline xt::xarray<double> view_to_xarray(const waypoint_accumulator::value_type& row) {
    const std::size_t dof = row.shape(0);
    xt::xarray<double> result = xt::zeros<double>(std::vector<std::size_t>{dof});
    for (std::size_t j = 0; j < dof; ++j) {
        result(j) = row(j);
    }
    return result;
}

inline xt::xarray<double> row_to_xarray(const xt::xarray<double>& arr, std::size_t row) {
    const std::size_t dof = arr.shape(1);
    xt::xarray<double> result = xt::zeros<double>(std::vector<std::size_t>{dof});
    for (std::size_t j = 0; j < dof; ++j) {
        result(j) = arr(row, j);
    }
    return result;
}

inline bool rows_bit_exact(const waypoint_accumulator::value_type& a, const xt::xarray<double>& b) {
    if (a.shape(0) != b.shape(0)) {
        return false;
    }
    for (std::size_t i = 0; i < b.shape(0); ++i) {
        if (a(i) != b(i)) {
            return false;
        }
    }
    return true;
}

inline xt::xarray<double> accumulator_to_xarray(const waypoint_accumulator& batch) {
    const std::size_t count = batch.size();
    const std::size_t dof = batch.dof();
    xt::xarray<double> result = xt::zeros<double>(std::vector<std::size_t>{count, dof});
    for (std::size_t i = 0; i < count; ++i) {
        const auto& row = batch.at(i);
        for (std::size_t j = 0; j < dof; ++j) {
            result(i, j) = row(j);
        }
    }
    return result;
}

// Caller must ensure batch.size() > from.
inline xt::xarray<double> accumulator_tail_to_xarray(const waypoint_accumulator& batch, std::size_t from) {
    const std::size_t count = batch.size() - from;
    const std::size_t dof = batch.dof();
    xt::xarray<double> result = xt::zeros<double>(std::vector<std::size_t>{count, dof});
    for (std::size_t i = 0; i < count; ++i) {
        const auto& row = batch.at(from + i);
        for (std::size_t j = 0; j < dof; ++j) {
            result(i, j) = row(j);
        }
    }
    return result;
}

inline xt::xarray<double> concat_active_with_batch_tail(const xt::xarray<double>& base,
                                                        const waypoint_accumulator& batch,
                                                        std::size_t batch_from) {
    const std::size_t n_base = base.shape(0);
    const std::size_t n_add = batch.size() - batch_from;
    const std::size_t dof = base.shape(1);
    xt::xarray<double> result = xt::zeros<double>(std::vector<std::size_t>{n_base + n_add, dof});
    for (std::size_t i = 0; i < n_base; ++i) {
        for (std::size_t j = 0; j < dof; ++j) {
            result(i, j) = base(i, j);
        }
    }
    for (std::size_t i = 0; i < n_add; ++i) {
        const auto& row = batch.at(batch_from + i);
        for (std::size_t j = 0; j < dof; ++j) {
            result(n_base + i, j) = row(j);
        }
    }
    return result;
}

inline xt::xarray<double> stack_anchor_and_staged(const xt::xarray<double>& anchor, const std::vector<xt::xarray<double>>& staged) {
    const std::size_t dof = anchor.shape(0);
    std::size_t total_rows = 1;
    for (const auto& s : staged) {
        total_rows += s.shape(0);
    }
    xt::xarray<double> result = xt::zeros<double>(std::vector<std::size_t>{total_rows, dof});
    for (std::size_t j = 0; j < dof; ++j) {
        result(0, j) = anchor(j);
    }
    std::size_t row = 1;
    for (const auto& s : staged) {
        for (std::size_t i = 0; i < s.shape(0); ++i) {
            for (std::size_t j = 0; j < dof; ++j) {
                result(row, j) = s(i, j);
            }
            ++row;
        }
    }
    return result;
}

}  // namespace viam::trajex::totg::streaming::detail
