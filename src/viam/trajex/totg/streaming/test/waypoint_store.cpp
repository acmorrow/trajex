// Tests for viam::trajex::totg::streaming::waypoint_store.

#include <viam/trajex/types/xt.hpp>

#include <viam/trajex/totg/streaming/private/waypoint_store.hpp>
#include <viam/trajex/totg/waypoint_accumulator.hpp>

#include <boost/test/unit_test.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

using viam::trajex::xmatrix;
using viam::trajex::xvector;
using viam::trajex::totg::waypoint_accumulator;
namespace streaming = viam::trajex::totg::streaming;

constexpr std::size_t k_dof = 2;

// Taken from the store rather than restated, so the boundary tests below follow it if it
// changes. At 2 DOF a chunk is 16 KiB, cheap enough to drive past repeatedly.
constexpr std::size_t k_chunk_rows = streaming::waypoint_store::k_chunk_rows;

// Row i is {i, -i}, so a row identifies itself and a transposition or an off-by-one in the
// chunk arithmetic shows up as a wrong value rather than as a plausible one.
xmatrix<> rows(std::size_t first, std::size_t count) {
    auto result = xmatrix<>::from_shape(std::vector<std::size_t>{count, k_dof});
    for (std::size_t i = 0; i != count; ++i) {
        result(i, 0) = static_cast<double>(first + i);
        result(i, 1) = -static_cast<double>(first + i);
    }
    return result;
}

// As above, at an arbitrary width. Element (i, j) is i * 100 + j, so a row read back through
// the wrong stride shows up as a neighbour's value rather than a plausible one.
xmatrix<> wide_rows(std::size_t count, std::size_t dof) {
    auto result = xmatrix<>::from_shape(std::vector<std::size_t>{count, dof});
    for (std::size_t i = 0; i != count; ++i) {
        for (std::size_t j = 0; j != dof; ++j) {
            result(i, j) = static_cast<double>((i * 100) + j);
        }
    }
    return result;
}

// waypoint_accumulator views its source rather than copying it, so the matrix has to outlive
// the accumulator. The store copies rows out during append, so pairing the two for the
// duration of the call is enough.
void append_rows(streaming::waypoint_store& store, std::size_t first, std::size_t count, std::size_t from = 0) {
    const auto matrix = rows(first, count);
    store.append(waypoint_accumulator{matrix}, from);
}

void check_row_is(const xvector<>& row, std::size_t expected) {
    BOOST_REQUIRE_EQUAL(row.size(), k_dof);
    BOOST_CHECK_EQUAL(row(0), static_cast<double>(expected));
    BOOST_CHECK_EQUAL(row(1), -static_cast<double>(expected));
}

// Every stored row, in order, read back through the accumulator the session would hand to
// path::create.
void check_contents_are(const streaming::waypoint_store& store, std::size_t first, std::size_t count) {
    BOOST_REQUIRE_EQUAL(store.size(), count);
    const auto& accumulator = store.waypoints();
    BOOST_REQUIRE_EQUAL(accumulator.size(), count);
    for (std::size_t i = 0; i != count; ++i) {
        check_row_is(xvector<>{accumulator.at(i)}, first + i);
    }
}

}  // namespace

BOOST_AUTO_TEST_SUITE(waypoint_store_tests)

BOOST_AUTO_TEST_CASE(empty_store_reports_empty_and_refuses_reads) {
    const streaming::waypoint_store store;

    BOOST_CHECK(store.empty());
    BOOST_CHECK_EQUAL(store.size(), 0U);
    BOOST_CHECK_EQUAL(store.dof(), 0U);
    BOOST_CHECK_THROW(static_cast<void>(store.waypoints()), std::out_of_range);
    BOOST_CHECK_THROW(static_cast<void>(store.last()), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(append_stores_rows_and_honours_from) {
    streaming::waypoint_store store;

    const auto first = rows(0, 4);
    store.append(waypoint_accumulator{first}, 0);
    check_contents_are(store, 0, 4);
    BOOST_CHECK_EQUAL(store.dof(), k_dof);

    // A following batch repeats the seam waypoint at row zero, which `from = 1` drops.
    const auto second = rows(3, 4);
    store.append(waypoint_accumulator{second}, 1);
    check_contents_are(store, 0, 7);
    check_row_is(store.last(), 6);
}

BOOST_AUTO_TEST_CASE(append_rejects_from_past_the_batch_and_ignores_an_empty_tail) {
    streaming::waypoint_store store;

    const auto batch = rows(0, 3);
    BOOST_CHECK_THROW(store.append(waypoint_accumulator{batch}, 4), std::out_of_range);
    BOOST_CHECK(store.empty());

    // `from == size()` is the no-new-waypoints case and must be silent, not an error.
    BOOST_CHECK_NO_THROW(store.append(waypoint_accumulator{batch}, 3));
    BOOST_CHECK(store.empty());
}

BOOST_AUTO_TEST_CASE(append_crossing_a_chunk_boundary_mid_batch) {
    streaming::waypoint_store store;

    // Land just short of the boundary, then cross it inside a single batch so the
    // allocate-on-end branch runs mid-loop rather than at entry.
    append_rows(store, 0, k_chunk_rows - 3);
    BOOST_REQUIRE_EQUAL(store.size(), k_chunk_rows - 3);

    append_rows(store, k_chunk_rows - 3, 10);
    check_contents_are(store, 0, k_chunk_rows + 7);
    check_row_is(store.last(), k_chunk_rows + 6);
}

BOOST_AUTO_TEST_CASE(append_landing_exactly_on_a_chunk_boundary) {
    streaming::waypoint_store store;

    append_rows(store, 0, k_chunk_rows);
    BOOST_REQUIRE_EQUAL(store.size(), k_chunk_rows);
    check_row_is(store.last(), k_chunk_rows - 1);

    // The next append enters with offset == k_chunk_rows and must allocate before writing.
    append_rows(store, k_chunk_rows, 5);
    check_contents_are(store, 0, k_chunk_rows + 5);
}

BOOST_AUTO_TEST_CASE(append_spanning_several_chunks) {
    streaming::waypoint_store store;

    const auto count = (3 * k_chunk_rows) + 11;
    append_rows(store, 0, count);
    check_contents_are(store, 0, count);
    check_row_is(store.last(), count - 1);
}

BOOST_AUTO_TEST_CASE(truncate_back_across_a_chunk_boundary_then_refill) {
    streaming::waypoint_store store;

    append_rows(store, 0, k_chunk_rows + 20);
    BOOST_REQUIRE_EQUAL(store.size(), k_chunk_rows + 20);

    // Back into the first chunk, abandoning the tail of chunk zero and all of chunk one.
    store.truncate(k_chunk_rows - 5);
    check_contents_are(store, 0, k_chunk_rows - 5);
    check_row_is(store.last(), k_chunk_rows - 6);

    // Refilling must reuse the emptied chunks rather than write past them.
    append_rows(store, 9000, k_chunk_rows);
    BOOST_REQUIRE_EQUAL(store.size(), (2 * k_chunk_rows) - 5);
    check_row_is(store.last(), 9000 + k_chunk_rows - 1);

    const auto& accumulator = store.waypoints();
    check_row_is(xvector<>{accumulator.at(k_chunk_rows - 6)}, k_chunk_rows - 6);
    check_row_is(xvector<>{accumulator.at(k_chunk_rows - 5)}, 9000);
}

BOOST_AUTO_TEST_CASE(truncate_to_zero_empties_the_store) {
    streaming::waypoint_store store;

    append_rows(store, 0, 6);
    store.truncate(0);

    BOOST_CHECK(store.empty());
    BOOST_CHECK_EQUAL(store.size(), 0U);
    BOOST_CHECK_THROW(static_cast<void>(store.waypoints()), std::out_of_range);
    BOOST_CHECK_THROW(static_cast<void>(store.last()), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(truncate_rejects_a_count_past_the_end) {
    streaming::waypoint_store store;

    append_rows(store, 0, 4);
    BOOST_CHECK_THROW(store.truncate(5), std::out_of_range);
    check_contents_are(store, 0, 4);
}

// A failed first extend truncates the store to zero and the session promises the caller may
// retry with a corrected batch. That retry is the case where a remembered DOF bites.
BOOST_AUTO_TEST_CASE(truncate_to_zero_forgets_the_dof_so_a_retry_can_change_it) {
    streaming::waypoint_store store;

    append_rows(store, 0, 3);
    BOOST_REQUIRE_EQUAL(store.dof(), k_dof);

    store.truncate(0);
    BOOST_CHECK_EQUAL(store.dof(), 0U);

    // The retry must not land in chunks shaped for the old width. Distinct values per element
    // catch the wrong stride, which overlaps neighbouring rows rather than failing.
    const auto wider = wide_rows(1030, 3);
    BOOST_CHECK_NO_THROW(store.append(waypoint_accumulator{wider}, 0));
    BOOST_REQUIRE_EQUAL(store.dof(), 3U);
    BOOST_REQUIRE_EQUAL(store.size(), 1030U);

    const auto& accumulator = store.waypoints();
    for (std::size_t i = 0; i != 1030; ++i) {
        const auto& row = accumulator.at(i);
        BOOST_REQUIRE_EQUAL(row.size(), 3U);
        for (std::size_t joint = 0; joint != 3; ++joint) {
            BOOST_CHECK_EQUAL(row(joint), wider(i, joint));
        }
    }
}

BOOST_AUTO_TEST_CASE(append_rejects_a_dof_change_while_rows_are_held) {
    streaming::waypoint_store store;

    append_rows(store, 0, 3);

    const auto three_dof = wide_rows(2, 3);
    BOOST_CHECK_THROW(store.append(waypoint_accumulator{three_dof}, 0), std::invalid_argument);
    check_contents_are(store, 0, 3);
}

BOOST_AUTO_TEST_CASE(reset_to_last_keeps_the_seam_when_it_lives_in_a_later_chunk) {
    streaming::waypoint_store store;

    const auto count = (2 * k_chunk_rows) + 30;
    append_rows(store, 0, count);

    store.reset_to_last();
    check_contents_are(store, count - 1, 1);
    check_row_is(store.last(), count - 1);

    // The survivor now lives at row zero of chunk zero, which is where it was about to be
    // overwritten from. Appending after it must continue from there.
    append_rows(store, 5000, 4);
    BOOST_REQUIRE_EQUAL(store.size(), 5U);
    const auto& accumulator = store.waypoints();
    check_row_is(xvector<>{accumulator.at(0)}, count - 1);
    check_row_is(xvector<>{accumulator.at(1)}, 5000);
    check_row_is(store.last(), 5003);
}

BOOST_AUTO_TEST_CASE(reset_to_last_when_the_seam_is_already_row_zero) {
    streaming::waypoint_store store;

    // Two rows in chunk zero: the survivor is read from row one and written over row zero.
    append_rows(store, 7, 2);
    store.reset_to_last();
    check_contents_are(store, 8, 1);
}

BOOST_AUTO_TEST_CASE(reset_to_last_is_a_no_op_at_or_below_one_row) {
    streaming::waypoint_store empty;
    BOOST_CHECK_NO_THROW(empty.reset_to_last());
    BOOST_CHECK(empty.empty());

    streaming::waypoint_store single;
    append_rows(single, 4, 1);
    single.reset_to_last();
    check_contents_are(single, 4, 1);
}

// The accumulator hands out views into the chunk arrays, so the rows it yields must be the
// store's own storage and must keep reading correctly as later appends grow the chunk list.
BOOST_AUTO_TEST_CASE(accumulator_rows_track_the_store_across_a_chunk_allocation) {
    streaming::waypoint_store store;

    append_rows(store, 0, 4);
    const double* const first_row = &store.waypoints().at(0)(0);

    append_rows(store, 4, k_chunk_rows + 8);

    // Allocating chunks must not have relocated what the earlier views point at.
    BOOST_CHECK_EQUAL(&store.waypoints().at(0)(0), first_row);
    check_contents_are(store, 0, k_chunk_rows + 12);
}

BOOST_AUTO_TEST_SUITE_END()
