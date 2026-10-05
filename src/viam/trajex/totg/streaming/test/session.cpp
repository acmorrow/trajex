// Tests for viam::trajex::totg::streaming::session.

#include <viam/trajex/types/xt.hpp>

#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/totg/streaming/session.hpp>
#include <viam/trajex/totg/trajectory.hpp>
#include <viam/trajex/totg/waypoint_accumulator.hpp>
#include <viam/trajex/types/hertz.hpp>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using viam::trajex::xmatrix;
using viam::trajex::xvector;
using viam::trajex::totg::path;
using viam::trajex::totg::trajectory;
using viam::trajex::totg::waypoint_accumulator;
namespace streaming = viam::trajex::totg::streaming;
namespace types = viam::trajex::types;

// A small but real fixture: 2 DOF, modest velocity / acceleration limits, and a 100 Hz
// sample rate. The waypoint sets below produce trajectories long enough that pulling a
// dozen samples at 100 Hz stays well within the trajectory's duration.

constexpr double k_sample_rate_hz = 100.0;

types::hertz default_sample_rate() {
    return types::hertz{k_sample_rate_hz};
}

trajectory::options default_trajectory_options() {
    trajectory::options topt;
    topt.max_velocity = xvector<>{2.0, 2.0};
    topt.max_acceleration = xvector<>{5.0, 5.0};
    return topt;
}

path::options default_path_options() {
    path::options popt;
    popt.set_max_blend_deviation(0.05);
    return popt;
}

xmatrix<> three_waypoints() {
    return xmatrix<>{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}};
}

xmatrix<> six_waypoints() {
    return xmatrix<>{
        {0.0, 0.0},
        {1.0, 0.0},
        {1.0, 1.0},
        {2.0, 1.0},
        {2.0, 2.0},
        {3.0, 2.0},
    };
}

// Builds a trajectory directly from waypoints with the same options the session uses.
// This is the reference any session sample stream should agree with where the active
// trajectory's geometry matches the full merged waypoint set.
trajectory reference_trajectory(const xmatrix<>& waypoints) {
    path p = path::create(waypoints, default_path_options());
    return trajectory::create(std::move(p), default_trajectory_options());
}

// Pins a waypoints matrix and a waypoint_accumulator over it together so the pair can be
// handed to session.extend() in a single expression without lifetime hazards.
//
// waypoint_accumulator holds views into its source matrix and explicitly deletes the
// rvalue-source constructor; ad-hoc factories returning an accumulator over a temporary
// matrix won't compile. This wrapper stores both pieces and pins itself.
//
// TODO(streaming-test-utils): hoist alongside other shared test helpers once a second
// streaming test file needs the same scaffolding. Until then, duplication here is cheap.
class pinned_waypoints {
   public:
    explicit pinned_waypoints(xmatrix<> data) : data_(std::move(data)), accumulator_(data_) {}

    pinned_waypoints(const pinned_waypoints&) = delete;
    pinned_waypoints& operator=(const pinned_waypoints&) = delete;
    pinned_waypoints(pinned_waypoints&&) = delete;
    pinned_waypoints& operator=(pinned_waypoints&&) = delete;

    const waypoint_accumulator& accumulator() const noexcept {
        return accumulator_;
    }
    const xmatrix<>& data() const noexcept {
        return data_;
    }

   private:
    xmatrix<> data_;
    waypoint_accumulator accumulator_;
};

// Returns true if every element of `a` is within `tolerance` of `b`. Used for direct
// configuration / velocity / acceleration comparisons between session samples and
// reference trajectory samples.
bool configs_match(const xvector<>& a, const xvector<>& b, double tolerance = 1e-9) {
    if (a.shape(0) != b.shape(0)) {
        return false;
    }
    for (std::size_t i = 0; i < a.shape(0); ++i) {
        if (std::abs(a(i) - b(i)) > tolerance) {
            return false;
        }
    }
    return true;
}

// Helper for the cornerstone equivalence test: verify that every sample the session
// emitted agrees with what a directly-built reference trajectory would produce at the
// same time. The reference is queried via `trajectory::sample(t)` which is random-access.
void check_samples_match_reference(const std::vector<struct trajectory::sample>& session_samples, const trajectory& reference) {
    for (const auto& s : session_samples) {
        if (s.time < trajectory::seconds{0.0} || s.time > reference.duration()) {
            continue;  // sample is past the reference's coverage; nothing to compare against
        }
        const auto expected = reference.sample(s.time);
        BOOST_CHECK(configs_match(s.configuration, expected.configuration));
        BOOST_CHECK(configs_match(s.velocity, expected.velocity));
        BOOST_CHECK(configs_match(s.acceleration, expected.acceleration));
    }
}

streaming::session fresh_session() {
    return streaming::session{default_path_options(), default_trajectory_options(), default_sample_rate()};
}

// The slow sample rate shared by the tests around the pivot overshoot guard, along with the
// durations it is derived from. The active trajectory is the three-waypoint one, and the
// candidate appends a tiny tail, {1.0, 1.05}, to it. The period is chosen so that resuming
// one period past D_act/2 overshoots the candidate's duration by a fixed margin. Setting
// period = D_cand - D_act/2 + margin makes that resume offset D_cand + margin whatever the
// actual durations are. The period then lies in [D_act/2, D_act), so quantized_for_trajectory
// samples the active trajectory on a three-sample grid at 0, D_act/2 and D_act.
struct slow_rate {
    trajectory::seconds active_duration;
    trajectory::seconds candidate_duration;
    double period_sec;

    streaming::session make_session() const {
        return streaming::session{default_path_options(), default_trajectory_options(), types::hertz{1.0 / period_sec}};
    }
};

slow_rate make_slow_rate() {
    constexpr double k_margin_sec = 0.05;
    const auto active_duration = reference_trajectory(three_waypoints()).duration();
    const auto candidate_duration = reference_trajectory(xmatrix<>{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {1.0, 1.05}}).duration();
    return {active_duration, candidate_duration, candidate_duration.count() - (active_duration.count() / 2.0) + k_margin_sec};
}

}  // namespace

BOOST_AUTO_TEST_SUITE(streaming_session_tests)

BOOST_AUTO_TEST_SUITE(empty_session)

BOOST_AUTO_TEST_CASE(fresh_session_has_zero_current_time) {
    auto sess = fresh_session();
    BOOST_CHECK_EQUAL(sess.current_time().count(), 0.0);
}

BOOST_AUTO_TEST_CASE(fresh_session_has_no_active_trajectory) {
    auto sess = fresh_session();
    BOOST_CHECK(sess.active_trajectory() == nullptr);
    BOOST_CHECK_EQUAL(sess.active_epoch().count(), 0.0);
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 0U);
}

BOOST_AUTO_TEST_CASE(fresh_session_sample_next_returns_empty) {
    auto sess = fresh_session();
    const auto samples = sess.sample_next(5);
    BOOST_CHECK(samples.empty());
}

BOOST_AUTO_TEST_CASE(fresh_session_sample_at_least_returns_empty) {
    auto sess = fresh_session();
    const auto samples = sess.sample_at_least(trajectory::seconds{1.0});
    BOOST_CHECK(samples.empty());
}

BOOST_AUTO_TEST_SUITE_END()  // empty_session

BOOST_AUTO_TEST_SUITE(first_extend)

BOOST_AUTO_TEST_CASE(first_extend_with_valid_batch_creates_active_trajectory) {
    auto sess = fresh_session();
    const pinned_waypoints wp(three_waypoints());

    const auto result = sess.extend(wp.accumulator());

    BOOST_CHECK(sess.active_trajectory() != nullptr);
    BOOST_CHECK_EQUAL(sess.active_epoch().count(), 0.0);
    BOOST_CHECK_EQUAL(sess.current_time().count(), 0.0);
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 1U);

    // There was nothing to branch from, so no slack is reported, and the whole of the new
    // trajectory counts as growth.
    BOOST_CHECK(result.kind == streaming::session::extend_result::kinds::k_first_build);
    BOOST_CHECK(!result.branch_slack.has_value());
    BOOST_REQUIRE(result.delta_active_duration.has_value());
    BOOST_CHECK_EQUAL(result.delta_active_duration->count(), sess.active_trajectory()->duration().count());
}

BOOST_AUTO_TEST_CASE(first_extend_with_single_waypoint_propagates_invalid_argument) {
    auto sess = fresh_session();
    const pinned_waypoints wp(xmatrix<>{{0.0, 0.0}});

    BOOST_CHECK_THROW(sess.extend(wp.accumulator()), std::invalid_argument);

    // A failed first extend must leave the session as fresh as it was, so that a caller can
    // retry with a corrected batch.
    BOOST_CHECK(sess.active_trajectory() == nullptr);
    BOOST_CHECK_EQUAL(sess.current_time().count(), 0.0);
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 0U);
}

BOOST_AUTO_TEST_CASE(first_extend_with_dof_mismatch_against_options_propagates_invalid_argument) {
    // Default options have 2-DOF velocity / acceleration limits; provide 3-DOF waypoints
    // so that trajectory construction rejects the result.
    auto sess = fresh_session();
    const pinned_waypoints wp(xmatrix<>{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}});

    BOOST_CHECK_THROW(sess.extend(wp.accumulator()), std::invalid_argument);

    BOOST_CHECK(sess.active_trajectory() == nullptr);
    BOOST_CHECK_EQUAL(sess.current_time().count(), 0.0);
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 0U);
}

BOOST_AUTO_TEST_SUITE_END()  // first_extend

BOOST_AUTO_TEST_SUITE(sampling_primitives)

BOOST_AUTO_TEST_CASE(sample_next_default_emits_one_sample) {
    auto sess = fresh_session();
    const pinned_waypoints wp(six_waypoints());
    sess.extend(wp.accumulator());

    const auto samples = sess.sample_next();
    BOOST_CHECK_EQUAL(samples.size(), 1U);
}

BOOST_AUTO_TEST_CASE(sample_next_n_emits_n_samples) {
    auto sess = fresh_session();
    const pinned_waypoints wp(six_waypoints());
    sess.extend(wp.accumulator());

    const auto samples = sess.sample_next(10);
    BOOST_CHECK_EQUAL(samples.size(), 10U);
}

BOOST_AUTO_TEST_CASE(sample_at_least_advances_at_least_horizon) {
    auto sess = fresh_session();
    const pinned_waypoints wp(six_waypoints());
    sess.extend(wp.accumulator());

    const auto horizon = trajectory::seconds{0.1};  // 100 ms
    const auto samples = sess.sample_at_least(horizon);
    BOOST_REQUIRE(!samples.empty());
    BOOST_CHECK_GE(sess.current_time().count(), horizon.count());
}

BOOST_AUTO_TEST_CASE(sample_at_least_zero_horizon_returns_exactly_one_sample) {
    auto sess = fresh_session();
    const pinned_waypoints wp(six_waypoints());
    sess.extend(wp.accumulator());

    // With a zero horizon the target is the current time itself, which the first emitted
    // sample always reaches, so exactly one sample comes back.
    const auto samples = sess.sample_at_least(trajectory::seconds{0.0});
    BOOST_CHECK_EQUAL(samples.size(), 1U);
}

BOOST_AUTO_TEST_CASE(current_time_tracks_last_emitted_sample) {
    auto sess = fresh_session();
    const pinned_waypoints wp(six_waypoints());
    sess.extend(wp.accumulator());

    const auto samples = sess.sample_next(7);
    BOOST_REQUIRE_EQUAL(samples.size(), 7U);
    BOOST_CHECK_EQUAL(sess.current_time().count(), samples.back().time.count());
}

BOOST_AUTO_TEST_SUITE_END()  // sampling_primitives

BOOST_AUTO_TEST_SUITE(single_trajectory_equivalence)

BOOST_AUTO_TEST_CASE(session_with_one_extend_matches_direct_trajectory) {
    auto sess = fresh_session();
    const auto waypoints = six_waypoints();
    const pinned_waypoints wp(waypoints);
    sess.extend(wp.accumulator());

    const auto reference = reference_trajectory(waypoints);
    const auto samples = sess.sample_at_least(reference.duration());
    BOOST_REQUIRE(!samples.empty());

    check_samples_match_reference(samples, reference);
}

BOOST_AUTO_TEST_SUITE_END()  // single_trajectory_equivalence

BOOST_AUTO_TEST_SUITE(seam_validation)

BOOST_AUTO_TEST_CASE(second_extend_with_seam_mismatch_throws_invalid_argument) {
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());

    // The last stored waypoint is {1.0, 1.0}. Provide a batch whose first waypoint differs.
    const pinned_waypoints mismatched(xmatrix<>{{9.0, 9.0}, {2.0, 2.0}});
    BOOST_CHECK_THROW(sess.extend(mismatched.accumulator()), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(second_extend_with_dof_mismatch_throws_invalid_argument) {
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());

    const pinned_waypoints wrong_dof(xmatrix<>{{1.0, 1.0, 0.0}, {2.0, 2.0, 0.0}});
    BOOST_CHECK_THROW(sess.extend(wrong_dof.accumulator()), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(second_extend_with_bit_exact_seam_matches_merged_reference) {
    // The merged waypoint set after seam-drop is {{0,0},{1,0},{1,1},{2,1},{2,2}}.
    // The sample stream after both extends should agree with a reference trajectory
    // built directly over that merged set.
    auto sess = fresh_session();
    const pinned_waypoints initial(xmatrix<>{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}});
    sess.extend(initial.accumulator());

    const pinned_waypoints extension(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(extension.accumulator());

    const xmatrix<> merged{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}};
    const auto reference = reference_trajectory(merged);

    const auto samples = sess.sample_at_least(reference.duration());
    BOOST_REQUIRE(!samples.empty());

    check_samples_match_reference(samples, reference);
}

BOOST_AUTO_TEST_CASE(seam_only_batch_leaves_the_session_untouched) {
    // A batch carrying nothing past the seam has no waypoints to absorb, so extend returns
    // without building a candidate. What matters is less the returned kind than that the call
    // is a true no-op, neither advancing the chain nor disturbing the sampler.
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    sess.sample_next(2);

    const auto generation_before = sess.trajectory_generation_count();
    const auto time_before = sess.current_time();
    const auto duration_before = sess.active_trajectory()->duration();

    const pinned_waypoints seam_only(xmatrix<>{{1.0, 1.0}});
    const auto result = sess.extend(seam_only.accumulator());

    BOOST_CHECK(result.kind == streaming::session::extend_result::kinds::k_noop);
    BOOST_CHECK(!result.branch_slack.has_value());
    BOOST_CHECK(!result.delta_active_duration.has_value());

    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), generation_before);
    BOOST_CHECK_EQUAL(sess.current_time().count(), time_before.count());
    BOOST_CHECK_EQUAL(sess.active_trajectory()->duration().count(), duration_before.count());
}

BOOST_AUTO_TEST_SUITE_END()  // seam_validation

BOOST_AUTO_TEST_SUITE(pivot)

BOOST_AUTO_TEST_CASE(extend_with_branch_ahead_of_last_sample_pivots) {
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);

    // After one sample the branch is still far ahead, since it sits near the terminal blend of
    // the first trajectory, most of a trajectory away.
    sess.sample_next(1);

    const auto duration_before = sess.active_trajectory()->duration();
    const pinned_waypoints extension(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    const auto result = sess.extend(extension.accumulator());

    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 2U);
    BOOST_CHECK(sess.active_trajectory() != nullptr);

    // The pivot is admitted precisely because the branch sits ahead of the last emitted
    // sample, so the reported slack must be positive. The duration delta has to be measured against the
    // trajectory being replaced, which means capturing its duration before the swap; reading
    // it afterwards would report zero.
    BOOST_CHECK(result.kind == streaming::session::extend_result::kinds::k_pivot);
    BOOST_REQUIRE(result.branch_slack.has_value());
    BOOST_CHECK_GT(result.branch_slack->count(), 0.0);
    BOOST_REQUIRE(result.delta_active_duration.has_value());
    BOOST_CHECK_EQUAL(result.delta_active_duration->count(), (sess.active_trajectory()->duration() - duration_before).count());
}

BOOST_AUTO_TEST_CASE(pivot_preserves_active_epoch) {
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    sess.sample_next(1);

    const auto pre_extend_epoch = sess.active_epoch();
    const pinned_waypoints extension(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(extension.accumulator());

    // The epoch check means nothing unless a pivot happened, so require one first.
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 2U);
    BOOST_CHECK_EQUAL(sess.active_epoch().count(), pre_extend_epoch.count());
}

BOOST_AUTO_TEST_CASE(pivot_preserves_current_time) {
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    sess.sample_next(3);

    const auto pre_extend_time = sess.current_time();
    const pinned_waypoints extension(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(extension.accumulator());

    // The current time check means nothing unless a pivot happened, so require one first.
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 2U);
    BOOST_CHECK_EQUAL(sess.current_time().count(), pre_extend_time.count());
}

BOOST_AUTO_TEST_CASE(pivot_whose_resume_offset_overshoots_candidate_stages) {
    // A pivot resumes its new sampler one sample period past the last emitted sample. If the
    // candidate has less than one sample period of trajectory left after the branch, that
    // offset lands at or past the candidate's end, where quantized_for_trajectory refuses to
    // start. The session should stage instead, since a pivot that close to the end would have
    // almost nothing new to emit, and the staged batch is delivered after the rebase.
    //
    // In production this arises at ordinary sample rates through corner-cutting. The active
    // trajectory's last waypoint is a hard endpoint, but in the candidate that waypoint becomes
    // interior and gets a circular blend, so the branch sits within one sample period of the
    // candidate's end. That geometry is fragile to reproduce deterministically, so we force
    // the same inequality with a slow sample rate instead, which leaves a small appended tail
    // with less than one long sample period of trajectory after the branch.
    //
    // To set it up, we use the slow rate from make_slow_rate and sample up to D_act/2, which is
    // the middle of the three-sample grid and comfortably ahead of the branch, so the extend
    // would otherwise pivot.
    const auto rate = make_slow_rate();
    auto sess = rate.make_session();

    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    sess.sample_next(2);

    // Check that the setup really does make the pivot's resume offset overshoot the candidate.
    BOOST_REQUIRE_GE(sess.current_time().count() + rate.period_sec, rate.candidate_duration.count());

    // The branch for the tiny appended tail is ahead of the last emitted sample, so it would
    // pivot, but the resume offset overshoots, so the session must stage instead of throwing.
    const pinned_waypoints extension(xmatrix<>{{1.0, 1.0}, {1.0, 1.05}});
    streaming::session::extend_result result{};
    BOOST_CHECK_NO_THROW(result = sess.extend(extension.accumulator()));
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 1U);

    // This is what tells the two kinds of compared stage apart. The branch was ahead of the
    // last emitted sample, so the slack is positive and lateness was never the problem.
    // Reporting k_staged_branch_sampled here would send a caller chasing the wrong remedy.
    BOOST_CHECK(result.kind == streaming::session::extend_result::kinds::k_staged_unsamplable);
    BOOST_REQUIRE(result.branch_slack.has_value());
    BOOST_CHECK_GT(result.branch_slack->count(), 0.0);

    // The batch starts the staged motion, so the growth is the whole of its duration.
    BOOST_REQUIRE(result.delta_active_duration.has_value());
    BOOST_CHECK_EQUAL(result.delta_active_duration->count(), reference_trajectory(extension.data()).duration().count());
}

BOOST_AUTO_TEST_CASE(overshoot_stage_then_drain_consumes_the_staged_batch) {
    // This continues the case above. Once the session has declined the pivot and staged the
    // tiny tail, draining it must still deliver that tail, which is valid, reachable motion
    // that must never be dropped. The tail is staged before the active trajectory's terminal
    // goes out, so the staged trajectory takes over at the seam and is sampled from its own
    // start, even though all of it fits inside one slow sample period.
    //
    // The construction mirrors the case above.
    const auto rate = make_slow_rate();
    auto sess = rate.make_session();

    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    sess.sample_next(2);  // up to D_act/2, mid-grid and ahead of the branch

    const pinned_waypoints extension(xmatrix<>{{1.0, 1.0}, {1.0, 1.05}});
    sess.extend(extension.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);  // staged, not pivoted

    // Draining through the active's end triggers the rebase. It must consume the staged tail,
    // neither throwing nor dropping it.
    std::vector<struct trajectory::sample> drained;
    BOOST_CHECK_NO_THROW(drained = sess.sample_next(8));

    // The rebase completed and made a trajectory for the staged batch active. The first sample
    // drained stands at the seam in place of the old terminal, and the second is the staged
    // trajectory's own terminal.
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 2U);
    BOOST_REQUIRE_EQUAL(drained.size(), 2U);
    BOOST_CHECK_EQUAL(drained.front().time.count(), rate.active_duration.count());

    // The batch's motion was delivered, since the last emitted sample reaches its final
    // waypoint.
    BOOST_REQUIRE(!drained.empty());
    const auto& terminal = drained.back();
    const xvector<> final_waypoint{1.0, 1.05};
    BOOST_CHECK(configs_match(terminal.configuration, final_waypoint, 1e-3));

    // The staged trajectory ends at rest, so its terminal sample has zero velocity and
    // acceleration. It lands exactly at that trajectory's end in global time, which is the
    // epoch, already advanced by the old duration, plus the staged trajectory's own duration.
    BOOST_CHECK_EQUAL(terminal.time.count(), (sess.active_epoch() + sess.active_trajectory()->duration()).count());
    BOOST_REQUIRE_EQUAL(terminal.velocity.shape(0), 2U);
    BOOST_REQUIRE_EQUAL(terminal.acceleration.shape(0), 2U);
    for (std::size_t i = 0; i < terminal.velocity.shape(0); ++i) {
        BOOST_CHECK_EQUAL(terminal.velocity(i), 0.0);
    }
    for (std::size_t i = 0; i < terminal.acceleration.shape(0); ++i) {
        BOOST_CHECK_EQUAL(terminal.acceleration(i), 0.0);
    }

    // Having consumed the batch, the session is now cleanly drained.
    std::vector<struct trajectory::sample> tail;
    BOOST_CHECK_NO_THROW(tail = sess.sample_next(1));
    BOOST_CHECK(tail.empty());
}

BOOST_AUTO_TEST_CASE(short_staged_motion_after_an_emitted_terminal_emits_only_its_terminal) {
    // When the active trajectory's terminal has already gone out, the staged trajectory is
    // sampled from one sample period in, so that the seam carries no duplicate sample. If the
    // staged trajectory is shorter than that period, the start would land past its end, where
    // quantized_for_trajectory refuses to start. The staged motion still has to be delivered,
    // so the rebase emits only the staged trajectory's terminal, where the arm has completed
    // the move.
    //
    // The sample rate is the same slow one as in the cases above, which gives the active
    // trajectory a three-sample grid and leaves the tiny tail's staged trajectory shorter than
    // one period.
    const auto rate = make_slow_rate();
    auto sess = rate.make_session();

    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());

    // Nothing is staged yet, so all three samples go out, the terminal among them.
    const auto first_chain = sess.sample_next(3);
    BOOST_REQUIRE_EQUAL(first_chain.size(), 3U);
    BOOST_REQUIRE_EQUAL(first_chain.back().time.count(), rate.active_duration.count());

    const pinned_waypoints extension(xmatrix<>{{1.0, 1.0}, {1.0, 1.05}});
    sess.extend(extension.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);  // staged

    const auto d_staged = reference_trajectory(xmatrix<>{{1.0, 1.0}, {1.0, 1.05}}).duration();
    BOOST_REQUIRE_GE(rate.period_sec, d_staged.count());

    std::vector<struct trajectory::sample> drained;
    BOOST_CHECK_NO_THROW(drained = sess.sample_next(8));
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 2U);

    // Only the staged trajectory's terminal comes out, at its end in global time.
    BOOST_REQUIRE_EQUAL(drained.size(), 1U);
    const auto& terminal = drained.front();
    BOOST_CHECK_EQUAL(terminal.time.count(), (rate.active_duration + d_staged).count());
    const xvector<> final_waypoint{1.0, 1.05};
    BOOST_CHECK(configs_match(terminal.configuration, final_waypoint, 1e-3));
}

BOOST_AUTO_TEST_SUITE_END()  // pivot

BOOST_AUTO_TEST_SUITE(stage_and_rebase)

BOOST_AUTO_TEST_CASE(extend_with_branch_behind_last_sample_stages) {
    // Sampling all the way to the active trajectory's terminal moves the last emitted sample
    // past the branch between the initial and merged trajectories, so the second extend must
    // stage rather than pivot.
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);

    const auto* initial_active = sess.active_trajectory();
    BOOST_REQUIRE(initial_active != nullptr);
    sess.sample_at_least(initial_active->duration());

    const pinned_waypoints extension(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    const auto result = sess.extend(extension.accumulator());

    // No new trajectory became active, so the generation count is unchanged.
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 1U);
    BOOST_CHECK_EQUAL(sess.active_epoch().count(), 0.0);

    // The last emitted sample has passed the branch, which is what forced the stage, so the
    // slack is non-positive. The batch starts the staged motion, so the growth is the whole of
    // its duration.
    BOOST_CHECK(result.kind == streaming::session::extend_result::kinds::k_staged_branch_sampled);
    BOOST_REQUIRE(result.branch_slack.has_value());
    BOOST_CHECK_LE(result.branch_slack->count(), 0.0);
    BOOST_REQUIRE(result.delta_active_duration.has_value());
    BOOST_CHECK_EQUAL(result.delta_active_duration->count(), reference_trajectory(extension.data()).duration().count());
}

BOOST_AUTO_TEST_CASE(staged_batch_rebases_when_sampling_past_terminal) {
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);

    const auto initial_duration = sess.active_trajectory()->duration();

    sess.sample_at_least(initial_duration);  // exhaust the active before extending

    const pinned_waypoints extension(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(extension.accumulator());

    // The batch staged, so the generation count is still 1.
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);

    // Sampling further triggers the rebase from the original trajectory's terminal pose.
    sess.sample_next(1);

    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 2U);
    BOOST_CHECK(sess.active_trajectory() != nullptr);
    BOOST_CHECK_EQUAL(sess.active_epoch().count(), initial_duration.count());
}

BOOST_AUTO_TEST_CASE(rebase_seam_configuration_is_continuous) {
    // The last sample of the original chain and the first sample of the rebased chain should
    // report the same joint configuration, since both correspond to the original trajectory's
    // terminal pose by the rest-to-rest invariant.
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());

    const auto initial_duration = sess.active_trajectory()->duration();
    // Capture the terminal pose directly from the trajectory the session is about to leave.
    const auto terminal_sample = sess.active_trajectory()->sample(initial_duration);

    sess.sample_at_least(initial_duration);

    const pinned_waypoints extension(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(extension.accumulator());

    const auto post_rebase_samples = sess.sample_next(1);
    BOOST_REQUIRE_EQUAL(post_rebase_samples.size(), 1U);
    // The seam comparison means nothing unless a rebase happened, so require one first.
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 2U);

    // The first sample after the rebase sits exactly one sample period into the new
    // trajectory, because that is where its sampler starts. Its configuration therefore
    // differs from the terminal pose by the motion planned over one sample period from rest,
    // which is at most 0.5 * max_accel * sample_period^2 = 0.5 * 5.0 * 0.01^2 = 2.5e-4 rad per
    // joint, plus a little for blend curvature. A tolerance of 1e-3 leaves an order of
    // magnitude of margin, and anything tighter would be brittle.
    BOOST_CHECK(configs_match(post_rebase_samples.front().configuration, terminal_sample.configuration, 1e-3));
}

BOOST_AUTO_TEST_CASE(rebase_seam_time_keeps_flowing_forward) {
    // Across a rebase the epoch advances and the new active trajectory has a fresh
    // local-time origin. The session must add the new epoch when reporting sample
    // times so the global clock keeps moving forward rather than restarting near zero.
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    const auto initial_duration = sess.active_trajectory()->duration();
    sess.sample_at_least(initial_duration);
    const auto pre_rebase_time = sess.current_time();
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);
    const pinned_waypoints extension(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(extension.accumulator());
    const auto post_rebase_samples = sess.sample_next(1);
    BOOST_REQUIRE_EQUAL(post_rebase_samples.size(), 1U);
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 2U);
    BOOST_CHECK_GT(post_rebase_samples.front().time.count(), pre_rebase_time.count());
}

BOOST_AUTO_TEST_CASE(staged_batch_that_fails_to_build_is_rejected_by_extend) {
    // A staged batch whose geometry cannot build a path must be rejected by the extend that
    // carries it, leaving the staged motion as it was, so the caller can correct the batch and
    // send it again. This is robot motion, so a bad batch must never be silently dropped, and
    // it must not wedge the session either. Duplicate consecutive waypoints yield a zero-length
    // linear segment, which path::create rejects. We disable linear coalescing by setting the
    // maximum linear deviation to zero, so that the duplicate is not quietly removed before it
    // can throw.
    path::options popt = default_path_options();
    popt.set_max_linear_deviation(0.0);
    streaming::session sess{popt, default_trajectory_options(), default_sample_rate()};

    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());

    // Drain to the terminal, so that the next extend's branch has already been sampled and the
    // session starts staging.
    sess.sample_at_least(sess.active_trajectory()->duration());

    const pinned_waypoints good(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}});
    sess.extend(good.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);  // staged
    const auto total_before = sess.remaining_total_duration();

    // The session is already staging, so this batch joins the staged motion. Its tail {2,1}
    // duplicates the last staged waypoint, so the staged waypoints {1,1},{2,1},{2,1} carry a
    // zero-length segment, and the extend throws. The C ABI maps this to an error return.
    const pinned_waypoints bad(xmatrix<>{{2.0, 1.0}, {2.0, 1.0}});
    BOOST_CHECK_THROW(sess.extend(bad.accumulator()), std::invalid_argument);

    // The staged motion is as it was before the bad batch.
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 1U);
    BOOST_CHECK_EQUAL(sess.remaining_total_duration().count(), total_before.count());

    // A corrected batch, joined at the same seam, is accepted, and draining delivers it.
    const pinned_waypoints corrected(xmatrix<>{{2.0, 1.0}, {2.0, 2.0}});
    const auto result = sess.extend(corrected.accumulator());
    BOOST_CHECK(result.kind == streaming::session::extend_result::kinds::k_staged_again);

    const auto drained = sess.sample_at_least(trajectory::seconds{1000.0});
    BOOST_REQUIRE(!drained.empty());
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 2U);
    const xvector<> final_waypoint{2.0, 2.0};
    BOOST_CHECK(configs_match(drained.back().configuration, final_waypoint, 1e-3));
}

BOOST_AUTO_TEST_SUITE_END()  // stage_and_rebase

BOOST_AUTO_TEST_SUITE(multi_extend)

BOOST_AUTO_TEST_CASE(repeated_admissible_extends_compose_into_long_trajectory) {
    // Three extends in a row, each issued before anything has been sampled, so each pivots.
    // The resulting sample stream should agree with a direct trajectory built over the
    // fully merged waypoint set.
    auto sess = fresh_session();

    const pinned_waypoints batch_1(xmatrix<>{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}});
    sess.extend(batch_1.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);

    const pinned_waypoints batch_2(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(batch_2.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 2U);

    const pinned_waypoints batch_3(xmatrix<>{{2.0, 2.0}, {3.0, 2.0}});
    sess.extend(batch_3.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 3U);

    const xmatrix<> merged{
        {0.0, 0.0},
        {1.0, 0.0},
        {1.0, 1.0},
        {2.0, 1.0},
        {2.0, 2.0},
        {3.0, 2.0},
    };
    const auto reference = reference_trajectory(merged);

    const auto samples = sess.sample_at_least(reference.duration());
    BOOST_REQUIRE(!samples.empty());

    check_samples_match_reference(samples, reference);
}

BOOST_AUTO_TEST_CASE(mixed_pivot_and_stage_eventually_drains_all_input) {
    // Pivot once, then sample to terminal to force the next extend to stage, then sample
    // past terminal to rebase. Generation count progression: 1 (initial), 2 (pivot),
    // 2 (stage, no new active), 3 (rebase).
    auto sess = fresh_session();

    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);

    // After one sample the branch is still ahead, so this extend pivots.
    sess.sample_next(1);
    const pinned_waypoints pivot_batch(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}});
    sess.extend(pivot_batch.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 2U);

    // Sample past where the next extend's branch will lie, forcing it to stage.
    const auto duration_before_stage = sess.active_trajectory()->duration();
    sess.sample_at_least(duration_before_stage);
    const pinned_waypoints stage_batch(xmatrix<>{{2.0, 1.0}, {2.0, 2.0}});
    sess.extend(stage_batch.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 2U);  // staged, not pivoted

    sess.sample_next(1);
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 3U);
    BOOST_CHECK(sess.active_trajectory() != nullptr);
    BOOST_CHECK_EQUAL(sess.active_epoch().count(), duration_before_stage.count());
}

BOOST_AUTO_TEST_CASE(multi_batch_staging_accumulates_into_single_rebase) {
    // While the session is staging, several extends accumulate into the staged motion, and a
    // single rebase makes it active. Every other rebase test stages exactly one batch, which
    // leaves accumulation over several batches unexercised. It also backs the reasoning that
    // more input arriving before a drain keeps the rebase on its normal path rather than the
    // one for staged motion shorter than a sample period, which only holds if multiple staged
    // batches merge correctly.
    auto sess = fresh_session();

    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    const auto initial_duration = sess.active_trajectory()->duration();

    // Drain to the terminal, so that the next extend's branch has already been sampled and the
    // session starts staging.
    sess.sample_at_least(initial_duration);

    const pinned_waypoints batch_a(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    const auto first_stage = sess.extend(batch_a.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);  // staged
    BOOST_CHECK(first_stage.kind == streaming::session::extend_result::kinds::k_staged_branch_sampled);

    // The second extend arrives while the session is staging, so it joins the staged batch
    // rather than causing a rebase.
    const pinned_waypoints batch_b(xmatrix<>{{2.0, 2.0}, {3.0, 2.0}});
    const auto second_stage = sess.extend(batch_b.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);  // still just accumulated

    // The second call is compared against the staged motion it extends. Nothing of that
    // motion has been sampled, so the slack is never negative, and the growth is what batch_b
    // added to the staged motion's duration.
    BOOST_CHECK(second_stage.kind == streaming::session::extend_result::kinds::k_staged_again);
    BOOST_REQUIRE(second_stage.branch_slack.has_value());
    BOOST_CHECK_GE(second_stage.branch_slack->count(), 0.0);
    BOOST_REQUIRE(second_stage.delta_active_duration.has_value());
    const auto staged_after_a = reference_trajectory(batch_a.data()).duration();
    const auto staged_after_b = reference_trajectory(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}, {3.0, 2.0}}).duration();
    BOOST_CHECK_EQUAL(second_stage.delta_active_duration->count(), (staged_after_b - staged_after_a).count());

    // Draining fires a single rebase that makes the motion from both staged batches active.
    const auto drained = sess.sample_at_least(trajectory::seconds{1000.0});
    BOOST_REQUIRE(!drained.empty());
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 2U);  // one rebase, not one-per-batch
    BOOST_CHECK_EQUAL(sess.active_epoch().count(), initial_duration.count());

    // Both batches were taken in, since the terminal reaches batch_b's last waypoint {3,2} at
    // rest. Had accumulation dropped or misassembled batch_b, the terminal would be batch_a's
    // last, {2,2}.
    const auto& terminal = drained.back();
    const xvector<> final_waypoint{3.0, 2.0};
    BOOST_CHECK(configs_match(terminal.configuration, final_waypoint, 1e-3));
    BOOST_REQUIRE_EQUAL(terminal.velocity.shape(0), 2U);
    BOOST_REQUIRE_EQUAL(terminal.acceleration.shape(0), 2U);
    for (std::size_t i = 0; i < terminal.velocity.shape(0); ++i) {
        BOOST_CHECK_EQUAL(terminal.velocity(i), 0.0);
    }
    for (std::size_t i = 0; i < terminal.acceleration.shape(0); ++i) {
        BOOST_CHECK_EQUAL(terminal.acceleration(i), 0.0);
    }

    // TODO: strengthen this to a full geometry-equivalence check that verifies the post-rebase
    // sample stream matches a trajectory built directly over the merged waypoint set (the
    // terminal pose followed by the staged tails), rather than only checking that it reaches the
    // endpoint. That is deferred because it needs an epoch-aware comparison. Post-rebase samples
    // carry global timestamps advanced by the prior chain's duration, and
    // check_samples_match_reference skips any sample past the reference's local duration, so
    // reusing it as-is would silently skip every post-rebase sample and pass without checking
    // anything. The full check would compare reference.sample(sample.time - epoch) through a
    // small epoch-shifted variant of that helper. This minimal test covers the property that
    // matters: both batches become active at one rebase, and the terminal reaches the last
    // staged waypoint at rest.
}

BOOST_AUTO_TEST_SUITE_END()  // multi_extend

BOOST_AUTO_TEST_SUITE(start_staging)

BOOST_AUTO_TEST_CASE(start_staging_before_first_extend_has_no_effect) {
    // There is nothing to stage behind yet. If the session remembered the request, it would
    // stage the second batch, so check that the second batch pivots.
    auto sess = fresh_session();
    sess.start_staging();

    const pinned_waypoints initial(three_waypoints());
    const auto first = sess.extend(initial.accumulator());
    BOOST_CHECK(first.kind == streaming::session::extend_result::kinds::k_first_build);

    sess.sample_next(1);
    const pinned_waypoints extension(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    const auto second = sess.extend(extension.accumulator());
    BOOST_CHECK(second.kind == streaming::session::extend_result::kinds::k_pivot);
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 2U);
}

BOOST_AUTO_TEST_CASE(extend_after_start_staging_stages_without_comparing) {
    // After one sample the branch for this extension is still well ahead, so without the
    // request the extend would pivot, as the previous case shows. With it, the batch stages.
    // There is no staged motion yet to compare it against, so there is no slack, and the
    // growth is the whole of the staged motion's duration.
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    sess.sample_next(1);

    sess.start_staging();
    const pinned_waypoints extension(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    const auto result = sess.extend(extension.accumulator());

    BOOST_CHECK(result.kind == streaming::session::extend_result::kinds::k_staged_again);
    BOOST_CHECK(!result.branch_slack.has_value());
    BOOST_REQUIRE(result.delta_active_duration.has_value());
    BOOST_CHECK_EQUAL(result.delta_active_duration->count(), reference_trajectory(extension.data()).duration().count());
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 1U);
}

BOOST_AUTO_TEST_CASE(rebase_after_start_staging_restarts_the_chain_and_pivoting_resumes) {
    // The point of starting to stage is that the rebase leaves behind every waypoint before
    // the end of the active trajectory, so later pivots work over a short chain. Check that by
    // comparing durations against trajectories built directly from the short chains.
    // Construction is deterministic, so equal inputs give bit-equal durations, and a chain
    // that still carried the original waypoints would come out longer.
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    const auto initial_duration = sess.active_trajectory()->duration();
    sess.sample_next(1);

    sess.start_staging();
    const pinned_waypoints staged(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(staged.accumulator());

    // Sampling through the end of the active trajectory rebases onto the staged motion.
    sess.sample_at_least(initial_duration);
    sess.sample_next(1);
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 2U);
    BOOST_CHECK_EQUAL(sess.active_epoch().count(), initial_duration.count());
    BOOST_CHECK_EQUAL(sess.active_trajectory()->duration().count(), reference_trajectory(staged.data()).duration().count());

    // Staging ended with the rebase, so the next extend compares and pivots.
    const pinned_waypoints extension(xmatrix<>{{2.0, 2.0}, {3.0, 2.0}});
    const auto result = sess.extend(extension.accumulator());
    BOOST_CHECK(result.kind == streaming::session::extend_result::kinds::k_pivot);
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 3U);

    const xmatrix<> short_chain{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}, {3.0, 2.0}};
    BOOST_CHECK_EQUAL(sess.active_trajectory()->duration().count(), reference_trajectory(short_chain).duration().count());
}

BOOST_AUTO_TEST_CASE(start_staging_holds_after_the_active_drains) {
    // Draining the active trajectory with nothing staged does not rebase, so staging must
    // continue. The next extend therefore reports a stage without a comparison, rather than
    // being compared and found to branch in time that has already been sampled.
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    sess.sample_next(1);

    sess.start_staging();
    sess.sample_at_least(trajectory::seconds{1000.0});
    BOOST_REQUIRE(sess.sample_next(1).empty());

    const pinned_waypoints extension(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    const auto result = sess.extend(extension.accumulator());
    BOOST_CHECK(result.kind == streaming::session::extend_result::kinds::k_staged_again);
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 1U);

    sess.sample_next(1);
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 2U);
}

BOOST_AUTO_TEST_CASE(failed_first_batch_after_start_staging_leaves_nothing_staged) {
    // The first batch after start_staging() starts the staged motion from the active
    // trajectory's last waypoint. If that motion cannot be built, the extend throws and nothing
    // is staged, so a corrected batch starts the staged motion afresh. Duplicate consecutive
    // waypoints yield a zero-length linear segment, which path::create rejects once linear
    // coalescing is disabled.
    path::options popt = default_path_options();
    popt.set_max_linear_deviation(0.0);
    streaming::session sess{popt, default_trajectory_options(), default_sample_rate()};

    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    sess.sample_next(1);
    sess.start_staging();

    const pinned_waypoints bad(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 1.0}});
    BOOST_CHECK_THROW(sess.extend(bad.accumulator()), std::invalid_argument);
    BOOST_CHECK_EQUAL(sess.remaining_total_duration().count(), sess.remaining_active_duration().count());

    // Still the first batch to be staged, so still nothing to compare against.
    const pinned_waypoints corrected(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    const auto result = sess.extend(corrected.accumulator());
    BOOST_CHECK(result.kind == streaming::session::extend_result::kinds::k_staged_again);
    BOOST_CHECK(!result.branch_slack.has_value());
    BOOST_REQUIRE(result.delta_active_duration.has_value());
    BOOST_CHECK_EQUAL(result.delta_active_duration->count(), reference_trajectory(corrected.data()).duration().count());

    const auto drained = sess.sample_at_least(trajectory::seconds{1000.0});
    BOOST_REQUIRE(!drained.empty());
    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 2U);
    const xvector<> final_waypoint{2.0, 2.0};
    BOOST_CHECK(configs_match(drained.back().configuration, final_waypoint, 1e-3));
}

BOOST_AUTO_TEST_SUITE_END()  // start_staging

BOOST_AUTO_TEST_SUITE(staged_motion)

BOOST_AUTO_TEST_CASE(staged_extend_reports_slack_and_growth_against_the_staged_motion) {
    // An extend that joins staged motion is compared against that motion, which starts where
    // the active trajectory ends. None of it has been sampled, so the branch is never earlier
    // than the active trajectory's end, and the slack is at least the active trajectory's
    // unsampled remainder. The growth is what the batch added to the staged motion's duration.
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    sess.sample_next(1);
    sess.start_staging();

    const pinned_waypoints first(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}});
    sess.extend(first.accumulator());

    const pinned_waypoints second(xmatrix<>{{2.0, 1.0}, {2.0, 2.0}});
    const auto result = sess.extend(second.accumulator());
    BOOST_CHECK(result.kind == streaming::session::extend_result::kinds::k_staged_again);
    BOOST_REQUIRE(result.branch_slack.has_value());
    BOOST_CHECK_GE(result.branch_slack->count(), sess.remaining_active_duration().count());

    const auto before = reference_trajectory(first.data()).duration();
    const auto after = reference_trajectory(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}}).duration();
    BOOST_REQUIRE(result.delta_active_duration.has_value());
    BOOST_CHECK_EQUAL(result.delta_active_duration->count(), (after - before).count());
}

BOOST_AUTO_TEST_CASE(staged_motion_built_over_several_extends_matches_a_direct_build) {
    // Staged motion grows batch by batch, and once it becomes active it must be the trajectory
    // a direct build over the same waypoints produces. Construction is deterministic, so equal
    // waypoints give bit-equal durations.
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    const auto initial_duration = sess.active_trajectory()->duration();
    sess.sample_next(1);
    sess.start_staging();

    const pinned_waypoints a(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}});
    const pinned_waypoints b(xmatrix<>{{2.0, 1.0}, {2.0, 2.0}});
    const pinned_waypoints c(xmatrix<>{{2.0, 2.0}, {3.0, 2.0}});
    sess.extend(a.accumulator());
    sess.extend(b.accumulator());
    sess.extend(c.accumulator());

    // The two remainders are differences of global times summed in different orders, so they
    // agree with the staged duration only to within rounding.
    const xmatrix<> staged{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}, {3.0, 2.0}};
    const auto staged_duration = reference_trajectory(staged).duration();
    BOOST_CHECK_SMALL((sess.remaining_total_duration() - sess.remaining_active_duration() - staged_duration).count(), 1e-12);

    // Sampling through the active trajectory's end makes the staged motion active.
    sess.sample_at_least(initial_duration);
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 2U);
    BOOST_CHECK_EQUAL(sess.active_trajectory()->duration().count(), staged_duration.count());
}

BOOST_AUTO_TEST_SUITE_END()  // staged_motion

BOOST_AUTO_TEST_SUITE(seam)

BOOST_AUTO_TEST_CASE(staged_motion_supplies_the_sample_at_the_seam) {
    // When motion is staged before the active trajectory's terminal goes out, the sample at the
    // seam comes from the staged motion's trajectory. It sits at the same instant and pose as
    // the terminal would have, with zero velocity, but carries the acceleration the new
    // trajectory starts with rather than the zero acceleration the old one ends with.
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    const auto initial_duration = sess.active_trajectory()->duration();
    sess.sample_next(1);

    sess.start_staging();
    const pinned_waypoints staged(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(staged.accumulator());

    // The first sample to reach the old trajectory's duration is the one at the seam.
    const auto samples = sess.sample_at_least(initial_duration);
    BOOST_REQUIRE(!samples.empty());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 2U);
    const auto& seam = samples.back();
    BOOST_REQUIRE_EQUAL(seam.time.count(), initial_duration.count());

    const auto expected = reference_trajectory(staged.data()).sample(trajectory::seconds{0.0});
    BOOST_CHECK(configs_match(seam.configuration, expected.configuration));
    BOOST_CHECK(configs_match(seam.velocity, expected.velocity));
    BOOST_CHECK(configs_match(seam.acceleration, expected.acceleration));

    double peak_acceleration = 0.0;
    for (std::size_t i = 0; i < seam.acceleration.shape(0); ++i) {
        BOOST_CHECK_EQUAL(seam.velocity(i), 0.0);
        peak_acceleration = std::max(peak_acceleration, std::abs(seam.acceleration(i)));
    }
    BOOST_CHECK_GT(peak_acceleration, 0.0);

    // Every sample before the seam came from the old trajectory, and the next one moves on into
    // the new trajectory rather than repeating the seam's time.
    for (std::size_t i = 0; i + 1 < samples.size(); ++i) {
        BOOST_CHECK_LT(samples[i].time.count(), seam.time.count());
    }
    const auto after = sess.sample_next(1);
    BOOST_REQUIRE_EQUAL(after.size(), 1U);
    BOOST_CHECK_GT(after.front().time.count(), seam.time.count());

    // The seam sample is the new trajectory's own first sample, so it respects the velocity and
    // acceleration limits by construction. Checking it and its neighbors on either side against
    // those limits states that intent, and would catch a seam that stitched in a value belonging
    // to neither trajectory. The tolerance allows for the last few bits of rounding in a sample
    // taken at the limit.
    BOOST_REQUIRE_GE(samples.size(), 2U);
    const auto limits = default_trajectory_options();
    const auto& max_velocity = limits.max_velocity.get();
    const auto& max_acceleration = limits.max_acceleration.get();
    constexpr double k_limit_tolerance = 1e-9;
    for (const auto* s : {&samples[samples.size() - 2], &seam, &after.front()}) {
        for (std::size_t i = 0; i < s->velocity.shape(0); ++i) {
            BOOST_CHECK_LE(std::abs(s->velocity(i)), max_velocity(i) + k_limit_tolerance);
            BOOST_CHECK_LE(std::abs(s->acceleration(i)), max_acceleration(i) + k_limit_tolerance);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()  // seam

BOOST_AUTO_TEST_SUITE(end_of_stream)

BOOST_AUTO_TEST_CASE(sample_next_after_exhaustion_returns_empty) {
    auto sess = fresh_session();
    const pinned_waypoints wp(three_waypoints());
    sess.extend(wp.accumulator());

    const auto* active = sess.active_trajectory();
    BOOST_REQUIRE(active != nullptr);
    sess.sample_at_least(active->duration() * 2.0);  // sample well past terminal

    // Nothing is staged, so further pulls return nothing.
    const auto samples = sess.sample_next(5);
    BOOST_CHECK(samples.empty());
}

BOOST_AUTO_TEST_CASE(extend_after_exhaustion_eventually_starts_new_chain) {
    // When the active trajectory is exhausted and nothing is staged, an arriving extend
    // stages. The next sample then triggers a rebase from the terminal pose, which makes a new
    // trajectory active and advances the epoch.
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);

    const auto initial_duration = sess.active_trajectory()->duration();

    sess.sample_at_least(initial_duration * 2.0);  // drain to empty

    const pinned_waypoints extension(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(extension.accumulator());

    // An extend on an exhausted session stages, so no new trajectory is active yet.
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);

    sess.sample_next(1);

    BOOST_CHECK_EQUAL(sess.trajectory_generation_count(), 2U);
    BOOST_CHECK(sess.active_trajectory() != nullptr);
    BOOST_CHECK_GE(sess.active_epoch().count(), initial_duration.count());
}

BOOST_AUTO_TEST_SUITE_END()  // end_of_stream

// These tests pin down a property that uniform_sampler's quantized-for-duration mode is
// supposed to give us at the trajectory level, and that the session needs to preserve
// across rebases: the last emitted sample lands exactly at the active trajectory's
// terminal, with zero joint velocity and zero joint acceleration by the rest-to-rest
// invariant.
BOOST_AUTO_TEST_SUITE(terminal_sampling)

BOOST_AUTO_TEST_CASE(final_emitted_sample_in_single_trajectory_lies_at_terminal_at_rest) {
    auto sess = fresh_session();
    const pinned_waypoints wp(six_waypoints());
    sess.extend(wp.accumulator());

    const auto duration = sess.active_trajectory()->duration();
    const auto samples = sess.sample_at_least(duration);
    BOOST_REQUIRE(!samples.empty());

    const auto& last = samples.back();

    BOOST_CHECK_EQUAL(last.time.count(), duration.count());
    BOOST_REQUIRE_EQUAL(last.velocity.shape(0), 2U);
    BOOST_REQUIRE_EQUAL(last.acceleration.shape(0), 2U);
    for (std::size_t i = 0; i < last.velocity.shape(0); ++i) {
        BOOST_CHECK_EQUAL(last.velocity(i), 0.0);
    }
    for (std::size_t i = 0; i < last.acceleration.shape(0); ++i) {
        BOOST_CHECK_EQUAL(last.acceleration(i), 0.0);
    }
}

BOOST_AUTO_TEST_CASE(final_emitted_sample_after_rebase_lies_at_rebased_terminal_at_rest) {
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());

    const auto initial_duration = sess.active_trajectory()->duration();
    sess.sample_at_least(initial_duration);  // drain the initial chain through its terminal

    // Extending after sampling to the terminal stages the extension.
    const pinned_waypoints extension(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(extension.accumulator());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 1U);

    // Drain the rest with a generous horizon to fire the rebase and run out the new chain.
    const auto post_rebase_samples = sess.sample_at_least(trajectory::seconds{1000.0});
    BOOST_REQUIRE(!post_rebase_samples.empty());
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 2U);

    const auto& last = post_rebase_samples.back();
    const auto rebased_duration = sess.active_trajectory()->duration();
    const auto rebased_epoch = sess.active_epoch();

    BOOST_CHECK_EQUAL(last.time.count(), (rebased_epoch + rebased_duration).count());
    BOOST_REQUIRE_EQUAL(last.velocity.shape(0), 2U);
    BOOST_REQUIRE_EQUAL(last.acceleration.shape(0), 2U);
    for (std::size_t i = 0; i < last.velocity.shape(0); ++i) {
        BOOST_CHECK_EQUAL(last.velocity(i), 0.0);
    }
    for (std::size_t i = 0; i < last.acceleration.shape(0); ++i) {
        BOOST_CHECK_EQUAL(last.acceleration(i), 0.0);
    }
}

BOOST_AUTO_TEST_SUITE_END()  // terminal_sampling

BOOST_AUTO_TEST_SUITE(remaining_duration)

BOOST_AUTO_TEST_CASE(remaining_active_duration_is_zero_without_a_trajectory) {
    auto sess = fresh_session();
    BOOST_CHECK_EQUAL(sess.remaining_active_duration().count(), 0.0);
}

BOOST_AUTO_TEST_CASE(remaining_active_duration_is_measured_in_global_time) {
    // The remainder is the active trajectory's end in global time less the most recently
    // emitted sample. While the epoch is still zero a local-frame computation agrees with a
    // global-frame one by coincidence, so the distinction only shows up after a rebase has
    // advanced the epoch -- at which point a local-frame implementation reports a negative
    // remainder and keeps doing so for the rest of the session. Drive past a rebase and check
    // it there, because that is the case the accessor exists to get right.
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());

    // Nothing emitted yet, so the whole active trajectory remains.
    BOOST_CHECK_EQUAL(sess.remaining_active_duration().count(), sess.active_trajectory()->duration().count());

    // Drain to the terminal, stage a batch, then sample once more to fire the rebase.
    sess.sample_at_least(sess.active_trajectory()->duration());
    const pinned_waypoints extension(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(extension.accumulator());
    sess.sample_next(1);
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 2U);
    BOOST_REQUIRE_GT(sess.active_epoch().count(), 0.0);

    const auto remaining = sess.remaining_active_duration();
    BOOST_CHECK_GT(remaining.count(), 0.0);
    BOOST_CHECK_EQUAL(remaining.count(), (sess.active_epoch() + sess.active_trajectory()->duration() - sess.current_time()).count());

    // Draining the rebased chain takes it to zero, and the clamp keeps it from going below.
    sess.sample_at_least(trajectory::seconds{1000.0});
    BOOST_CHECK_GE(sess.remaining_active_duration().count(), 0.0);
    BOOST_CHECK_SMALL(sess.remaining_active_duration().count(), 1e-9);
}

BOOST_AUTO_TEST_CASE(remaining_total_duration_is_zero_without_a_trajectory) {
    auto sess = fresh_session();
    BOOST_CHECK_EQUAL(sess.remaining_total_duration().count(), 0.0);
}

BOOST_AUTO_TEST_CASE(remaining_total_duration_counts_staged_motion) {
    // With nothing staged the total is the active remainder. Staged motion adds its whole
    // duration, since none of it has been sampled, and once it becomes active the total is
    // the active remainder again. The comparisons allow for rounding, because the remainders
    // are differences of global times summed in different orders.
    auto sess = fresh_session();
    const pinned_waypoints initial(three_waypoints());
    sess.extend(initial.accumulator());
    const auto initial_duration = sess.active_trajectory()->duration();
    sess.sample_next(1);
    BOOST_CHECK_EQUAL(sess.remaining_total_duration().count(), sess.remaining_active_duration().count());

    sess.start_staging();
    BOOST_CHECK_EQUAL(sess.remaining_total_duration().count(), sess.remaining_active_duration().count());

    const pinned_waypoints staged(xmatrix<>{{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}});
    sess.extend(staged.accumulator());
    const auto staged_duration = reference_trajectory(staged.data()).duration();
    BOOST_CHECK_SMALL((sess.remaining_total_duration() - sess.remaining_active_duration() - staged_duration).count(), 1e-12);

    sess.sample_at_least(initial_duration);
    BOOST_REQUIRE_EQUAL(sess.trajectory_generation_count(), 2U);
    BOOST_CHECK_EQUAL(sess.remaining_total_duration().count(), sess.remaining_active_duration().count());
}

BOOST_AUTO_TEST_SUITE_END()  // remaining_duration

BOOST_AUTO_TEST_SUITE_END()  // streaming_session_tests
