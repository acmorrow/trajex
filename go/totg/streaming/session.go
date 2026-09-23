//go:build !windows

// Package streaming exposes the trajex TOTG streaming session as a Go API,
// layered on top of github.com/viam-modules/trajex's TensorMap. Sessions are
// stateful: callers construct a session with a fixed configuration, extend it
// with waypoint batches over time, and pull samples incrementally.
//
// The schema-key constants (KeyWaypointsRads, KeyVelocityLimitsRadsPerSec,
// etc.) are re-exported from the parent totg package so callers using both
// stateless Generate and streaming sessions need to import only one set of
// key names.
package streaming

/*
#cgo CFLAGS: -I${SRCDIR}/../../artifacts/include

#include <viam/trajex/capi/capi.h>
*/
import "C"

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/pkg/errors"

	trajex "github.com/viam-modules/trajex/go"
	"github.com/viam-modules/trajex/go/totg"
)

// Re-export the shared schema-key constants from totg so streaming callers
// have a single source of truth. The values are identical to totg.Key* by
// construction (both initialized from the same C externs at package init).
var (
	KeyVelocityLimitsRadsPerSec      = totg.KeyVelocityLimitsRadsPerSec      //nolint:revive
	KeyAccelerationLimitsRadsPerSec2 = totg.KeyAccelerationLimitsRadsPerSec2 //nolint:revive
	KeyPathToleranceDeltaRads        = totg.KeyPathToleranceDeltaRads        //nolint:revive
	KeyPathColinearizationRatio      = totg.KeyPathColinearizationRatio      //nolint:revive
	KeyTrajectorySamplingFreqHz      = totg.KeyTrajectorySamplingFreqHz      //nolint:revive
	KeyWaypointsRads                 = totg.KeyWaypointsRads                 //nolint:revive
	KeySampleTimesSec                = totg.KeySampleTimesSec                //nolint:revive
	KeyConfigurationsRads            = totg.KeyConfigurationsRads            //nolint:revive
	KeyVelocitiesRadsPerSec          = totg.KeyVelocitiesRadsPerSec          //nolint:revive
	KeyAccelerationsRadsPerSec2      = totg.KeyAccelerationsRadsPerSec2      //nolint:revive
)

// Session is a Go-owned handle to a CAPI streaming session. Close must be
// called (typically via defer) to release the underlying C resource. Session
// is not safe for concurrent use; concurrent operations on the same handle
// are the caller's responsibility, matching the C ABI's contract.
type Session struct {
	handle *C.viam_trajex_totg_streaming_session_t
}

// New constructs a streaming session from a configuration tensor map. The
// options map carries the velocity / acceleration limits, path tolerance,
// sample rate, and optional path colinearization ratio; see the C ABI header
// for the full schema. The options map is read-only during construction and
// may be closed by the caller as soon as New returns.
func New(options *trajex.TensorMap) (*Session, error) {
	optHandle := (*C.viam_trajex_tensor_map_t)(options.UnsafeHandle())
	var errOut *C.char
	h := C.viam_trajex_totg_streaming_session_create(optHandle, &errOut)
	if h == nil {
		msg := C.GoString(errOut)
		C.viam_trajex_string_destroy(errOut)
		return nil, errors.Errorf("trajex/totg/streaming: New failed: %s", msg)
	}
	return &Session{handle: h}, nil
}

// Close releases the underlying C session. Safe to call on a nil receiver
// and idempotent: subsequent calls are no-ops.
func (s *Session) Close() {
	if s == nil || s.handle == nil {
		return
	}
	C.viam_trajex_totg_streaming_session_destroy(s.handle)
	s.handle = nil
}

// ExtendKind describes how Extend handled a batch. The values match the
// VIAM_TRAJEX_TOTG_STREAMING_SESSION_EXTEND_* constants in the C ABI and the
// session::extend_result::kinds enumerators in C++.
//
// A batch either builds the session's first trajectory, replaces the active
// trajectory with one that incorporates it, or waits in staging until the
// active trajectory has been sampled through. The difference matters to a
// caller pacing its own sends: a pivot is invisible to the arm, but every
// trajectory ends at rest, so a stage means the active trajectory will run to
// its end and bring the arm to a stop before the staged motion begins.
//
// The two values for a stage that followed a comparison distinguish the reasons
// a pivot was refused. One says the call arrived after the point it needed to
// change had already been handed out; the other says it arrived in time but
// carried less than one sample period of motion. The remedies differ, so the
// values do too.
type ExtendKind int32

const (
	// ExtendFirstBuild means the batch built the session's first trajectory.
	// Nothing existed to branch from, so no branch slack is reported.
	ExtendFirstBuild ExtendKind = 0

	// ExtendPivot means a trajectory incorporating the batch replaced the
	// active one, and sampling continues across the change without a
	// discontinuity.
	ExtendPivot ExtendKind = 1

	// ExtendStagedBranchSampled means sampling had already passed the branch,
	// so adopting the candidate would have contradicted samples already handed
	// out and the batch was staged. The branch slack is how much sooner the
	// call needed to arrive.
	ExtendStagedBranchSampled ExtendKind = 2

	// ExtendStagedUnsamplable means the branch lay ahead of the last emitted
	// sample but the candidate ended before the next one was due, so pivoting
	// would have produced nothing to emit and the batch was staged. The batch
	// carried less than one sample period of motion.
	ExtendStagedUnsamplable ExtendKind = 3

	// ExtendStagedAgain means batches were already staged, so no candidate was
	// built and this one joined them. Nothing was compared, so there is no
	// branch slack. It can only follow one of the other two staged kinds.
	ExtendStagedAgain ExtendKind = 4

	// ExtendNoop means the batch carried nothing beyond the seam waypoint, so
	// the session is unchanged.
	ExtendNoop ExtendKind = 5
)

// String renders the kind in the snake_case spelling shared with the C++ and C
// names, so that logs and metric labels read the same way across the three
// layers.
func (d ExtendKind) String() string {
	switch d {
	case ExtendFirstBuild:
		return "first_build"
	case ExtendPivot:
		return "pivot"
	case ExtendStagedBranchSampled:
		return "staged_branch_sampled"
	case ExtendStagedUnsamplable:
		return "staged_unsamplable"
	case ExtendStagedAgain:
		return "staged_again"
	case ExtendNoop:
		return "noop"
	default:
		return fmt.Sprintf("ExtendKind(%d)", int32(d))
	}
}

// ExtendResult reports what one Extend call did and the timing it produced
// along the way. Both times are pointers because they only exist on some paths
// through Extend: a branch slack needs a candidate trajectory to compare
// against the active one, which a seam-only call, or one arriving when batches
// are already staged, never builds, and a duration delta needs a trajectory to
// have been installed, which staging by definition does not do. A pivot is the
// only kind that reports both.
type ExtendResult struct {
	// Kind is how the batch was handled.
	Kind ExtendKind

	// BranchSlack is how far the branch sits from the most recently emitted
	// sample, the branch being the point at which the candidate first stops
	// agreeing with the active trajectory. Positive means the branch was still
	// ahead of everything handed out and the call beat the deadline by that
	// much; negative means it sat in the already-emitted past, which is what
	// forces a stage, and the magnitude is how much earlier the call needed to
	// happen. Note that the comparison is against what the session has emitted,
	// not what the arm has executed, so a caller that pulls samples far ahead
	// of execution spends its own slack doing so. Nil for ExtendFirstBuild,
	// ExtendStagedAgain and ExtendNoop.
	BranchSlack *time.Duration

	// DeltaActiveDuration is how much longer the newly installed trajectory is
	// than the one it replaced. Comparing it against the
	// interval between calls says whether the caller is adding motion faster
	// than sampling consumes it. It can in principle be negative, because the
	// replacement no longer has to stop at the old terminal waypoint and so
	// covers the shared part of the path faster than its predecessor did. Set
	// only for ExtendFirstBuild, where it is the whole of the new trajectory's
	// duration, and ExtendPivot.
	DeltaActiveDuration *time.Duration
}

// Extend appends a waypoint batch to the session. The batch must contain a
// waypoints_rads tensor of shape [n_waypoints, n_dof]; on calls after the
// first, batch[0] must compare bit-exactly equal to the session's most
// recently stored waypoint (the seam contract).
//
// Extend honors ctx at entry only: if ctx is already cancelled when Extend is
// called, it returns ctx.Err() without invoking the C ABI. Once the C call
// begins it cannot be interrupted, so a cancellation landing mid-call does not
// abort it; the operation runs to completion and its result (including any
// committed state mutation) is always reported.
//
// The returned ExtendResult is meaningful only when the error is nil. A call
// that failed has no outcome to describe, so it returns the zero value, which
// carries no information and should not be inspected.
func (s *Session) Extend(ctx context.Context, batch *trajex.TensorMap) (ExtendResult, error) {
	if err := ctx.Err(); err != nil {
		return ExtendResult{}, err
	}
	batchHandle := (*C.viam_trajex_tensor_map_t)(batch.UnsafeHandle())
	var kind C.viam_trajex_totg_streaming_session_extend_kind_t
	var branchSlackSec C.double
	var deltaActiveDurationSec C.double
	var errOut *C.char
	rc := C.viam_trajex_totg_streaming_session_extend(
		s.handle, batchHandle, &kind, &branchSlackSec, &deltaActiveDurationSec, &errOut)
	if rc != 0 {
		msg := C.GoString(errOut)
		C.viam_trajex_string_destroy(errOut)
		return ExtendResult{}, errors.Errorf("trajex/totg/streaming: Extend failed: %s", msg)
	}
	return ExtendResult{
		Kind:                ExtendKind(kind),
		BranchSlack:         optionalSeconds(branchSlackSec),
		DeltaActiveDuration: optionalSeconds(deltaActiveDurationSec),
	}, nil
}

// optionalSeconds converts a C ABI duration into a Go duration, treating the
// NaN the C layer writes for an absent value as nil. The NaN convention exists
// so that a caller need not initialize the out parameters and so that a real
// zero stays distinguishable from nothing at all; it is not visible past here.
func optionalSeconds(sec C.double) *time.Duration {
	f := float64(sec)
	if math.IsNaN(f) {
		return nil
	}
	d := time.Duration(f * float64(time.Second))
	return &d
}

// SampleNext pulls up to n samples into outputs. The output map's prior
// contents are replaced. If the session is exhausted, outputs carries
// zero-length sample tensors.
//
// Honors ctx like Extend.
func (s *Session) SampleNext(ctx context.Context, n int, outputs *trajex.TensorMap) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	outHandle := (*C.viam_trajex_tensor_map_t)(outputs.UnsafeHandle())
	var errOut *C.char
	rc := C.viam_trajex_totg_streaming_session_sample_next(s.handle, C.size_t(n), outHandle, &errOut)
	if rc != 0 {
		msg := C.GoString(errOut)
		C.viam_trajex_string_destroy(errOut)
		return errors.Errorf("trajex/totg/streaming: SampleNext failed: %s", msg)
	}
	return nil
}

// SampleAtLeast pulls samples until the most recent sample's time is at least
// CurrentTime + horizon, writing them into outputs.
//
// Honors ctx like Extend.
func (s *Session) SampleAtLeast(ctx context.Context, horizon time.Duration, outputs *trajex.TensorMap) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	outHandle := (*C.viam_trajex_tensor_map_t)(outputs.UnsafeHandle())
	var errOut *C.char
	rc := C.viam_trajex_totg_streaming_session_sample_at_least(s.handle, C.double(horizon.Seconds()), outHandle, &errOut)
	if rc != 0 {
		msg := C.GoString(errOut)
		C.viam_trajex_string_destroy(errOut)
		return errors.Errorf("trajex/totg/streaming: SampleAtLeast failed: %s", msg)
	}
	return nil
}

// CurrentTime returns the global time of the most recently emitted sample,
// or zero if no samples have been emitted yet.
func (s *Session) CurrentTime() time.Duration {
	var out C.double
	C.viam_trajex_totg_streaming_session_current_time_sec(s.handle, &out)
	return time.Duration(float64(out) * float64(time.Second))
}

// GenerationCount returns the cumulative number of trajectories the session
// has installed as active (first build + each pivot + each rebase). Zero for
// a fresh session before the first Extend.
func (s *Session) GenerationCount() int64 {
	var out C.int64_t
	C.viam_trajex_totg_streaming_session_generation_count(s.handle, &out)
	return int64(out)
}

// HasActiveTrajectory reports whether the session has an active trajectory.
// False iff fresh (no successful Extend has occurred yet).
func (s *Session) HasActiveTrajectory() bool {
	var out C.int
	C.viam_trajex_totg_streaming_session_has_active_trajectory(s.handle, &out)
	return out != 0
}

// ActiveDuration returns the duration of the active trajectory. Returns zero
// when no active trajectory is present.
//
// This is the trajectory's own length, measured from its own origin, and not a
// position in the session's global time. Subtracting CurrentTime from it
// therefore means nothing once the session has rebased at least once, since the
// two are then expressed in different frames. For the unsampled remainder, use
// RemainingActiveDuration.
func (s *Session) ActiveDuration() time.Duration {
	var out C.double
	C.viam_trajex_totg_streaming_session_active_duration_sec(s.handle, &out)
	return time.Duration(float64(out) * float64(time.Second))
}

// RemainingActiveDuration returns how much of the active trajectory has not yet
// been sampled, or zero if there is none.
//
// This counts only the active trajectory. Motion sitting in staged batches has
// no trajectory yet, and so has no duration to report, which means this value
// drains toward zero while batches are staged even though the session still has
// work queued, and then jumps back up when the rebase builds a trajectory for
// that work. A caller pacing itself against this number needs to know that.
func (s *Session) RemainingActiveDuration() time.Duration {
	var out C.double
	C.viam_trajex_totg_streaming_session_remaining_active_duration_sec(s.handle, &out)
	return time.Duration(float64(out) * float64(time.Second))
}
