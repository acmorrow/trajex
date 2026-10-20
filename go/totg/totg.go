//go:build !windows

// Package totg exposes the trajex TOTG (time-optimal trajectory generation)
// algorithm as a Go API, layered on top of github.com/viam-modules/trajex's
// TensorMap.
package totg

/*
#cgo CFLAGS: -I${SRCDIR}/../artifacts/include

#include <viam/trajex/capi/capi.h>
*/
import "C"

import (
	"context"
	"unsafe"

	"github.com/pkg/errors"

	trajex "github.com/viam-modules/trajex/go"
	"github.com/viam-modules/trajex/go/internal/capi"
)

// Schema key string constants for inputs and outputs honored by Generate.
// Initialized at package init from the cgo-exposed viam_trajex_totg_key_*
// externs in capi.h so the strings cannot drift between Go and C without a
// recompile. Use these in place of hard-coded string literals.
var (
	KeyWaypointsRads                      string //nolint:revive
	KeyVelocityLimitsRadsPerSec           string //nolint:revive
	KeyAccelerationLimitsRadsPerSec2      string //nolint:revive
	KeyPathToleranceDeltaRads             string //nolint:revive
	KeyPathColinearizationRatio           string //nolint:revive
	KeyWaypointDeduplicationToleranceRads string //nolint:revive
	KeyTrajectorySamplingFreqHz           string //nolint:revive
	KeySampleTimesSec                     string //nolint:revive
	KeyConfigurationsRads                 string //nolint:revive
	KeyVelocitiesRadsPerSec               string //nolint:revive
	KeyAccelerationsRadsPerSec2           string //nolint:revive
	KeyTCPMaxLinearSpeed                  string //nolint:revive
)

func init() {
	KeyWaypointsRads = capi.CStr(unsafe.Pointer(&C.viam_trajex_totg_key_waypoints_rads))
	KeyVelocityLimitsRadsPerSec = capi.CStr(unsafe.Pointer(&C.viam_trajex_totg_key_velocity_limits_rads_per_sec))
	KeyAccelerationLimitsRadsPerSec2 = capi.CStr(unsafe.Pointer(&C.viam_trajex_totg_key_acceleration_limits_rads_per_sec2))
	KeyPathToleranceDeltaRads = capi.CStr(unsafe.Pointer(&C.viam_trajex_totg_key_path_tolerance_delta_rads))
	KeyPathColinearizationRatio = capi.CStr(unsafe.Pointer(&C.viam_trajex_totg_key_path_colinearization_ratio))
	KeyWaypointDeduplicationToleranceRads = capi.CStr(unsafe.Pointer(&C.viam_trajex_totg_key_waypoint_deduplication_tolerance_rads))
	KeyTrajectorySamplingFreqHz = capi.CStr(unsafe.Pointer(&C.viam_trajex_totg_key_trajectory_sampling_freq_hz))
	KeySampleTimesSec = capi.CStr(unsafe.Pointer(&C.viam_trajex_totg_key_sample_times_sec))
	KeyConfigurationsRads = capi.CStr(unsafe.Pointer(&C.viam_trajex_totg_key_configurations_rads))
	KeyVelocitiesRadsPerSec = capi.CStr(unsafe.Pointer(&C.viam_trajex_totg_key_velocities_rads_per_sec))
	KeyAccelerationsRadsPerSec2 = capi.CStr(unsafe.Pointer(&C.viam_trajex_totg_key_accelerations_rads_per_sec2))
	KeyTCPMaxLinearSpeed = capi.CStr(unsafe.Pointer(&C.viam_trajex_totg_key_tcp_max_linear_speed))
}

// Generate computes a time-parameterized trajectory from the named-tensor
// inputs map, writing results into outputs. See the trajex C ABI header
// for the input/output schema; the key string constants in this package
// (KeyWaypointsRads, etc.) name the keys.
//
// inputs and outputs must be distinct, non-closed TensorMap handles.
//
// Generate honors ctx at entry and exit: if ctx is already cancelled when
// Generate is called, it returns ctx.Err() without invoking the C ABI; if
// ctx is cancelled while the C call is in flight (which itself cannot be
// interrupted), the result is discarded and ctx.Err() is returned.
func Generate(ctx context.Context, inputs, outputs *trajex.TensorMap) error {
	return generate(ctx, nil, inputs, outputs)
}

// GenerateWithKinematics behaves like Generate, but also takes a kinematic
// chain describing the arm, which is what allows inputs to carry limits such
// as KeyTCPMaxLinearSpeed. Having kinematics is a property of the caller,
// while wanting a cap is a property of a particular call, so a caller with a
// chain may pass it on every call. When inputs carry no such limit, the
// trajectory is exactly the one Generate would produce.
//
// The chain must not be nil, must not already have been released with Close,
// and its number of rotational and translational joints must equal the
// number of degrees of freedom in the waypoints. The chain is only read during the call, so it may be released as
// soon as the call returns.
func GenerateWithKinematics(ctx context.Context, chain *trajex.KinematicChain, inputs, outputs *trajex.TensorMap) error {
	// A null chain would reach the C ABI as an ordinary request without
	// kinematics. That is harmless when no cap was asked for, but it hides a
	// mistake on the caller's part, so we refuse it here instead.
	chainHandle := (*C.viam_trajex_kinematic_chain_t)(chain.UnsafeHandle())
	if chainHandle == nil {
		return errors.New("trajex/totg: GenerateWithKinematics requires a kinematic chain that is not nil and has not been released")
	}
	return generate(ctx, chainHandle, inputs, outputs)
}

// Generate and GenerateWithKinematics both end up here. They are separate
// functions, rather than one taking a chain that may be nil, because a bare
// nil in an argument list tells someone reading the call nothing about what
// it means. The streaming package makes the same split for its constructors.
func generate(ctx context.Context, chain *C.viam_trajex_kinematic_chain_t, inputs, outputs *trajex.TensorMap) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	inHandle := (*C.viam_trajex_tensor_map_t)(inputs.UnsafeHandle())
	outHandle := (*C.viam_trajex_tensor_map_t)(outputs.UnsafeHandle())

	var errOut *C.char
	rc := C.viam_trajex_totg_generate(chain, inHandle, outHandle, &errOut)
	if rc != 0 {
		msg := C.GoString(errOut)
		C.viam_trajex_string_destroy(errOut)
		return errors.Errorf("trajex/totg: generate failed: %s", msg)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return nil
}
