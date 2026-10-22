//go:build !windows

// Package kinematics provides Go bindings for the kinematic chains of the
// trajex C ABI. A Chain describes an arm so that limits can be placed on how
// its tool moves, and is passed to the algorithm entry points that accept
// one, such as totg.GenerateWithKinematics.
package kinematics

/*
#cgo CFLAGS: -I${SRCDIR}/../artifacts/include

#include <viam/trajex/capi/capi.h>
*/
import "C"

import (
	"unsafe"

	"github.com/pkg/errors"

	// Nothing here is used by name, but this package calls the C ABI, so it
	// must depend on the package that carries the link directives for it.
	_ "github.com/viam-modules/trajex/go/internal/capi"
)

// jointKind records which constructor made a Joint, which is how
// NewChain knows which C builder call to make for it. The
// enumeration starts with an unset value so that a zero Joint, which no
// constructor produced, can be recognized and refused rather than quietly
// treated as some particular kind of joint.
type jointKind int

const (
	jointKindUnset jointKind = iota
	jointKindFixed
	jointKindRotational
	jointKindTranslational
)

// Joint describes one joint of a serial kinematic chain, and is made with
// RotationalJoint, TranslationalJoint, or FixedJoint. Its fields are
// unexported so that those constructors are the only way to describe a joint.
//
// Each joint has an origin, which is the rigid transform from the previous
// joint's frame into this one, and which is applied before the joint moves.
// The origin is given as a translation xyz and a rotation wxyz, which is a
// quaternion in w, x, y, z order using the Hamilton convention. Neither the
// quaternion nor an axis needs to be of unit length.
type Joint struct {
	kind jointKind
	xyz  [3]float64
	wxyz [4]float64
	axis [3]float64
}

// RotationalJoint describes a joint that rotates about axis, which is
// expressed in the joint's own frame, and whose position is an angle in
// radians. Revolute and continuous joints are both described this way, since
// the chain carries no position limits and so has no way to tell them apart.
func RotationalJoint(xyz [3]float64, wxyz [4]float64, axis [3]float64) Joint {
	return Joint{kind: jointKindRotational, xyz: xyz, wxyz: wxyz, axis: axis}
}

// TranslationalJoint describes a prismatic joint that slides along axis,
// which is expressed in the joint's own frame. Its position is a
// displacement, in whatever length unit the chain's translations use.
func TranslationalJoint(xyz [3]float64, wxyz [4]float64, axis [3]float64) Joint {
	return Joint{kind: jointKindTranslational, xyz: xyz, wxyz: wxyz, axis: axis}
}

// FixedJoint describes a rigid transform with no motion of its own. It is how
// a chain is carried past its last moving joint, for instance from a flange
// out to the tool center point that a speed cap should apply to.
func FixedJoint(xyz [3]float64, wxyz [4]float64) Joint {
	return Joint{kind: jointKindFixed, xyz: xyz, wxyz: wxyz}
}

// Chain is a Go-owned handle to a serial kinematic chain held on the
// C side of the trajex C ABI. Trajex uses a chain to limit how the end of an
// arm moves through space, for example to cap the speed of its tool center
// point. Construct one with NewChain and release it with Close.
//
// The chain has no units of its own. Whatever length unit the joint
// translations use, a TCP speed cap is in that unit per second. The point
// being capped is the origin of the chain's last frame, so a caller who wants
// the cap applied at a tool rather than at the flange should end the chain at
// the tool, adding a FixedJoint if necessary.
//
// Nothing in a chain changes once it is built, so any number of goroutines
// may use one at the same time without coordinating. Close is the exception,
// and must not race with any other use. The functions that take a chain do
// not hold on to it after they return, so a chain may be released as soon as
// the last call using it has returned.
type Chain struct {
	handle *C.viam_trajex_kinematics_chain_t
}

// NewChain builds a chain from joints, which are listed in order from
// the base to the end effector. It returns an error if there are no joints,
// or if any joint was not made by a Joint constructor, has a translation that
// is not finite, or has a quaternion or axis that is zero or not finite. When
// a particular joint is at fault, the error names its index. The returned
// chain must be released by calling Close.
func NewChain(joints []Joint) (*Chain, error) {
	builder := C.viam_trajex_kinematics_chain_builder_create()
	if builder == nil {
		return nil, errors.New("trajex: kinematic chain builder allocation failed")
	}
	// Building consumes the builder whether or not it succeeds, but the
	// handle itself still has to be destroyed, so we do that on every path
	// out of this function.
	defer C.viam_trajex_kinematics_chain_builder_destroy(builder)

	// The C builder only refuses to append a joint when given a null
	// argument or when an allocation fails. Problems with the values
	// themselves, such as a zero axis, are not reported until we build,
	// since that is where the C side knows the index of the offending joint
	// and can name it in the message.
	for i := range joints {
		j := &joints[i]
		xyz := (*C.double)(unsafe.Pointer(&j.xyz[0]))
		wxyz := (*C.double)(unsafe.Pointer(&j.wxyz[0]))
		var rc C.int
		switch j.kind {
		case jointKindFixed:
			rc = C.viam_trajex_kinematics_chain_builder_extend_fixed(builder, xyz, wxyz)
		case jointKindRotational:
			rc = C.viam_trajex_kinematics_chain_builder_extend_rotational(
				builder, xyz, wxyz, (*C.double)(unsafe.Pointer(&j.axis[0])))
		case jointKindTranslational:
			rc = C.viam_trajex_kinematics_chain_builder_extend_translational(
				builder, xyz, wxyz, (*C.double)(unsafe.Pointer(&j.axis[0])))
		default:
			return nil, errors.Errorf("trajex: joint %d was not made by a Joint constructor", i)
		}
		if rc != 0 {
			return nil, errors.Errorf("trajex: kinematic chain builder failed to append joint %d", i)
		}
	}

	var errOut *C.char
	h := C.viam_trajex_kinematics_chain_builder_build(builder, &errOut)
	if h == nil {
		msg := C.GoString(errOut)
		C.viam_trajex_string_destroy(errOut)
		return nil, errors.Errorf("trajex: building kinematic chain failed: %s", msg)
	}
	return &Chain{handle: h}, nil
}

// Close releases the underlying C chain. It may be called on a nil chain, and
// calls after the first do nothing.
func (c *Chain) Close() {
	if c == nil || c.handle == nil {
		return
	}
	C.viam_trajex_kinematics_chain_destroy(c.handle)
	c.handle = nil
}

// UnsafeHandle returns the underlying C chain handle as an unsafe.Pointer, or
// nil if the chain is nil or has been released with Close. It exists for the sibling
// packages in this module that call the C ABI themselves, such as
// totg.GenerateWithKinematics, and is not meant for other callers.
func (c *Chain) UnsafeHandle() unsafe.Pointer {
	if c == nil {
		return nil
	}
	return unsafe.Pointer(c.handle)
}

// EndPosition returns the position of the origin of the chain's last frame
// when the joints are at q, in the length unit of the chain's translations.
// q must hold one position for each rotational or translational joint, in the
// order those joints appear in the chain. Fixed joints take no position.
//
// Trajex has no use for this itself. It is here so that a caller can compare
// the chain it built against its own forward kinematics at a few joint
// positions, which is the most direct way to find out whether the origins,
// axes, rotations, and units went in the way the caller intended.
func (c *Chain) EndPosition(q []float64) ([3]float64, error) {
	var position [3]float64
	if c == nil || c.handle == nil {
		return position, errors.New("trajex: EndPosition on a nil or released kinematic chain")
	}
	var qPtr *C.double
	if len(q) > 0 {
		qPtr = (*C.double)(unsafe.Pointer(&q[0]))
	}
	var errOut *C.char
	rc := C.viam_trajex_kinematics_chain_end_position(
		c.handle, qPtr, C.size_t(len(q)), (*C.double)(unsafe.Pointer(&position[0])), &errOut)
	if rc != 0 {
		msg := C.GoString(errOut)
		C.viam_trajex_string_destroy(errOut)
		return position, errors.Errorf("trajex: EndPosition failed: %s", msg)
	}
	return position, nil
}
