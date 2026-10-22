//go:build !windows && cgo

// Package rdk turns RDK's kinematic models into trajex joints. It is the one
// place in this module that knows what an RDK model is; everything else
// describes kinematic chains.
package rdk

import (
	"slices"
	"sort"

	"github.com/pkg/errors"
	"go.viam.com/rdk/referenceframe"
	"go.viam.com/rdk/spatialmath"

	"github.com/viam-modules/trajex/go/kinematics"
)

// JointsFromModel describes the serial chain from the base of m out to a tool
// center point, which is where a TCP speed cap applies. The result is meant
// to be passed to kinematics.NewChain.
//
// tcp is the pose of the tool center point relative to the model's own output
// frame, the frame whose pose m.Transform reports. Where that point lies
// depends on what is mounted on the arm, which only the caller knows, so it
// must always be given. A caller capping the end of the arm itself passes
// spatialmath.NewZeroPose().
//
// Lengths are in millimetres, which is what RDK's models use, and prismatic
// joint positions are in millimetres for the same reason. The translation in
// tcp must be in millimetres as well, and a TCP speed cap for the resulting
// chain is in millimetres per second, so a cap held in metres per second,
// such as MaxTCPSpeedMPerSec, must be multiplied by 1000.
//
// JointsFromModel only maps geometry. The model's own joint limits and TCP
// cap are not carried over, since how they combine with what a particular
// call asks for is the caller's policy.
//
// It returns an error for a model that was built in code rather than loaded
// from a file, since such a model carries no description of its geometry, and
// for any model that trajex cannot drive with the same joint positions RDK
// would use: one with a mimic joint on the path to its output frame, or with
// moving joints anywhere else in the model, such as a gripper's fingers.
func JointsFromModel(m referenceframe.Model, tcp spatialmath.Pose) ([]kinematics.Joint, error) {
	if m == nil {
		return nil, errors.New("trajex/rdk: model is nil")
	}
	if tcp == nil {
		return nil, errors.New("trajex/rdk: tcp pose is nil; pass spatialmath.NewZeroPose() to cap the end of the arm itself")
	}
	cfg := m.ModelConfig()
	if cfg == nil {
		return nil, errors.Errorf("trajex/rdk: model %q has no configuration, which is the case for models built in code", m.Name())
	}

	elements, err := elementsFromConfig(cfg)
	if err != nil {
		return nil, err
	}
	output, err := outputFrameOf(cfg, elements)
	if err != nil {
		return nil, err
	}

	// Walk from the output frame up to the root, which is whichever element
	// names a parent that is not itself an element of the model, usually
	// "world". RDK has already refused models with cycles, but a model can be
	// assembled by hand, so we still guard against walking one forever.
	var path []string
	seen := map[string]bool{}
	for id := output; ; {
		e, ok := elements[id]
		if !ok {
			break
		}
		if seen[id] {
			return nil, errors.Errorf("trajex/rdk: the path from %q to the root of the model is a cycle", output)
		}
		seen[id] = true
		path = append(path, id)
		id = e.parent
	}
	slices.Reverse(path)

	// Now walk back down from the root. In RDK's models a link is a static
	// pose relative to its parent and a joint is pure motion about an axis in
	// its parent's frame, while a trajex joint carries the static transform
	// that precedes its motion as its origin. So we compose the poses of the
	// links between one joint and the next, and hand the result to the next
	// joint as its origin.
	var joints []kinematics.Joint
	var moving []string
	origin := spatialmath.NewZeroPose()
	for _, id := range path {
		e := elements[id]
		if !e.isJoint {
			origin = spatialmath.Compose(origin, e.pose)
			continue
		}
		if e.mimic {
			return nil, errors.Errorf("trajex/rdk: joint %q mimics another joint, which trajex cannot express", id)
		}
		xyz, wxyz := originComponents(origin)
		switch e.jointType {
		case referenceframe.RevoluteJoint:
			joints = append(joints, kinematics.RotationalJoint(xyz, wxyz, e.axis))
		case referenceframe.PrismaticJoint:
			joints = append(joints, kinematics.TranslationalJoint(xyz, wxyz, e.axis))
		default:
			return nil, errors.Errorf("trajex/rdk: joint %q has type %q, which trajex does not support", id, e.jointType)
		}
		moving = append(moving, id)
		origin = spatialmath.NewZeroPose()
	}

	// The tool center point is given relative to the output frame, so it
	// extends whatever links follow the last joint, and the two together
	// become the chain's final fixed joint. We add that joint even when both
	// are trivial, since it costs nothing and gives every chain the same
	// shape whatever the caller passed.
	xyz, wxyz := originComponents(spatialmath.Compose(origin, tcp))
	joints = append(joints, kinematics.FixedJoint(xyz, wxyz))

	// RDK takes one input for every moving joint in the whole model, in its
	// own order, and a caller will hand trajex those same inputs. Unless the
	// moving joints on our path are exactly the model's moving joints, in the
	// same order, those inputs would drive the wrong joints, so we refuse the
	// model rather than produce a chain that silently disagrees with it.
	// MoveableFrameNames is not part of the Model interface, but every model
	// loaded from a file provides it.
	named, ok := m.(interface{ MoveableFrameNames() []string })
	if !ok {
		return nil, errors.Errorf("trajex/rdk: model %q does not report its moving joints", m.Name())
	}
	if want := named.MoveableFrameNames(); !slices.Equal(moving, want) {
		return nil, errors.Errorf(
			"trajex/rdk: model %q takes inputs for joints %v, but the path to its output frame %q moves joints %v",
			m.Name(), want, output, moving)
	}

	return joints, nil
}

// element is one link or joint of a model's configuration. A link has a pose
// and no axis, and a joint has an axis and no pose.
type element struct {
	parent    string
	isJoint   bool
	pose      spatialmath.Pose
	jointType string
	axis      [3]float64
	mimic     bool
}

// elementsFromConfig indexes a model's links and joints by name. JSON and URDF
// models both arrive as SVA links and joints, since RDK converts URDF on
// load. DH parameters are expanded the way RDK expands them, into a joint
// rotating about z followed by a link whose pose RDK computes for us.
func elementsFromConfig(cfg *referenceframe.ModelConfigJSON) (map[string]*element, error) {
	elements := map[string]*element{}
	add := func(id string, e *element) error {
		if _, ok := elements[id]; ok {
			return errors.Errorf("trajex/rdk: model names %q more than once", id)
		}
		elements[id] = e
		return nil
	}

	switch cfg.KinParamType {
	case "SVA", "":
		for i := range cfg.Links {
			link := &cfg.Links[i]
			pose, err := link.Pose()
			if err != nil {
				return nil, errors.Wrapf(err, "trajex/rdk: link %q", link.ID)
			}
			if err := add(link.ID, &element{parent: link.Parent, pose: pose}); err != nil {
				return nil, err
			}
		}
		for i := range cfg.Joints {
			joint := &cfg.Joints[i]
			e := &element{
				parent:    joint.Parent,
				isJoint:   true,
				jointType: joint.Type,
				axis:      [3]float64{joint.Axis.X, joint.Axis.Y, joint.Axis.Z},
				mimic:     joint.Mimic != nil,
			}
			if err := add(joint.ID, e); err != nil {
				return nil, err
			}
		}
	case "DH":
		for i := range cfg.DHParams {
			dh := &cfg.DHParams[i]
			_, link, err := dh.ToDHFrames()
			if err != nil {
				return nil, errors.Wrapf(err, "trajex/rdk: DH parameters %q", dh.ID)
			}
			pose, err := link.Transform([]referenceframe.Input{})
			if err != nil {
				return nil, errors.Wrapf(err, "trajex/rdk: DH parameters %q", dh.ID)
			}
			// ToDHFrames names the joint this way and always turns it about z.
			jointID := dh.ID + "_j"
			joint := &element{parent: dh.Parent, isJoint: true, jointType: referenceframe.RevoluteJoint, axis: [3]float64{0, 0, 1}}
			if err := add(jointID, joint); err != nil {
				return nil, err
			}
			if err := add(dh.ID, &element{parent: jointID, pose: pose}); err != nil {
				return nil, err
			}
		}
	default:
		return nil, errors.Errorf("trajex/rdk: unsupported kinematic parameter type %q", cfg.KinParamType)
	}
	return elements, nil
}

// outputFrameOf finds the frame a model reports its pose for. RDK keeps that
// to itself, so we apply the rule RDK applies when it builds a model: the
// configured output frame if there is one, and otherwise the model's single
// leaf. RDK refuses to build a model that satisfies neither, and so do we.
func outputFrameOf(cfg *referenceframe.ModelConfigJSON, elements map[string]*element) (string, error) {
	if len(cfg.OutputFrames) > 0 {
		output := cfg.OutputFrames[0]
		if _, ok := elements[output]; !ok {
			return "", errors.Errorf("trajex/rdk: output frame %q is not part of the model", output)
		}
		return output, nil
	}
	hasChild := map[string]bool{}
	for _, e := range elements {
		hasChild[e.parent] = true
	}
	var leaves []string
	for id := range elements {
		if !hasChild[id] {
			leaves = append(leaves, id)
		}
	}
	if len(leaves) != 1 {
		sort.Strings(leaves)
		return "", errors.Errorf("trajex/rdk: model has no output frame and %d leaves %v rather than one", len(leaves), leaves)
	}
	return leaves[0], nil
}

// originComponents splits a pose into the translation and w, x, y, z
// quaternion that trajex takes as a joint's origin.
func originComponents(pose spatialmath.Pose) ([3]float64, [4]float64) {
	p := pose.Point()
	q := pose.Orientation().Quaternion()
	return [3]float64{p.X, p.Y, p.Z}, [4]float64{q.Real, q.Imag, q.Jmag, q.Kmag}
}
