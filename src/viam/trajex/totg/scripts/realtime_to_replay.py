#!/usr/bin/env python3
"""
Normalize a TOTG record into the canonical replay-record schema that
viam-trajex-totg-replay consumes, then write it to stdout.

Two inputs are accepted and auto-detected:

  * A replay record (already has "waypoints_rads"): emitted unchanged, so this
    script is a safe pass-through in front of the replay tool.
  * A realtime trajectory log (has "waypoints_rad", written by the robot on a
    successful move): its planning inputs are remapped to the replay schema so
    the move can be re-planned and rendered.

A realtime log records the executed trajectory, not the planner's phase plane,
so re-planning regenerates the limit curves. The path tolerance is reproduced
only if the log recorded it; older logs fall back to the driver default. No arm
driver records its kinematics, so the plan is joint-only.
"""

import json
import sys

# Driver default for path_tolerance_rad when a realtime log does not record it.
DEFAULT_PATH_TOLERANCE_RADS = 0.1


def realtime_to_replay(d):
    """Map a realtime trajectory log to a replay record."""
    cfg = d.get('configuration', {})
    max_vel = cfg.get('max_velocity_rad_per_sec')
    max_acc = cfg.get('max_acceleration_rad_per_sec2')
    if max_vel is None or max_acc is None:
        sys.exit('error: realtime log missing configuration velocity/acceleration limits')

    out = {
        'schema_version': 3,
        'waypoints_rads': d['waypoints_rad'],
        'max_velocity_vec_rads_per_sec': max_vel,
        'max_acceleration_vec_rads_per_sec2': max_acc,
    }

    if 'path_tolerance_delta_rads' in d:
        out['path_tolerance_delta_rads'] = d['path_tolerance_delta_rads']
    else:
        out['path_tolerance_delta_rads'] = DEFAULT_PATH_TOLERANCE_RADS
        print(f'note: realtime log records no path tolerance; assuming driver default '
              f'{DEFAULT_PATH_TOLERANCE_RADS}', file=sys.stderr)

    return out


def normalize(d):
    if 'waypoints_rads' in d:
        return d  # already a replay record
    if 'waypoints_rad' in d:
        return realtime_to_replay(d)
    sys.exit('error: unrecognized record (no "waypoints_rads" or "waypoints_rad"); '
             'this is neither a replay record nor a realtime trajectory log')


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        data = json.load(open(src)) if src else json.load(sys.stdin)
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f'error: cannot read record: {e}')
    json.dump(normalize(data), sys.stdout)


if __name__ == '__main__':
    main()
