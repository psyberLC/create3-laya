# Contributors

| Who | Role |
|---|---|
| [psyberLC](https://github.com/psyberLC) | Owner. Set the goals and made the design decisions (local models only, which simulators, how the robot should behave), tested the robot in the simulators and directed the fixes. |
| Claude (Anthropic): Claude Opus 5.5 in Claude Code | AI coding assistant. Wrote most of the code, scripts, tests and documentation at the owner's direction, and ran the simulations and measurements reported in the README. |

## AI use

Most of this project was written by an AI coding assistant, Claude, working in Claude Code
at the owner's direction. Commits Claude contributed to carry a
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` trailer.

The code was checked with the project's test suites and the simulation runs described in the
README. As with any code, review it before relying on it, especially before using it on a real
robot.

## Third-party parts

- `sim2d/worlds/maze.sdf` is iRobot's, from create3_sim (BSD-3-Clause; see
  `sim2d/worlds/NOTICE`).
- The Laya and OWLv2 models are downloaded during setup, not included here.
