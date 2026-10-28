# SPDX-License-Identifier: Apache-2.0
"""A scripted solution: pick each gear, drop it on its peg, press the button.

**Four things this cost measurements to get right**, none of them visible from
the code that appears to control them.

1. **The jaws must close ACROSS the chain, not along it.** A gear is gripped on
   its rim, so the fingers sit on the diameter -- and the diameter along the
   chain is exactly where the neighbouring gear is. Closing that way puts a
   finger through the gear already seated next door. The wrist is therefore
   yawed to put the hand's own ``close_axis`` on world x, and the yaw is
   *computed* from the solved orientation rather than written down, because
   which way "zero yaw" faces is a fact about the hand's mount.

2. **A gear on a flat mat can be gripped from the side, and only just.** The
   fingertips have to straddle a 10 mm rim, so they descend to about 2.5 mm
   above whatever the gear is lying on -- and that clearance is the same number
   at the pick and at the seat, because the grasp height relative to the gear is
   fixed the moment the jaws close. At the seat it is what keeps the fingertips
   off the plate.

3. **The descent onto a peg is a drop, not a placement.** The fit is 0.3 mm of
   radial clearance against an arm that lands within about 1.5 mm, so the gear
   is not lowered all the way down and released -- it is brought to just above
   the peg's coned lead-in, released there, and allowed to fall the last few
   millimetres. The cone turns a miss into a slide; a still-gripped gear cannot
   slide, because the jaws hold it exactly where the arm's error put it.

4. **The button is pressed through its stop.** Its plunger is a position servo
   held out by its own return spring, so a command that stops at the latch
   depth stops *at* it and the latch reads a boundary. The command goes to the
   bottom of the well and the joint limit takes the difference, which is what a
   finger does to a button.
"""

from __future__ import annotations

import math

import numpy as np

from robosimstudio.arms import arm_slot, arm_slots
from robosimstudio.core.math import quat_multiply, quat_to_mat

__all__ = ["RoutineGearMesh"]

OPEN, SHUT = 0.0, 1.0
"""The Franka hand closes at the HIGH end of a normalised gripper command
(``GripperSpec.action_closed_at_high``)."""

MAX_POS, MAX_ROT = 0.006, math.radians(8.0)
"""What a leg's worst waypoint may miss by before it is refused rather than
driven. An unreachable waypoint is a lurch, and a lurch with a gear in the hand
throws it across the bench."""

JAW_MARGIN = 0.012
"""How much wider than the gear the jaws wait before closing [m].

**A pre-close, and the smallest gear is why.** Driven straight from fully open,
the pads on a 45 mm gear travel 17 mm each before they touch and arrive with
speed -- measured, the 16-tooth gear was knocked out from between them every
time while the 55 and 65 mm ones, whose pads start 7 mm out, were gripped
cleanly. Waiting 6 mm a side makes the final close a squeeze rather than a
swipe, at every size."""

RIM_CLEAR = 0.0025
"""How far the fingertips stop above whatever the gear is lying on [m].

The one clearance the whole grasp is built on. Below it the pads catch the mat;
above it they close on the top of the rim and the gear tips out. It is also what
keeps them off the plate at the seat, because the grasp height relative to the
gear stops changing the moment the jaws shut."""

ATTEMPTS = 4
"""How many times a gear is worked down onto its peg before the hand gives up.

**Closed loop, and the only closed loop in this routine.** Everything else here
is derived geometry driven open -- the pegs are where the board says they are --
but whether a gear went home is not a fact about geometry, it is a fact about
whether its teeth happened to meet a space or a tip. The board publishes
``gear<i>_seated`` and this reads it between attempts, lifting back to the
staging height and searching again. Measured on the three-gear board, one pass
seats two of three; the third goes in on a retry."""

SEARCH_TURNS = 5.0
"""Rocking cycles the wrist makes over the final descent.

**The insertion search, and it is the whole of what makes the last gear go in.**
A gear's last job on its way home is to turn until its teeth find its
neighbour's spaces, and it cannot do that for itself while the jaws are on it --
but the jaws are also the only thing keeping it level, and a gear let go to find
its own way tips and wedges across the shaft every time. So the ARM turns it:
the wrist rocks half a tooth pitch either way while it descends, which sweeps a
whole tooth space, and the gear stays flat throughout because it is still held.

Five cycles of a WHOLE tooth pitch either way. Half a pitch sweeps exactly one
space and is enough for most insertions -- two of three gears go in on it -- but
the one that does not is left sitting level and high, resting tip on tip, which
is a search that never reached a valid phase rather than a gear that jammed."""

RELEASE_GAP = 0.002
"""How far above its seated height the gear is let go [m].

**The release height is set by the teeth, not by the bore.** The last thing a
gear does on its way home is turn until its teeth find its neighbour's spaces,
and it can only do that once the jaws are off it -- so it has to be released
above the plane its neighbour's teeth are in, with the bore already captured by
parallel shaft so it cannot tip while it falls. That is a face width plus a
couple of millimetres, and it is why the peg carries a face width of staging
shaft above a seated gear (``SpecGearBoard.peg_lead``).

Measured the two ways this can be got wrong: carried all the way down while
gripped, the FIRST gear seats perfectly -- 0.00 degrees of tilt, dead on the
seat -- and every gear after it jams on the neighbour it cannot turn to clear.
Released above the peg's tip instead, it lands on the cone off-centre with
nothing holding it level and wedges diagonally at 11 to 21 degrees.

**The gear is carried down the cone, level, and released at the bottom.** Both
alternatives were built and measured, and the reason this one wins is the shape
of the lead-in rather than anything about the arm:

* *Pushed onto a blunt peg* the tip catches the inside of the bore, the arm keeps
  descending, and the gear pivots about that contact in jaws that cannot stop it
  turning -- 27 degrees of tilt, jammed five millimetres up.
* *Dropped from above the tip* it lands on the cone off-centre and tips for the
  same reason nothing is holding it level -- 11 to 21 degrees, wedged diagonally
  across the shaft, and no better with a longer peg.

What fixes it is a cone sharp enough to enter the bore at the arm's own error
(1.2 mm tip inside a 5.9 mm bore) and long enough to centre the gear over the
whole descent -- and then the jaws are an *advantage*, because a gripped gear
cannot tip while the cone does its work. So the arm keeps hold all the way down
and lets go two millimetres above the seat, with the bore already on parallel
shaft."""

CARRY_Z = 0.83
"""World height a carried gear's TCP travels at [m].

Above every peg by a clear margin -- the tallest stands to 0.774 and a carried
gear's underside hangs about 12 mm below the TCP -- and inside the band the
reach probe measured solvable over every station (``gear_bench.py reach``
reports 0.79-1.04 mm at 0.834 across the four-gear board)."""

SETTLE = 20
"""Control steps held after a leg, for the arm to converge before the next
solve is seeded from it."""


class RoutineGearMesh:
    """Assemble the train, then start it."""

    MODE = "joint_position"

    def __init__(self, env, **kwargs) -> None:
        del kwargs
        self.env = env
        self.work = env.handles.work_arm or arm_slots(env)[0]
        self.dof = len(env.idle_action(self.MODE)) // len(arm_slots(env))
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        env = self.env
        self.held = {n: self._q(n) for n in arm_slots(env)}
        self.grip = dict.fromkeys(arm_slots(env), OPEN)
        self.seed = {n: np.asarray(env.state.joint_q.numpy(), dtype=np.float64) for n in arm_slots(env)}
        self.refused: list[str] = []
        self.placed = 0
        self.retried = 0
        self.pressed = 0.0
        self._steps = self._choreography()

    def _q(self, name):
        ik = self.env.arm(name)
        return np.asarray(self.env.state.joint_q.numpy(), dtype=np.float64)[ik.arm_coords]

    def row_extras(self) -> dict[str, float]:
        return {
            "gears_placed": float(self.placed),
            "insert_retries": float(self.retried),
            "button_pressed": float(self.pressed),
            "legs_refused": float(len(self.refused)),
        }

    # ------------------------------------------------------------------ drive
    def act(self) -> np.ndarray:
        try:
            next(self._steps)
        except StopIteration:
            pass
        return self._action()

    def _action(self) -> np.ndarray:
        a = np.asarray(self.env.idle_action(self.MODE), dtype=np.float64)
        for name in arm_slots(self.env):
            s = arm_slot(self.env, name) * self.dof
            a[s : s + self.dof - 1] = self.held[name][: self.dof - 1]
            a[s + self.dof - 1] = self.grip[name]
        return a

    # ------------------------------------------------------------------- legs
    def _hold(self, n: int):
        for _ in range(n):
            yield

    def _leg(self, target, quat, *, speed=0.10, grip=None, label="", gate=True):
        """Solve the whole leg, gate it, then drive it one waypoint per step.

        box_decant's leg planner, and for its reasons: one IK solve held for N
        steps is a single target the servo slews to at full rate, which throws a
        carried part, and an unreachable waypoint comes back tens of millimetres
        off with nothing raised. ``gate`` is off for a free approach, whose first
        waypoints legitimately miss while the arm converges out of its home pose.
        """
        name = self.work
        ik = self.env.arm(name)
        start, _ = ik.ee_pose(self.env.state)
        target = np.asarray(target, dtype=np.float64)
        n = max(2, int(np.linalg.norm(target - start) / speed * 60.0))
        sols, worst_p, worst_r, seed = [], 0.0, 0.0, self.seed[name]
        for k in range(n):
            # **Eased, not linear.** A linear interpolation is constant velocity
            # with an instantaneous start and stop, and the stop is what throws a
            # carried gear: measured, every gear was lifted cleanly to its carry
            # height and then dropped part-way along the traverse, landing on the
            # plate beside its peg. A gear is held on its rim by two pads touching
            # tooth tips, which is a strong grip against gravity and a poor one
            # against a jerk. Smoothstep starts and ends at zero speed.
            t = (k + 1) / n
            step = start + (target - start) * (t * t * (3.0 - 2.0 * t))
            sol = ik.solve_q_full(seed, step, np.asarray(quat))
            p, r = ik.pose_errors(sol, step, np.asarray(quat))
            worst_p, worst_r = max(worst_p, p), max(worst_r, r)
            sols.append(sol)
            seed = sol
        if gate and (worst_p > MAX_POS or worst_r > MAX_ROT):
            self.refused.append(f"{label or 'leg'}: {worst_p * 1e3:.1f} mm / {math.degrees(worst_r):.1f} deg")
            return
        if grip is not None:
            self.grip[name] = grip
        for sol in sols:
            self.seed[name] = sol
            self.held[name] = np.asarray(sol)[ik.arm_coords]
            yield

    def _seat(self, at, quat, body, *, sweep, speed=0.015, budget=5.0):
        """Work the gear down onto its peg, rocking, and only while it is moving.

        **The descent is advanced by the gear, not by the clock**, and that is
        what makes this an insertion rather than a press. A gear whose teeth have
        landed on its neighbour's tips does not go down however hard it is
        pushed; what it needs is to be turned until it finds the space. So the
        schedule only steps forward while the gear is actually falling, and when
        it stops the hand holds that height and keeps searching.

        Driven open-loop instead, the arm simply forces a jammed gear: measured,
        the wedge then holds the wrist so firmly that the *retry* could not lift
        it either -- the target went up 11 mm and the tool came up 3.

        ``sweep`` is a whole tooth pitch, so a cycle sweeps well past any single
        space; see :data:`SEARCH_TURNS`.
        """
        name = self.work
        ik = self.env.arm(name)
        start, _ = ik.ee_pose(self.env.state)
        target = np.asarray(at, dtype=np.float64)
        n = max(2, int(np.linalg.norm(target - start) / speed * 60.0))
        lowest = float(self.env.state.body_q.numpy()[body][2])
        progress, stalled = 0.0, 0
        for _ in range(int(n * budget)):
            height = float(self.env.state.body_q.numpy()[body][2])
            if height < lowest - 2.0e-5:
                stalled = 0
                lowest = height
            else:
                stalled += 1
            if stalled < 8 and progress < 1.0:
                progress = min(1.0, progress + 1.0 / n)
            step = start + (target - start) * (progress * progress * (3.0 - 2.0 * progress))
            turn = sweep * math.sin(2.0 * math.pi * SEARCH_TURNS * progress + 0.35 * stalled)
            rocked = quat_multiply([0.0, 0.0, math.sin(0.5 * turn), math.cos(0.5 * turn)], np.asarray(quat))
            sol = ik.solve_q_full(self.seed[name], step, np.asarray(rocked))
            self.seed[name] = sol
            self.held[name] = np.asarray(sol)[ik.arm_coords]
            yield
            if progress >= 1.0 and stalled > 30:
                return

    def _down_quat(self, at, yaw_deg=0.0):
        """A top-down wrist orientation over ``at``, turned ``yaw_deg`` about z."""
        ik = self.env.arm(self.work)
        q0, _ = ik.solve_best([self.seed[self.work]], np.asarray(at), (0.0, 0.0, -1.0))
        _, home = ik.fk_ee(np.asarray(q0, dtype=np.float32))
        y = math.radians(yaw_deg)
        return quat_multiply([0.0, 0.0, math.sin(y / 2), math.cos(y / 2)], home)

    def _hand(self) -> tuple[np.ndarray, float]:
        """``(jaw axis in the tool's own frame, fingertip drop below the TCP)``.

        **Measured off the built model, not read off a spec.** Both numbers are
        facts about the hand as it actually stands in this scene -- the runtime
        ``Gripper`` does not carry either, and the registered ``GripperSpec``
        describes the hand rather than the arm it ended up on. Taken once, in the
        tool's own frame, so they stay true as the wrist turns.
        """
        model = self.env.scene.pipeline.model
        labels = list(model.body_label)
        tips = [i for i, lab in enumerate(labels) if "finger" in lab and "tip" in lab]
        if len(tips) < 2:
            tips = [i for i, lab in enumerate(labels) if "finger" in lab]
        if len(tips) < 2:
            raise LookupError("gear_mesh routine: cannot find two finger bodies to measure the jaw axis")
        body_q = np.asarray(self.env.state.body_q.numpy(), dtype=np.float64)
        a, b = body_q[tips[0]][:3], body_q[tips[1]][:3]
        at, quat = self.env.arm(self.work).ee_pose(self.env.state)
        rot = quat_to_mat(np.asarray(quat, dtype=np.float64))
        jaw = rot.T @ (a - b)
        jaw = jaw / float(np.linalg.norm(jaw))
        # How far the TCP sits BACK from the fingertips, as a positive distance
        # along the tool's own approach axis. The magnitude, deliberately: the
        # tool approaches along its local +z, which for a top-down wrist points
        # DOWN, so the signed component comes out negative and the first build
        # subtracted it -- putting the fingers a full tip-reach BELOW the gear,
        # where they foul the peg and shove the gear off its station. Measured
        # 11.4 mm on this hand.
        drop = abs(float((rot.T @ (np.asarray(at, dtype=np.float64) - 0.5 * (a + b)))[2]))
        return jaw, drop

    def _across_quat(self, at, jaw_local):
        """Top-down, with the jaws closing along world x.

        **Computed, not written down.** Which way the hand's jaws face at zero
        yaw is a fact about its mount, and a task that guessed would put a finger
        through the gear seated next door -- silently, because the finger simply
        pushes it off and the trace shows a gear that "fell".
        """
        home = self._down_quat(at, 0.0)
        jaw = quat_to_mat(np.asarray(home, dtype=np.float64)) @ jaw_local
        return self._down_quat(at, -math.degrees(math.atan2(float(jaw[1]), float(jaw[0]))))

    # -------------------------------------------------------------- resolving
    def _board(self):
        """The board entity, out of the same bin a goal reads a state from.

        Through ``make_context().extras`` rather than off the compiled scene: a
        ``CompiledScene`` has no ``extras``, and this bin is the one the
        evaluator itself resolves published states from, so the routine and the
        verdict are looking at the same object.
        """
        for entity in self.env.make_context().extras.values():
            if type(entity).__name__ == "EntityGearBoard":
                return entity
        raise LookupError("gear_mesh routine: the scene has no gear board")

    def _state(self) -> dict[str, float]:
        return dict(self._board().published_states(self.env.make_context()))

    # ---------------------------------------------------------- choreography
    def _choreography(self):
        env = self.env
        board = self._board()
        seats = np.asarray(board._seat_world, dtype=np.float64)
        up = np.asarray(board._up_world, dtype=np.float64)
        at_start = self._state()

        # Grip height above whatever the gear rests on, fixed once and for all:
        # the fingertips clear that surface by RIM_CLEAR and the TCP sits a
        # tip_reach above them. It is the same offset at the pick and the seat.
        hand = self.env.gripper(self.work)
        jaw_local, tip = self._hand()
        lift = tip + RIM_CLEAR - 0.5 * board.spec.face

        # **Get to carry height before going anywhere.** The arm starts high and
        # over the mat, and a free approach straight to a gear crosses the ones
        # beside it on the way down -- measured, the first gear was swept 30 mm
        # along the mat and the jaws closed on air. Every horizontal move in this
        # routine happens at CARRY_Z, which is clear of everything on the board.
        here, _ = self.env.arm(self.work).ee_pose(self.env.state)
        yield from self._leg(
            [here[0], here[1], CARRY_Z],
            self._across_quat(here, jaw_local),
            grip=OPEN,
            gate=False,
            speed=0.30,
            label="up to carry height",
        )
        yield from self._hold(SETTLE)

        for index, gear_body in enumerate(board.gear_bodies):
            if at_start.get(f"gear{index}_seated", 0.0) > 0.5:
                self.placed += 1
                continue
            gear = np.asarray(env.state.body_q.numpy()[gear_body], dtype=np.float64)[:3]
            pick = np.array([gear[0], gear[1], gear[2] + lift])
            quat = self._across_quat(pick, jaw_local)
            span = 2.0 * board.geometry["frames"][index + 1]["r_tip"]
            ready = float(hand.command_for_width(span + JAW_MARGIN))
            # Where the bore is captured by parallel shaft and the teeth are
            # still above the neighbour's: the top of the staging zone.
            staged = seats[index] + up * (board.spec.face + RELEASE_GAP + lift)
            drop = seats[index] + up * (RELEASE_GAP + lift)

            yield from self._leg(
                [pick[0], pick[1], CARRY_Z], quat, grip=ready, gate=False, speed=0.22, label=f"over {index}"
            )
            yield from self._hold(SETTLE)
            yield from self._leg(pick, quat, speed=0.06, gate=False, label=f"descend to {index}")
            yield from self._hold(SETTLE)
            self.grip[self.work] = SHUT
            yield from self._hold(70)
            yield from self._leg([pick[0], pick[1], CARRY_Z], quat, speed=0.05, label=f"lift {index}")
            yield from self._hold(SETTLE)
            yield from self._leg([drop[0], drop[1], CARRY_Z], quat, speed=0.05, label=f"carry {index}")
            yield from self._hold(SETTLE)
            yield from self._leg(staged, quat, speed=0.03, label=f"onto peg {index}")
            yield from self._hold(20)
            sweep = 2.0 * math.pi / float(board.geometry["counts"][index + 1])
            for attempt in range(ATTEMPTS):
                yield from self._seat(drop, quat, gear_body, sweep=sweep)
                yield from self._hold(20)
                # **Let go before judging.** A gear resting tooth on tooth is
                # held there by the jaws as much as by the teeth: opening them
                # frees it to turn on its own peg, which is the one thing it
                # needs and the one thing a grip cannot allow. Half the
                # insertions that fail while held go in the moment they are
                # released.
                self.grip[self.work] = OPEN
                yield from self._hold(70)
                if self._state().get(f"gear{index}_seated", 0.0) > 0.5:
                    break
                if attempt < ATTEMPTS - 1:
                    self.retried += 1
                    self.grip[self.work] = SHUT
                    yield from self._hold(50)
                    yield from self._leg(staged, quat, speed=0.04, label=f"retry {index}")
                    yield from self._hold(15)
            yield from self._leg([drop[0], drop[1], CARRY_Z], quat, grip=OPEN, speed=0.12, label=f"clear {index}")
            yield from self._hold(SETTLE)
            self.placed += 1

        # --- press ---------------------------------------------------------
        button = np.asarray(board._button_rest, dtype=np.float64)
        over = button + up * 0.06
        bq = self._across_quat(over, jaw_local)
        yield from self._leg([over[0], over[1], CARRY_Z], bq, grip=SHUT, speed=0.25, label="over the button")
        yield from self._hold(SETTLE)
        # Through the stop: the plunger's own joint limit takes the difference,
        # which is what a finger does to a button. Stopping at the latch depth
        # asks the latch to fire on a boundary.
        press = button + up * (tip - float(board.spec.button_travel) - 0.001)
        yield from self._leg(press, bq, speed=0.05, label="press")
        yield from self._hold(45)
        self.pressed = float(self._state().get("started", 0.0))
        yield from self._leg([press[0], press[1], CARRY_Z], bq, speed=0.12, label="off the button")

        # --- and watch it run ----------------------------------------------
        while True:
            yield
