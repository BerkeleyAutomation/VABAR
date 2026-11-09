# Industrial Benchmark Context

This context describes the physical workcell language shared by the industrial FR3 and YAM manipulation tasks.

## Language

**Industrial Workcell Task**:
An FR3 or YAM manipulation task built around the Canonical Table and an explicit Task Layout.
_Avoid_: All tasks, benchmark task

**Canonical Table**:
The Stainless Metal Table A03 asset used as the physical support surface for an industrial workcell. Its pose is defined relative to the workcell robot rather than by one universal world transform.
_Avoid_: Gear table, syringe table, replacement table

**Embodiment Table Frame**:
The canonical pose of the Canonical Table relative to one robot embodiment. FR3 and YAM each have exactly one such frame shared by their Industrial Workcell Tasks.
_Avoid_: Universal table pose, task-specific table offset

**Workcell Fixture**:
An immovable, collision-bearing component whose initial pose is declared by the Task Layout, such as a sorting box, Pegboard Rack, or Tool Holder.
_Avoid_: Background object, scenery

**Layout Anchor**:
A fixed reference object whose authored pose is an input to relation solving rather than an output of it. The Canonical Table is the Layout Anchor for table-supported fixtures.
Exact-pose Kinematic Fixtures are also Layout Anchors when their authored assembly intentionally intersects another fixture.
_Avoid_: Dynamic prop, solver output

**Kinematic Fixture**:
A Workcell Fixture that remains immovable under episode physics. Its exact Task Layout pose may make it a Layout Anchor; a relation-solved fixture remains kinematic after placement.
_Avoid_: Dynamic prop

**Task Layout**:
The declarative arrangement of workcell fixtures and manipulable objects around the robot and Canonical Table.
_Avoid_: Scene setup, baked background

**Pegboard Rack**:
The shared four-panel pegboard Workcell Fixture used by tool-hanging tasks.
_Avoid_: Tool hanging rack, rack background

**Tool Holder**:
A task-specific Workcell Fixture mounted on the Pegboard Rack to support a particular hanging tool, such as a hook or screwdriver rack.
_Avoid_: Tool hanger, holder background

**Hand-Authored Asset**:
A benchmark-owned asset maintained directly as part of the industrial task definition rather than regenerated from a converter.
_Avoid_: Generated overlay, converter output
