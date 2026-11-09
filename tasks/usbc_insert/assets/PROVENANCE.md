# `usbc/` — provenance

Moved here from the repository's `assets/PROVENANCE.md` when the meshes
moved, and for the same reason: a folder that is the whole asset has to
carry its own attribution, or sharing it means sharing something whose
licence and origin live in a file that did not come along.

---

## `usbc/` — USB-C plug and receptacle meshes

**Source:** `uksangyoo/newton_tasks` @ `2b081aa`,
`Assets/usb_c_mesh/{USB_C_Plug.stl,USB_C_Port.stl}`. Byte-identical copies,
authored in metres; the suite applies its ×2.0 scale at load.

The receptacle's bore is a **real modelled cavity** in the mesh, which is the
whole point: the suite collides it through Newton's exact mesh narrow phase
(`use_mujoco_contacts=False`), because MuJoCo's convexification would fill the
cavity and no plug could ever enter. Nominal clearance at ×2 scale is
0.15 mm × 0.20 mm per side; bore depth 13 mm.

---
