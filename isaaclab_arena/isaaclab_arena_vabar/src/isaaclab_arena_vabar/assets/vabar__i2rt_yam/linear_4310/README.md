# linear_4310 hand

The YAM's `linear_4310` gripper, selected through the `gripper` variant set of
`../i2rt_yam_default.usda` (default `crank_4310`). AUTOLab's cable graphs are
written against this hand.

Copied unchanged from the generated robot USD of AUTOLab's
`assets/embodiments/linear_4310` MJCF. The variant block in `../i2rt_yam_default.usda`
follows the generated asset's, without its MJCF actuator; Isaac Lab drives the fingers.
It selects the hand's `physics` flavour, so nothing couples the two fingers (AUTOLab
drives each on its own), and adds `mjc:gravcomp` to the hand bodies, the name Newton
reads (the generated hand only has `mjc:body:gravcomp`).

The hand's `grasp_site` marks the fingertip end, 144.7 mm along `link_6`. AUTOLab's
grasp site, and the TCP used here, is 134.7 mm, the same point as the crank hand's.
