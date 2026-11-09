# SPDX-License-Identifier: Apache-2.0
"""What the two cable suites need that upstream RoboSimStudio does not ship.

* :mod:`.curve_tools` -- the unprivileged ``curve.*`` GaP tools (a mask and a
  depth frame in, an ordered centreline out) the routing graphs' perception
  nodes call. Registered by ``tasks/cable_route/tools.py``.
* :mod:`.trackdlo_skeleton` -- the TrackDLO skeleton ordering those tools are
  built on, vendored under its own MIT licence (``LICENSE.trackdlo``).

Both come from the RoboSimStudio revision the suites were written against
(KeplerC/RoboSimStudio@eeb2aa88); each file says what was edited.
"""
