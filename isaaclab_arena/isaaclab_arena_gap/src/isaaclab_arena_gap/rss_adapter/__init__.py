"""Run AUTOLab's GaP graphs on Arena tasks through RoboSimStudio's own policy agent.

AUTOLab owns the graphs and the runtime (RoboSimStudio's ``SolutionGraph``,
connector, IK, servo and tools); Arena owns the world and the score. This
package is only the seam between them: an Env that RoboSimStudio's runtime
drives, whose robot is Arena's asset carrying AUTOLab's registered hand, whose
time and state are Arena's, and which holds no simulated object a graph could
read. It runs in AUTOLab's own environment (``AUTOLAB_VABAR_VENV``, from the sources
:func:`~isaaclab_arena_gap.task_sources.autolab_root` resolves), launched by :class:`~isaaclab_arena_gap.gap_remote_policy.GapRemotePolicy`
through ``scripts/run_rss_graph.sh``.
"""
