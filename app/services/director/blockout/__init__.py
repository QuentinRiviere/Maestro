"""Deterministic 3D blockout of storyboard shots (see docs/Director-blockout.md).

``layout``: a location's rough 3D layout (room, openings, furniture) and its validation.
``scene``: the shot's set, people and camera from the scene-locked blocking map; camera moves.
``render``: the CPU ray caster (depth, semantic colours, normals, edges).
``guide``: the Director hook, off by default (``director_blockout_guide``).
"""
