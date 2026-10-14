"""Named Qwen Image 2.1 sampling profiles: trained sigma nodes and scheduler overrides.

Kept free of model imports so the schedules can be checked without the pipeline.
"""


# These are the published Viggle Turbo schedules used by Wan2GP. Keep the
# accelerated paths explicit: using a few steps with the base schedule does
# not reproduce the distilled adapter's training trajectory.
VIGGLE_SIGMAS = {
    4: (1.0, 0.75, 0.5, 0.25),
    5: (1.0, 0.875, 0.75, 0.5, 0.25),
    6: (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25),
}
_VIGGLE_PROFILE_STEPS = {"viggle_v01": 4, "viggle_v02": 5, "viggle_v021": 6}

# The official Qwen-Image-2.1-Turbo checkpoint ships its sampling schedule as
# `sample_sigmas` in model_index.json (Qwen/Qwen-Image-2.1-Turbo at revision
# fc13ba1986fcb905854dc8de2f52eee395f49b5b). Its scheduler_config.json turns
# dynamic shifting and the terminal stretch off with shift=1.0, so the nodes
# reach the Euler sampler unchanged and the scheduler appends the final 0.
QWEN_TURBO_PROFILE = "qwen_turbo_8step"
QWEN_TURBO_SIGMAS = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)
QWEN_TURBO_SCHEDULER_OVERRIDES = {"use_dynamic_shifting": False, "shift": 1.0, "shift_terminal": None}


def qwen21_sampling_profile(sample_solver: str, num_inference_steps: int):
    """Return scheduler overrides and sigma nodes for a named Qwen profile."""
    solver = (sample_solver or "default").lower()
    if solver in ("", "default", "euler"):
        return {}, None
    if solver == QWEN_TURBO_PROFILE:
        if int(num_inference_steps) != len(QWEN_TURBO_SIGMAS):
            raise ValueError(f"{solver} requires exactly {len(QWEN_TURBO_SIGMAS)} inference steps.")
        return dict(QWEN_TURBO_SCHEDULER_OVERRIDES), QWEN_TURBO_SIGMAS
    if solver in _VIGGLE_PROFILE_STEPS:
        required = _VIGGLE_PROFILE_STEPS[solver]
        if int(num_inference_steps) != required:
            raise ValueError(f"{solver} requires exactly {required} inference steps.")
        return {"shift_terminal": None}, VIGGLE_SIGMAS[int(num_inference_steps)]
    raise ValueError(f"Unknown Qwen Image 2.1 sampler profile: {sample_solver!r}.")
