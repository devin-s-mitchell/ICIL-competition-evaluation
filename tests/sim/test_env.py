import numpy as np
import pytest

from icilval.rng import HashRng
from icilval.sim import perturb
from icilval.sim.libero_env import LiberoEnv, load_init_states
from icilval.sim.lighting import sample_lighting
from icilval.sim.video import VideoWriter, is_faststart

pytestmark = pytest.mark.sim


def first_base_task(pool):
    return next(t for t in pool.tasks.values() if t.suite == "libero_spatial")


def test_env_reset_observe_render_and_predicates(spec, smoke_pool):
    task = first_base_task(smoke_pool)
    env = LiberoEnv(smoke_pool.path(task.bddl), spec)
    states = load_init_states(smoke_pool.path(task.init))
    obs = env.reset(7, states[0])
    assert obs["agentview"].shape == (128, 128, 3) and obs["agentview"].dtype == np.uint8
    assert obs["ee_quat"].shape == (4,) and obs["gripper"].shape == (2,)
    assert env.success() is False and env.goal_status() == [False] * len(task.goal)
    frame = env.render()
    assert frame.shape == (spec.media["video"]["resolution"], spec.media["video"]["resolution"], 3)
    # restoring the same state twice gives the same observation
    again = env.reset(7, states[0])
    assert np.array_equal(again["agentview"], obs["agentview"])
    env.close()


def test_lighting_changes_render_and_restores(spec, smoke_pool):
    task = first_base_task(smoke_pool)
    env = LiberoEnv(smoke_pool.path(task.bddl), spec)
    states = load_init_states(smoke_pool.path(task.init))
    env.reset(7, states[0])
    base = env.render().astype(int)
    snap = perturb.snapshot_lighting(env)
    perturb.apply_lighting(env, sample_lighting(HashRng("t"), spec.axis("environment")["lighting"]))
    assert np.abs(env.render().astype(int) - base).mean() > 2.0
    perturb.restore_lighting(env, snap)
    assert np.abs(env.render().astype(int) - base).mean() < 0.5
    env.close()


def test_displacement_feasibility(spec, smoke_pool):
    task = first_base_task(smoke_pool)
    env = LiberoEnv(smoke_pool.path(task.bddl), spec)
    states = load_init_states(smoke_pool.path(task.init))
    target = env.movable_objects()[0]
    env.reset(7, states[0])
    real = perturb.displace_objects(
        env, {target: {"delta_xy": [0.08, 0.0], "yaw": 0.3}}, min_delta_m=0.05
    )
    assert real[target]["distance_m"] >= 0.05
    env.reset(7, states[0])
    with pytest.raises(perturb.Infeasible):
        perturb.displace_objects(
            env, {target: {"delta_xy": [0.9, 0.0], "yaw": 0.0}}, min_delta_m=0.05
        )
    env.reset(7, states[0])
    with pytest.raises(perturb.Infeasible):
        perturb.displace_objects(
            env, {target: {"delta_xy": [0.001, 0.0], "yaw": 0.0}}, min_delta_m=0.05
        )
    env.close()


def test_video_writer_faststart(spec, smoke_pool, tmp_path):
    task = first_base_task(smoke_pool)
    env = LiberoEnv(smoke_pool.path(task.bddl), spec)
    states = load_init_states(smoke_pool.path(task.init))
    env.reset(7, states[0])
    out = tmp_path / "clip.mp4"
    with VideoWriter(out, int(spec.media["video"]["fps"]), spec.media["video"]) as w:
        for _ in range(10):
            env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
            w.write(env.render())
    assert out.stat().st_size > 1000 and is_faststart(out)
    env.close()


def test_replay_demo_actions_succeed(spec, smoke_pool):
    """Executing a demonstration's own actions from its own initial state reproduces success."""
    task = first_base_task(smoke_pool)
    from icilval.pools.demos import load_demo

    demo = load_demo(smoke_pool.path("demos") / f"{task.demos[0]}.npz")
    env = LiberoEnv(smoke_pool.path(task.bddl), spec)
    env.reset(7, demo["init_state"])
    ok = False
    for a in demo["actions"]:
        env.step(a)
        if env.success():
            ok = True
            break
    env.close()
    assert ok
