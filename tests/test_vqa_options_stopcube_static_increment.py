"""
StopCube 任务中「remain static」选项的单元测试。

测试 vqa_options._options_stopcube 里「保持静止」选项的步数递增与饱和逻辑：
- 每次选择「remain static」时，内部会调用 solve_hold_obj_absTimestep(absTimestep)；
- absTimestep 按 100 -> 200 -> ... 递增，且不超过 final_target（由 steps_press、interval 算出）；
- 若 env.elapsed_steps 回退，内部状态应重置，下次再从 100 开始。
"""
from pathlib import Path
import importlib.util
import sys
import types


# 与 vqa_options 所依赖的 planner 模块中需存在的符号一致，用于构造 stub
PLANNER_SYMBOLS = [
    "grasp_and_lift_peg_side",
    "insert_peg",
    "solve_button",
    "solve_button_ready",
    "solve_hold_obj",
    "solve_hold_obj_absTimestep",
    "solve_pickup",
    "solve_pickup_bin",
    "solve_push_to_target",
    "solve_push_to_target_with_peg",
    "solve_putdown_whenhold",
    "solve_putonto_whenhold",
    "solve_putonto_whenhold_binspecial",
    "solve_swingonto",
    "solve_swingonto_withDirection",
    "solve_swingonto_whenhold",
    "solve_strong_reset",
]


def _load_vqa_options_module():
    """注入 stub 的 planner 模块并加载 vqa_options，返回 (模块, hold_calls 列表)。"""
    hold_calls = []

    planner_stub = types.ModuleType("robomme.robomme_env.utils.planner")

    def _noop(*args, **kwargs):
        return None

    for symbol in PLANNER_SYMBOLS:
        setattr(planner_stub, symbol, _noop)

    def _hold_spy(env, planner, absTimestep):
        """记录每次「remain static」实际传入的 absTimestep，用于断言。"""
        hold_calls.append(int(absTimestep))
        return None

    planner_stub.solve_hold_obj_absTimestep = _hold_spy

    robomme_pkg = types.ModuleType("robomme")
    robomme_pkg.__path__ = []
    robomme_env_pkg = types.ModuleType("robomme.robomme_env")
    robomme_env_pkg.__path__ = []
    utils_pkg = types.ModuleType("robomme.robomme_env.utils")
    utils_pkg.__path__ = []

    robomme_pkg.robomme_env = robomme_env_pkg
    robomme_env_pkg.utils = utils_pkg
    utils_pkg.planner = planner_stub

    injected = {
        "robomme": robomme_pkg,
        "robomme.robomme_env": robomme_env_pkg,
        "robomme.robomme_env.utils": utils_pkg,
        "robomme.robomme_env.utils.planner": planner_stub,
    }
    previous = {key: sys.modules.get(key) for key in injected}
    sys.modules.update(injected)

    try:
        repo_root = Path(__file__).resolve().parents[1]
        module_path = repo_root / "src" / "robomme" / "robomme_env" / "utils" / "vqa_options.py"
        spec = importlib.util.spec_from_file_location("vqa_options_under_test", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for key, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = old_module

    return module, hold_calls


class _DummyEnv:
    """模拟环境，仅提供 elapsed_steps 供 _options_stopcube 使用。"""
    def __init__(self, elapsed_steps=0):
        self.elapsed_steps = elapsed_steps


class _DummyBase:
    """模拟 StopCube 的 base，提供 steps_press、interval，用于计算 final_target。"""
    def __init__(self, steps_press, interval=30):
        self.steps_press = steps_press
        self.interval = interval
        self.button = object()


def _get_remain_static_solver(options):
    """从 StopCube 选项列表中取出 label 为 'remain static' 的 solve 函数。"""
    for option in options:
        if option.get("label") == "remain static":
            return option["solve"]
    raise AssertionError("Missing 'remain static' option")


def test_stopcube_remain_static_increment_and_saturation():
    """测试：多次选「remain static」时，absTimestep 按 100→200→… 递增，且不超过 final_target 并饱和。"""
    module, hold_calls = _load_vqa_options_module()
    env = _DummyEnv(elapsed_steps=0)
    base = _DummyBase(steps_press=270, interval=30)  # final_target = 240
    options = module._options_stopcube(env, planner=None, require_target=lambda: None, base=base)
    solve_remain_static = _get_remain_static_solver(options)

    for _ in range(4):
        solve_remain_static()

    assert hold_calls == [100, 200, 240, 240]


def test_stopcube_remain_static_small_final_target():
    """测试：final_target 较小时（如 60），首次即达上限，后续调用保持 60 不变（饱和）。"""
    module, hold_calls = _load_vqa_options_module()
    env = _DummyEnv(elapsed_steps=0)
    base = _DummyBase(steps_press=90, interval=30)  # final_target = 60
    options = module._options_stopcube(env, planner=None, require_target=lambda: None, base=base)
    solve_remain_static = _get_remain_static_solver(options)

    solve_remain_static()
    solve_remain_static()

    assert hold_calls == [60, 60]


def test_stopcube_remain_static_resets_when_elapsed_steps_go_back():
    """测试：elapsed_steps 被改小（如从 150 改回 0）时，内部步数状态应重置，下次再从 100 开始。"""
    module, hold_calls = _load_vqa_options_module()
    env = _DummyEnv(elapsed_steps=0)
    base = _DummyBase(steps_press=270, interval=30)  # final_target = 240
    options = module._options_stopcube(env, planner=None, require_target=lambda: None, base=base)
    solve_remain_static = _get_remain_static_solver(options)

    solve_remain_static()  # 100
    env.elapsed_steps = 150
    solve_remain_static()  # 200
    env.elapsed_steps = 0
    solve_remain_static()  # reset -> 100

    assert hold_calls == [100, 200, 100]


def test_stopcube_option_label_order_stays_stable():
    """测试：StopCube 选项的 label 顺序固定为：先准备、再保持静止、最后按按钮。"""
    module, _ = _load_vqa_options_module()
    env = _DummyEnv(elapsed_steps=0)
    base = _DummyBase(steps_press=270, interval=30)
    options = module._options_stopcube(env, planner=None, require_target=lambda: None, base=base)

    labels = [option.get("label") for option in options]
    assert labels == [
        "move to the top of the button to prepare",
        "remain static",
        "press button to stop the cube",
    ]
