from __future__ import annotations

import time


def test_logger_records_failed_episode98(monkeypatch, reload_module):
    logger = reload_module("logger")

    calls = []

    def _fake_log_user_action_hdf5(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(logger, "log_user_action_hdf5", _fake_log_user_action_hdf5)

    logger.log_user_action(
        username="user1",
        env_id="BinFill",
        episode_idx=98,
        action_data={"done": True, "status": "FAILED"},
        status="FAILED",
    )

    assert len(calls) == 1


def test_load_next_task_wrapper_treats_episode98_as_normal(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")

    create_calls = []
    expected = ("SENTINEL",)

    monkeypatch.setattr(callbacks.user_manager, "assert_lease", lambda username, uid: None)
    monkeypatch.setattr(callbacks, "update_session_activity", lambda uid: None)
    monkeypatch.setattr(
        callbacks.user_manager,
        "login",
        lambda username, uid: (
            True,
            "ok",
            {"is_done_all": False, "current_task": {"env_id": "BinFill", "episode_idx": 98}},
        ),
    )
    monkeypatch.setattr(callbacks, "has_existing_actions", lambda username, env_id, ep_num: True)
    monkeypatch.setattr(
        callbacks,
        "create_new_attempt",
        lambda username, env_id, ep_num: create_calls.append((username, env_id, ep_num)),
    )
    monkeypatch.setattr(callbacks, "login_and_load_task", lambda username, uid: expected)

    result = callbacks.load_next_task_wrapper("user1", "uid1")

    assert create_calls == [("user1", "BinFill", 98)]
    assert result == expected


def test_execute_step_failed_episode98_still_advances(monkeypatch, reload_module):
    callbacks = reload_module("gradio_callbacks")

    class _FakeSession:
        def __init__(self):
            self.env_id = "BinFill"
            self.episode_idx = 98
            self.base_frames = []
            self.raw_solve_options = [{"available": False}]
            self.available_options = [("run", 0)]
            self.difficulty = "hard"
            self.language_goal = "goal"
            self.seed = 123
            self.non_demonstration_task_length = None

        def update_observation(self, use_segmentation=False):
            return None

        def get_pil_image(self, use_segmented=False):
            return "IMG"

        def execute_action(self, option_idx, click_coords):
            return "IMG", "FAILED", True

    fake_session = _FakeSession()
    complete_calls = []

    monkeypatch.setattr(callbacks, "get_session_activity", lambda uid: time.time())
    monkeypatch.setattr(callbacks, "update_session_activity", lambda uid: None)
    monkeypatch.setattr(callbacks.user_manager, "assert_lease", lambda username, uid: None)
    monkeypatch.setattr(callbacks, "get_session", lambda uid: fake_session)
    monkeypatch.setattr(callbacks.FrameQueueManager, "init_queue", lambda uid, count: None)
    monkeypatch.setattr(callbacks, "_wait_for_livestream_drain", lambda uid: None)
    monkeypatch.setattr(callbacks, "get_option_selects", lambda uid: [])
    monkeypatch.setattr(callbacks, "clear_option_selects", lambda uid: None)
    monkeypatch.setattr(callbacks, "get_coordinate_clicks", lambda uid: [])
    monkeypatch.setattr(callbacks, "clear_coordinate_clicks", lambda uid: None)
    monkeypatch.setattr(callbacks, "increment_execute_count", lambda username, env_id, episode_idx: 1)
    monkeypatch.setattr(callbacks, "log_user_action", lambda *args, **kwargs: None)

    def _fake_complete_current_task(*args, **kwargs):
        payload = dict(kwargs)
        if args:
            payload["username"] = args[0]
        complete_calls.append(payload)
        return {"is_done_all": False, "current_task": {"env_id": "MoveCube", "episode_idx": 7}}

    monkeypatch.setattr(callbacks.user_manager, "complete_current_task", _fake_complete_current_task)

    result = callbacks.execute_step("uid1", "user1", 0, "No need for coordinates")

    assert len(complete_calls) == 1
    assert complete_calls[0]["episode_idx"] == 98
    assert complete_calls[0]["status"] == "failed"
    assert result[2] == "Task Completed! Next: MoveCube (Ep 7)"
