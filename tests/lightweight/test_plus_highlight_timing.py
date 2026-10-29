"""Temporal constraints and reproducibility for both PickHighlight cue modes."""

import unittest

from robomme_plus.tasks.reference.utils.highlight_timing import (
    highlight_schedule,
    resolve_highlight_mode,
)
from robomme_plus.tasks.reference.utils.highlight_levels import DURATION_STEPS


class HighlightTimingTest(unittest.TestCase):
    def test_simultaneous_targets_share_start_end_and_level_duration(self):
        for duration_range in DURATION_STEPS.values():
            low, high = ((duration_range, duration_range)
                         if isinstance(duration_range, int) else duration_range)
            seen = set()
            for seed in range(300):
                schedule = highlight_schedule(
                    seed, 9, duration_steps=duration_range, mode="simultaneous"
                )
                self.assertEqual(len(schedule), 9)
                self.assertTrue(all(entry == schedule[0] for entry in schedule))
                self.assertEqual(schedule[0]["start_step"], 10)
                self.assertLess(schedule[0]["end_step"], 120)
                duration = schedule[0]["end_step"] - schedule[0]["start_step"] + 1
                self.assertTrue(low <= duration <= high)
                seen.add(duration)
                self.assertEqual(schedule, highlight_schedule(
                    seed, 9, duration_steps=duration_range, mode="simultaneous"
                ))
            self.assertEqual(seen, set(range(low, high + 1)))

    def test_seeded_mode_choice_is_repeatable_and_approximately_balanced(self):
        modes = [resolve_highlight_mode(seed) for seed in range(1000)]
        self.assertEqual(modes, [resolve_highlight_mode(seed) for seed in range(1000)])
        self.assertTrue(450 <= modes.count("simultaneous") <= 550)
        for seed, mode in enumerate(modes[:50]):
            self.assertEqual(
                highlight_schedule(seed, 7, mode="random"),
                highlight_schedule(seed, 7, mode=mode),
            )
            for fixed_mode in ("simultaneous", "staggered"):
                self.assertEqual(resolve_highlight_mode(seed, fixed_mode), fixed_mode)

    def test_simultaneous_mode_does_not_require_room_for_staggering(self):
        self.assertEqual(highlight_schedule(
            0, 9, start_step=10, end_step=54, mode="simultaneous"
        ), [{"start_step": 10, "end_step": 54}] * 9)
        with self.assertRaises(ValueError):
            highlight_schedule(0, 9, end_step=53, mode="simultaneous")

    def test_invalid_modes_are_rejected(self):
        for mode in (None, "", "flashing", True):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "highlight_mode"):
                highlight_schedule(0, 7, mode=mode)

    def test_fixed_duration_has_unique_windows_and_no_common_overlap(self):
        for count in (3, 4, 5, 6, 7, 8, 9):
            for seed in range(200):
                schedule = highlight_schedule(seed, count)
                starts = [entry["start_step"] for entry in schedule]
                ends = [entry["end_step"] for entry in schedule]
                self.assertEqual(len(set(starts)), count)
                self.assertEqual(len(set(ends)), count)
                self.assertGreaterEqual(min(starts), 10)
                self.assertLess(max(ends), 120)
                self.assertTrue(all(end - start + 1 == 45 for start, end in zip(starts, ends)))
                for step in range(121):
                    active = sum(start <= step <= end for start, end in zip(starts, ends))
                    self.assertLess(active, count)
                self.assertEqual(schedule, highlight_schedule(seed, count))

    def test_easy_keeps_existing_timing(self):
        self.assertEqual(highlight_schedule(0, 4, duration_steps=DURATION_STEPS['easy']), [
            {'start_step': 40, 'end_step': 84},
            {'start_step': 56, 'end_step': 100},
            {'start_step': 19, 'end_step': 63},
            {'start_step': 65, 'end_step': 109},
        ])

    def test_variable_durations_preserve_staggering_and_cover_inclusive_ranges(self):
        for level, counts in (('medium', (4, 5, 6)), ('hard', (5, 6, 7, 8, 9))):
            low, high = DURATION_STEPS[level]
            for count in counts:
                seen = set()
                mixed = 0
                for seed in range(300):
                    schedule = highlight_schedule(seed, count, duration_steps=(low, high))
                    starts = [entry['start_step'] for entry in schedule]
                    ends = [entry['end_step'] for entry in schedule]
                    durations = [end-start+1 for start, end in zip(starts, ends)]
                    self.assertTrue(all(low <= duration <= high for duration in durations))
                    self.assertEqual(len(set(starts)), count)
                    self.assertEqual(len(set(ends)), count)
                    self.assertGreaterEqual(min(starts), 10)
                    self.assertLess(max(ends), 120)
                    self.assertGreater(max(starts), min(ends))
                    self.assertEqual(schedule, highlight_schedule(seed, count, duration_steps=(low, high)))
                    seen.update(durations)
                    mixed += len(set(durations)) > 1
                self.assertEqual(seen, set(range(low, high+1)))
                self.assertGreater(mixed, 290)

    def test_seed_changes_timing_and_target_order(self):
        schedules = [highlight_schedule(seed, 7) for seed in range(20)]
        self.assertGreater(len({tuple(x["start_step"] for x in s) for s in schedules}), 15)
        self.assertTrue(any(s[0]["start_step"] != min(x["start_step"] for x in s) for s in schedules))

    def test_schedule_handles_one_or_two_targets(self):
        for count in (1, 2):
            schedule = highlight_schedule(0, count)
            self.assertEqual(len(schedule), count)
            self.assertTrue(all(x["end_step"] - x["start_step"] + 1 == 45 for x in schedule))
        with self.assertRaises(ValueError):
            highlight_schedule(0, 4, end_step=60)

    def test_invalid_duration_ranges_are_rejected(self):
        for duration in ((0, 45), (45, 25), (16.5, 45)):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                highlight_schedule(0, 7, duration_steps=duration)


if __name__ == "__main__":
    unittest.main()
