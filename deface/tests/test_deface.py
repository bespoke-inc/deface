import unittest
import numpy as np

from deface.deface import (
    has_overlap,
    has_overlap_with_union,
    unionize_overlapping_dets,
    get_union_rep,
    filter_by_dets_history,
    ThresholdTimeline
)


class TestDeface(unittest.TestCase):
    def test_has_overlap(self):
        # THEN
        self.assertEqual(has_overlap([0, 0, 10, 10, 0.5], [15, 15, 25, 25, 0.5]),
            False, "Should be false as there is no overlap in both X and Y axis")
        self.assertEqual(has_overlap([0, 0, 10, 10, 0.5], [0, 20, 10, 30, 0.5]),
            False, "Should be false as the later rectangle is below the former without overlap")
        self.assertEqual(has_overlap([0, 0, 10, 10, 0.5], [20, 0, 30, 10, 0.5]),
            False, "Should be false as the later is on the right of the former without overlap")
        self.assertEqual(has_overlap([0, 0, 10, 10, 0.5], [5, 5, 15, 15, 0.5]),
            True, "Should be true as overlap with both X and Y axis")
        self.assertEqual(has_overlap([0, 0, 10, 10, 0.5], [0, 0, 10, 10, 0.5]),
            True, "Should be true as the same rectangle")

    def test_has_overlap_with_union(self):
        # GIVEN
        test_det = [100, 100, 200, 200, 0.5]
        overlap_union = [
            [50, 50, 150, 150, 0.5],
            [300, 300, 400, 400, 0.5]]

        not_overlap_union = [
            [100, 250, 200, 350, 0.5],
            [300, 100, 400, 200, 0.5],
            [250, 250, 350, 350, 0.5]]

        # THEN
        self.assertEqual(has_overlap_with_union(test_det, overlap_union),
            True, "should overlap as there is one overlap rectangle")
        self.assertEqual(has_overlap_with_union(test_det, not_overlap_union),
            False, "should not overlap as the rectangles are below, right side and diagonal without overlap")

    def test_unionize_overlapping_dets(self):
        # GIVEN
        # 2 overlapping
        dets_1 = [
            [100, 100, 200, 200, 0.5],
            [150, 150, 190, 190, 0.25],
            [500, 500, 700, 700, 0.25],
        ]
        # none overlapping
        dets_2 = [
            [0, 0, 100, 100, 0.5],
            [150, 150, 190, 190, 0.25],
            [500, 500, 700, 700, 0.25],
        ]
        # all overlapping
        dets_3 = [
            [0, 0, 100, 100, 0.5],
            [50, 50, 90, 90, 0.25],
            [50, 50, 200, 200, 0.25],
        ]

        # THEN
        self.assertEqual(len(unionize_overlapping_dets(dets_1)), 2, "2 overlapping among 3, should produce 2 unions")
        self.assertEqual(len(unionize_overlapping_dets(dets_2)), 3, "non overlapping among 3, should produce 3 unions")
        self.assertEqual(len(unionize_overlapping_dets(dets_3)), 1, "all overlapping among 3, should produce 1 union ")

    def test_get_union_rep(self):
        # WHEN
        # 2 unions, should produce 2 representatives
        unions_1 = [
            [
                [100, 100, 200, 200, 0.5],
                [150, 150, 190, 190, 0.25]
            ],
            [
                [500, 500, 700, 700, 0.25]
            ]
        ]

        # 3 unions, should produce 3 presentatives
        # Since each union only have 1 member, they are their own representatives.
        unions_2 = [
            [
                [100, 100, 200, 200, 0.5]
            ],
            [
                [250, 250, 290, 290, 0.25]
            ],
            [
                [500, 500, 700, 700, 0.25]
            ]
        ]

        # 1 union, should produce only 1 representative
        unions_3 = [
            [
                [100, 100, 200, 200, 0.5],
                [150, 150, 190, 190, 0.25],
                [500, 500, 700, 700, 0.25]
            ]
        ]


        reps_1 = get_union_rep(unions_1)
        reps_2 = get_union_rep(unions_2)
        reps_3 = get_union_rep(unions_3)

        # THEN
        self.assertTrue(len(reps_1) == 2, "2 unions, should produce 2 representatives")
        self.assertTrue((reps_1[0] == np.asarray([102, 102, 200, 200, 0.5], dtype=np.float32)).all(),
            "incorrect representative calculation for the first union")

        self.assertTrue(len(reps_2) == 3, "3 unions, should produce 3 presentatives")
        self.assertTrue(
            (
                reps_2
                == np.asarray(
                    [
                        [100, 100, 200, 200, 0.5],
                        [250, 250, 290, 290, 0.25],
                        [500, 500, 700, 700, 0.25],
                    ],
                    dtype=np.float32,
                )
            ).all(),
            "the union members should be their own representatives",
        )

        self.assertTrue(len(reps_3) == 1, "1 union, should produce only 1 representative")
        self.assertTrue((reps_3 == np.asarray([[399, 399, 599, 599, 0.5]], dtype=np.float32)).all(),
            "incorrect representative calculation for the only union",
        )

    def test_filter_by_dets_history(self):
        # GIVEN
        # new detection that also appears 3 times before
        new_dets_1 = np.array([[100, 105, 200, 205, 0.5]], dtype=np.float32)
        # new detection that  appears 2 times before
        new_dets_2 = np.array([[10, 10, 20, 20, 0.5]], dtype=np.float32)
        # new detection that never appears before
        new_dets_3 = np.array([[1000, 1000, 2000, 2000, 0.5]], dtype=np.float32)


        # A history of previous detections,
        history = [
            [
                [
                    [10, 10, 20, 20, 0.2],
                    [5, 5, 25, 25, 0.2],
                ],
                [
                    [100, 100, 200, 200, 0.2], # reliable
                    [75, 75, 125, 125, 0.2],
                ],
            ],
            [
                [
                    [105, 105, 205, 205, 0.2], # reliable
                ],
            ],
            [
                [
                    [210, 210, 220, 220, 0.2],
                    [110, 110, 210, 210, 0.2], # reliable
                    [85, 85, 125, 125, 0.2],
                ],
            ],
            [
                []
            ],
            [
                [
                    [10, 10, 20, 20, 0.2],
                ],
            ],
        ]

        # THEN
        self.assertEqual(
            len(filter_by_dets_history(new_dets_1, history.copy(), 3)[0]), 1,
            "there should be one reliable detection as a result of the consistency in history"
        )
        self.assertEqual(
            len(filter_by_dets_history(new_dets_2, history.copy(), 3)[0]), 0,
            "there should be no reliable detection due to high consistency_threshold"
        )
        self.assertEqual(
            len(filter_by_dets_history(new_dets_3, history.copy(), 0)[0]), 1,
            "the new detection is never seen before but there is no consistency_threshold so it becomes reliable "
        )
        self.assertEqual(
            len(filter_by_dets_history(new_dets_3, history.copy(), 1)[0]), 0,
            "the same new detection that is never seen before now is reject due to the consistency_threshold is set to 1 "
        )

    def test_threshold_timeline(self):
        # GIVEN
        thresholds_timeline_empty = ThresholdTimeline({}, 0.5, 1)
        thresholds_timeline_normal = ThresholdTimeline({1: 0.2, 5: 0.6}, 0.5, 2)

        # THEN
        self.assertTrue(thresholds_timeline_empty.thresholds == {0: 0.5},
            "even an empty timeline should have at threshold starting at frame 0")
        self.assertTrue(thresholds_timeline_normal.thresholds == {0: 0.5, 2: 0.2, 10: 0.6},
            "normal threshold timeline should be reflected")
        self.assertTrue(thresholds_timeline_normal.fps == 2,
            "fps must match")
        self.assertTrue(thresholds_timeline_normal.default_threshold == 0.5,
            "default threshold must match")

        self.assertTrue(thresholds_timeline_normal.threshold_for_frame(1) == 0.5, "2 > frame 1 > 0 so it should have a threshold of 0.5")
        self.assertTrue(thresholds_timeline_normal.threshold_for_frame(2) == 0.2, "frame 2 == 2, should have a threshold of 0.2")
        self.assertTrue(thresholds_timeline_normal.threshold_for_frame(100) == 0.6, "frame 100 > 10, should have a threshold of 0.6")


