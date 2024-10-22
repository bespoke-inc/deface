import unittest
import numpy as np

from deface.deface import (
    has_overlap,
    has_overlap_with_union,
    unionize_overlapping_dets,
    get_union_rep,
    filter_by_dets_history,
)


class TestDeface(unittest.TestCase):
    def setUp(self):
        # A detection have [x1, y1, x2, y2, score]
        # Detect three faces, 2 overlapping
        self.dets_1 = [
            [100, 100, 200, 200, 0.5],
            [150, 150, 190, 190, 0.25],
            [500, 500, 700, 700, 0.25],
        ]
        # Detect three faces, none overlapping
        self.dets_2 = [
            [0, 0, 100, 100, 0.5],
            [150, 150, 190, 190, 0.25],
            [500, 500, 700, 700, 0.25],
        ]
        # Detect three faces, all overlapping
        self.dets_3 = [
            [0, 0, 100, 100, 0.5],
            [50, 50, 90, 90, 0.25],
            [50, 50, 200, 200, 0.25],
        ]

        # A history of previous detections, there is only 1 reliable detection
        # [100, 100, 200, 200] which appears 3 times
        self.history = [
            [
                [
                    [10, 10, 20, 20, 0.2],
                    [5, 5, 25, 25, 0.2],
                ],
                [
                    [100, 100, 200, 200, 0.2],
                    [75, 75, 125, 125, 0.2],
                ],
            ],
            [
                [
                    [300, 300, 400, 400, 0.2],
                    [500, 500, 625, 625, 0.2],
                ],
                [
                    [105, 105, 205, 205, 0.2],
                    [65, 65, 120, 120, 0.2],
                ],
            ],
            [
                [
                    [210, 210, 220, 220, 0.2],
                ],
                [
                    [110, 110, 210, 210, 0.2],
                    [85, 85, 125, 125, 0.2],
                ],
            ],
            [
                [
                    [310, 310, 320, 320, 0.2],
                ],
            ],
            [
                [
                    [10, 10, 20, 20, 0.2],
                ],
            ],
        ]

    def test_has_overlap(self):
        # WHEN
        is_overlapped_1_1 = has_overlap(self.dets_1[0], self.dets_1[1])
        is_overlapped_1_2 = has_overlap(self.dets_1[0], self.dets_1[2])
        is_overlapped_1_3 = has_overlap(self.dets_1[1], self.dets_1[2])

        is_overlapped_2_1 = has_overlap(self.dets_2[0], self.dets_2[1])
        is_overlapped_2_2 = has_overlap(self.dets_2[0], self.dets_2[2])
        is_overlapped_2_3 = has_overlap(self.dets_2[1], self.dets_2[2])

        is_overlapped_3_1 = has_overlap(self.dets_3[0], self.dets_3[1])
        is_overlapped_3_2 = has_overlap(self.dets_3[0], self.dets_3[2])
        is_overlapped_3_3 = has_overlap(self.dets_3[1], self.dets_3[2])

        # THEN
        self.assertEqual(is_overlapped_1_1, True, "incorrect overlap assertion")
        self.assertEqual(is_overlapped_1_2, False, "incorrect overlap assertion")
        self.assertEqual(is_overlapped_1_3, False, "incorrect overlap assertion")

        self.assertEqual(is_overlapped_2_1, False, "incorrect overlap assertion")
        self.assertEqual(is_overlapped_2_2, False, "incorrect overlap assertion")
        self.assertEqual(is_overlapped_2_3, False, "incorrect overlap assertion")

        self.assertEqual(is_overlapped_3_1, True, "incorrect overlap assertion")
        self.assertEqual(is_overlapped_3_2, True, "incorrect overlap assertion")
        self.assertEqual(is_overlapped_3_3, True, "incorrect overlap assertion")

    def test_has_overlap_with_union(self):
        # GIVEN
        test_dets = [400, 400, 600, 600, 0.5]

        # WHEN
        is_overlapped_1 = has_overlap_with_union(test_dets, self.dets_1)
        is_overlapped_2 = has_overlap_with_union(test_dets, self.dets_2)
        is_overlapped_3 = has_overlap_with_union(test_dets, self.dets_3)

        # THEN
        self.assertEqual(is_overlapped_1, True, "incorrect overlap assertion")
        self.assertEqual(is_overlapped_2, True, "incorrect overlap assertion")
        self.assertEqual(is_overlapped_3, False, "incorrect overlap assertion")

    def test_unionize_overlapping_dets(self):
        # WHEN
        groups_1 = unionize_overlapping_dets(self.dets_1)
        groups_2 = unionize_overlapping_dets(self.dets_2)
        groups_3 = unionize_overlapping_dets(self.dets_3)

        # THEN
        self.assertEqual(len(groups_1), 2, "incorrect number of groups")
        self.assertEqual(len(groups_2), 3, "incorrect number of groups")
        self.assertEqual(len(groups_3), 1, "incorrect number of groups")

    def test_get_union_rep(self):
        # WHEN
        groups_1 = unionize_overlapping_dets(self.dets_1)
        groups_2 = unionize_overlapping_dets(self.dets_2)
        groups_3 = unionize_overlapping_dets(self.dets_3)

        centeroid_1 = get_union_rep(groups_1)
        centeroid_2 = get_union_rep(groups_2)
        centeroid_3 = get_union_rep(groups_3)

        # THEN
        self.assertTrue(
            (
                centeroid_1
                == np.asarray(
                    [
                        [102, 102, 200, 200, 0.5],
                        [500, 500, 700, 700, 0.25],
                    ],
                    dtype=np.float32,
                )
            ).all(),
            "incorrect centeroid calculation",
        )
        self.assertTrue(
            (
                centeroid_2
                == np.asarray(
                    [
                        [0, 0, 100, 100, 0.5],
                        [150, 150, 190, 190, 0.25],
                        [500, 500, 700, 700, 0.25],
                    ],
                    dtype=np.float32,
                )
            ).all(),
            "incorrect centeroid calculation",
        )
        self.assertTrue(
            (
                centeroid_3 == np.asarray([[25, 25, 175, 175, 0.5]], dtype=np.float32)
            ).all(),
            "incorrect centeroid calculation",
        )

    def test_filter_by_dets_history(self):
        # WHEN
        new_dets_1, _ = filter_by_dets_history(self.dets_1, self.history.copy(), 3)
        new_dets_2, _ = filter_by_dets_history(self.dets_2, self.history.copy(), 5)
        new_dets_3, _ = filter_by_dets_history(self.dets_3, self.history.copy(), 0)

        # THEN
        self.assertEqual(
            len(new_dets_1), 1, "incorrect number of detection after filtering"
        )
        self.assertEqual(
            len(new_dets_2), 0, "incorrect number of detection after filtering"
        )
        self.assertEqual(
            len(new_dets_3), 1, "incorrect number of detection after filtering"
        )
