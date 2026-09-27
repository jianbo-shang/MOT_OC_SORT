import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate_mot17 import motchallenge_row, read_sequence_info
from mot_core import Detection, Track


class Mot17EvaluationTests(unittest.TestCase):
    def test_read_sequence_info(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seqinfo.ini"
            path.write_text(
                "[Sequence]\nname=fixture\nimDir=img1\nframeRate=30\n"
                "seqLength=12\nimWidth=640\nimHeight=480\nimExt=.jpg\n",
                encoding="utf-8",
            )
            info = read_sequence_info(path)
            self.assertEqual(info["seqLength"], 12)
            self.assertEqual(info["imDir"], "img1")

    def test_motchallenge_row_uses_xywh_and_clips_bounds(self):
        detection = Detection(np.array([-5, 10, 110, 95], dtype=np.float32), 0.8, 0)
        track = Track.create(3, detection, 8)
        row = motchallenge_row(7, track, 100, 80)
        self.assertIsNotNone(row)
        self.assertEqual(row[0:2], (7, 3))
        self.assertEqual(row[2:6], (0.0, 10.0, 100.0, 70.0))
        self.assertEqual(row[7:], (-1, -1, -1))


if __name__ == "__main__":
    unittest.main()
