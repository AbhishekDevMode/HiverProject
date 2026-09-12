import tempfile
import unittest
from pathlib import Path

from src.eval_harness import load_golden


class LoadGoldenFallbackTest(unittest.TestCase):
    def test_load_golden_creates_missing_golden_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_path = Path(tmp) / "sample_golden_set.json"
            records = load_golden(missing_path)

            self.assertEqual(len(records), 200)
            self.assertTrue(missing_path.exists())
            self.assertIn("tweet_id", records[0])
            self.assertIn("incoming_text", records[0])
            self.assertIn("ground_truth_intent", records[0])
            self.assertIn("ground_truth_escalate", records[0])
            self.assertIn("reference_reply", records[0])


if __name__ == "__main__":
    unittest.main()
