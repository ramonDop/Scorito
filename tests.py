import unittest
from luuk_main import calculate_points
class TestCalculatePoints(unittest.TestCase):

    def test_exact_scoreline_tokai(self):
        predicted_results = "2-1"
        actual_results = "2-1"
        game = "Tokai"
        expected_points = 200
        self.assertEqual(calculate_points(predicted_results, actual_results, game), expected_points)

    def test_correct_draw_not_exact_score_tokai(self):
        predicted_results = "1-1"
        actual_results = "0-0"
        game = "Tokai"
        expected_points = 100
        self.assertEqual(calculate_points(predicted_results, actual_results, game), expected_points)

    def test_correct_winner_and_one_score_exact_tokai(self):
        predicted_results = "2-1"
        actual_results = "3-1"
        game = "Tokai"
        expected_points = 95
        self.assertEqual(calculate_points(predicted_results, actual_results, game), expected_points)

    def test_correct_winner_no_scores_exact_tokai(self):
        predicted_results = "2-1"
        actual_results = "4-2"
        game = "Tokai"
        expected_points = 75
        self.assertEqual(calculate_points(predicted_results, actual_results, game), expected_points)

    def test_one_score_correct_but_winner_not_tokai(self):
        predicted_results = "2-1"
        actual_results = "2-3"
        game = "Tokai"
        expected_points = 20
        self.assertEqual(calculate_points(predicted_results, actual_results, game), expected_points)

    def test_no_score_correct_and_winner_not_tokai(self):
        predicted_results = "2-1"
        actual_results = "1-3"
        game = "Tokai"
        expected_points = 0
        self.assertEqual(calculate_points(predicted_results, actual_results, game), expected_points)

    def test_exact_scoreline_scorito(self):
        predicted_results = "1-0"
        actual_results = "1-0"
        game = "Scorito"
        expected_points = 90
        self.assertEqual(calculate_points(predicted_results, actual_results, game), expected_points)

    def test_correct_winner_scorito(self):
        predicted_results = "1-0"
        actual_results = "2-1"
        game = "Scorito"
        expected_points = 60
        self.assertEqual(calculate_points(predicted_results, actual_results, game), expected_points)

    def test_no_match_scorito(self):
        predicted_results = "1-0"
        actual_results = "0-2"
        game = "Scorito"
        expected_points = 0
        self.assertEqual(calculate_points(predicted_results, actual_results, game), expected_points)

    def test_incorrect_draw_scorito(self):
        predicted_results = "1-1"
        actual_results = "0-0"
        game = "Scorito"
        expected_points = 60
        self.assertEqual(calculate_points(predicted_results, actual_results, game), expected_points)

if __name__ == '__main__':
    unittest.main()
