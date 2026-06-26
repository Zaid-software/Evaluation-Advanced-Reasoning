ABS_TOLERANCE = 1e-4


def grade_answer(predicted: float, ground_truth: float, tolerance: float = ABS_TOLERANCE) -> bool:
    if predicted is None:
        return False
    return abs(predicted - ground_truth) <= tolerance
