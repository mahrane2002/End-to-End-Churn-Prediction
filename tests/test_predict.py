import numpy as np

from src.models.predict import predict, predict_proba


class FakeModel:

    def predict(self, X):
        return np.array([0, 1])

    def predict_proba(self, X):
        return np.array([
            [0.8, 0.2],
            [0.3, 0.7],
        ])


def test_predict():
    model = FakeModel()

    X = [[1], [2]]

    result = predict(model, X)

    assert len(result) == 2
    assert list(result) == [0, 1]


def test_predict_proba():
    model = FakeModel()

    X = [[1], [2]]

    result = predict_proba(model, X)

    assert len(result) == 2
    assert np.all(result >= 0)
    assert np.all(result <= 1)