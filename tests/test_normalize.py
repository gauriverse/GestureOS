import numpy as np
from gestureos.features.normalize import normalize_hand, FEATURE_DIM


def make_hand(seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.2, 0.8, size=(21, 3)).astype(np.float32)
    return pts


def test_output_shape():
    out = normalize_hand(make_hand())
    assert out.shape == (FEATURE_DIM,), out.shape


def test_wrist_is_origin():
    out = normalize_hand(make_hand()).reshape(21, 3)
    assert np.allclose(out[0], 0.0)


def test_middle_knuckle_has_unit_length():
    out = normalize_hand(make_hand()).reshape(21, 3)
    assert np.isclose(np.linalg.norm(out[9, :2]), 1.0)


def test_scale_invariant():
    hand = make_hand()
    a = normalize_hand(hand)
    b = normalize_hand(hand * 0.5 + 0.1)  
    assert np.allclose(a.reshape(21, 3)[:, :2], b.reshape(21, 3)[:, :2], atol=1e-5)


def test_degenerate_hand_returns_zeros():
    hand = np.zeros((21, 3), dtype=np.float32)
    out = normalize_hand(hand)
    assert np.all(out == 0.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("All normalization tests passed.")