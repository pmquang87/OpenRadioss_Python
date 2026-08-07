import sys

def fix_test_m41_qeph():
    with open("tests/test_m41_qeph.py", "r") as f:
        content = f.read()

    helper = """def _condensed_length(xl: np.ndarray, area: np.ndarray, facdt: float) -> np.ndarray:
    x, y = xl[:, :, 0], xl[:, :, 1]
    x13 = 0.5 * (x[:, 0] - x[:, 2])
    y13 = 0.5 * (y[:, 0] - y[:, 2])
    x24 = 0.5 * (x[:, 1] - x[:, 3])
    y24 = 0.5 * (y[:, 1] - y[:, 3])
    ll = np.maximum(x13 ** 2 + y13 ** 2, x24 ** 2 + y24 ** 2)
    rx = x[:, 1] + x[:, 2] - x[:, 3] - x[:, 0]
    ry = y[:, 1] + y[:, 2] - y[:, 3] - y[:, 0]
    sx = -x[:, 1] + x[:, 2] + x[:, 3] - x[:, 0]
    sy = -y[:, 1] + y[:, 2] + y[:, 3] - y[:, 0]
    c1 = np.sqrt(rx ** 2 + ry ** 2)
    c2 = np.sqrt(sx ** 2 + sy ** 2)
    cmax = np.maximum(c1, c2)
    cmin = np.maximum(np.minimum(c1, c2), 1e-20)
    fac1 = np.minimum(0.5, 0.25 * (cmax / cmin - 1.0)) + 1.0
    fac2 = 4.0 * area / np.maximum(c1 * c2, 1e-20)
    fac2 = 3.413 * np.maximum(0.0, fac2 - 0.7071)
    fac2 = 0.78 + 0.22 * fac2 ** 3
    faci = 2.0 * fac1 * fac2
    lm = np.maximum(np.abs(x[:, 1] * y[:, 3] - y[:, 1] * x[:, 3]),
                    np.abs(x[:, 0] * y[:, 2] - y[:, 0] * x[:, 2]))
    s = np.sqrt(faci * (facdt + lm / np.maximum(area, 1e-20)) * ll)
    return area / np.maximum(s, 1e-20)

"""

    content = content.replace("def test_dt_claim_condensed_length(tmp_path):", helper + "def test_dt_claim_condensed_length(tmp_path):")
    content = content.replace("ll = shell_bt4._condensed_length(xl, G[\"area\"], 1.25)", "ll = _condensed_length(xl, G[\"area\"], 1.25)")

    with open("tests/test_m41_qeph.py", "w") as f:
        f.write(content)

fix_test_m41_qeph()
