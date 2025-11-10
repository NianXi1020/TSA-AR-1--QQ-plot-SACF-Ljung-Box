import math
import random
import statistics
import struct
import zlib
from typing import List, Tuple


Color = Tuple[int, int, int]


class SimplePNG:
    def __init__(self, width: int, height: int, background: Color = (255, 255, 255)) -> None:
        self.width = width
        self.height = height
        self.pixels: List[Color] = [background for _ in range(width * height)]

    def set_pixel(self, x: int, y: int, color: Color) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[y * self.width + x] = color

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def draw_rectangle(self, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                self.set_pixel(x, y, color)

    def save(self, path: str) -> None:
        raw = bytearray()
        for y in range(self.height):
            raw.append(0)
            for x in range(self.width):
                r, g, b = self.pixels[y * self.width + x]
                raw.extend((r, g, b))
        compressed = zlib.compress(bytes(raw), level=9)

        def chunk(tag: bytes, data: bytes) -> bytes:
            length = struct.pack("!I", len(data))
            crc = struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            return length + tag + data + crc

        header = chunk(
            b"IHDR",
            struct.pack("!IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0),
        )
        data = chunk(b"IDAT", compressed)
        end = chunk(b"IEND", b"")
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
            f.write(header)
            f.write(data)
            f.write(end)


def simulate_ar1(n: int, phi: float, sigma: float, seed: int) -> List[float]:
    random.seed(seed)
    series = [0.0] * n
    for i in range(n):
        noise = random.gauss(0.0, sigma)
        if i == 0:
            series[i] = noise
        else:
            series[i] = phi * series[i - 1] + noise
    return series


def fit_ar1(series: List[float]) -> Tuple[float, float]:
    if len(series) < 2:
        return 0.0, 0.0
    n = len(series)
    sum_y_prev = 0.0
    sum_y_prev_sq = 0.0
    sum_y_curr = 0.0
    sum_y_curr_prev = 0.0
    for t in range(1, n):
        y_prev = series[t - 1]
        y_curr = series[t]
        sum_y_prev += y_prev
        sum_y_prev_sq += y_prev * y_prev
        sum_y_curr += y_curr
        sum_y_curr_prev += y_curr * y_prev
    m = n - 1
    det = m * sum_y_prev_sq - sum_y_prev * sum_y_prev
    if abs(det) < 1e-12:
        return 0.0, 0.0
    intercept = (sum_y_prev_sq * sum_y_curr - sum_y_prev * sum_y_curr_prev) / det
    phi = (m * sum_y_curr_prev - sum_y_prev * sum_y_curr) / det
    return intercept, phi


def compute_residuals(series: List[float], intercept: float, phi: float) -> List[float]:
    residuals = []
    if abs(1.0 - phi) < 1e-12:
        steady_state = intercept
    else:
        steady_state = intercept / (1.0 - phi)
    if series:
        residuals.append(series[0] - steady_state)
    for t in range(1, len(series)):
        prediction = intercept + phi * series[t - 1]
        residuals.append(series[t] - prediction)
    return residuals


def standardize(values: List[float]) -> Tuple[List[float], float]:
    if not values:
        return [], 0.0
    mean_val = sum(values) / len(values)
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    std = math.sqrt(variance) if variance > 0 else 1.0
    standardized = [(v - mean_val) / std for v in values]
    return standardized, std


def normal_quantiles(count: int) -> List[float]:
    dist = statistics.NormalDist()
    return [dist.inv_cdf((i + 0.5) / count) for i in range(count)]


def sample_acf(values: List[float], lag_max: int) -> List[float]:
    n = len(values)
    mean_val = sum(values) / n if n else 0.0
    denom = sum((v - mean_val) ** 2 for v in values)
    if denom == 0:
        return [0.0] * (lag_max + 1)
    acf_values = []
    for lag in range(lag_max + 1):
        num = 0.0
        for t in range(lag, n):
            num += (values[t] - mean_val) * (values[t - lag] - mean_val)
        acf_values.append(num / denom)
    return acf_values


def _gamma_series(a: float, x: float) -> float:
    term = 1.0 / a
    total = term
    n = 1
    while True:
        term *= x / (a + n)
        total += term
        if abs(term) < abs(total) * 1e-12:
            break
        n += 1
    return total * math.exp(-x + a * math.log(x))


def regularized_gamma_p(a: float, x: float) -> float:
    if x <= 0:
        return 0.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        return _gamma_series(a, x) * math.exp(-gln)
    b = x + 1.0 - a
    c = 1.0 / 1e-30
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - gln) * h


def ljung_box(residuals: List[float], lag: int) -> Tuple[float, float]:
    n = len(residuals)
    acf_vals = sample_acf(residuals, lag)
    total = 0.0
    for k in range(1, lag + 1):
        total += acf_vals[k] ** 2 / (n - k)
    q_stat = n * (n + 2) * total
    degrees = max(lag - 1, 1)
    p_value = 1.0 - regularized_gamma_p(0.5 * degrees, 0.5 * q_stat)
    return q_stat, p_value


def _map_point(x: float, x_min: float, x_max: float, a: int, b: int) -> int:
    if x_max == x_min:
        return a
    ratio = (x - x_min) / (x_max - x_min)
    return int(round(a + ratio * (b - a)))


def plot_residuals(residuals: List[float], path: str) -> None:
    width, height, margin = 800, 500, 60
    img = SimplePNG(width, height)
    x0, x1 = margin, width - margin
    y0, y1 = height - margin, margin
    img.draw_line(x0, y0, x1, y0, (0, 0, 0))
    img.draw_line(x0, y0, x0, y1, (0, 0, 0))
    n = len(residuals)
    if n == 0:
        img.save(path)
        return
    y_min = min(residuals)
    y_max = max(residuals)
    span = y_max - y_min
    padding = 0.1 * span if span > 0 else 0.5
    y_min -= padding
    y_max += padding
    points = []
    for idx, value in enumerate(residuals):
        x = _map_point(idx, 0, max(n - 1, 1), x0, x1)
        y = _map_point(value, y_min, y_max, y0, y1)
        points.append((x, y))
    if y_min < 0 < y_max:
        zero_y = _map_point(0.0, y_min, y_max, y0, y1)
        img.draw_line(x0, zero_y, x1, zero_y, (200, 0, 0))
    for i in range(len(points) - 1):
        img.draw_line(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], (0, 0, 255))
    for x, y in points:
        img.draw_rectangle(x - 2, y - 2, x + 2, y + 2, (0, 0, 0))
    for idx in range(n):
        x = _map_point(idx, 0, max(n - 1, 1), x0, x1)
        img.draw_line(x, y0, x, y0 + 5, (0, 0, 0))
    for i in range(5):
        value = y_min + i * (y_max - y_min) / 4
        y = _map_point(value, y_min, y_max, y0, y1)
        img.draw_line(x0 - 5, y, x0, y, (0, 0, 0))
    img.save(path)


def plot_qq(theoretical: List[float], empirical: List[float], path: str) -> None:
    width, height, margin = 800, 500, 60
    img = SimplePNG(width, height)
    x0, x1 = margin, width - margin
    y0, y1 = height - margin, margin
    img.draw_line(x0, y0, x1, y0, (0, 0, 0))
    img.draw_line(x0, y0, x0, y1, (0, 0, 0))
    if not theoretical:
        img.save(path)
        return
    x_min = min(theoretical)
    x_max = max(theoretical)
    y_min = min(empirical)
    y_max = max(empirical)
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_padding = 0.1 * x_span if x_span > 0 else 0.5
    y_padding = 0.1 * y_span if y_span > 0 else 0.5
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding
    diag_points = []
    diag_min = min(x_min, y_min)
    diag_max = max(x_max, y_max)
    diag_min_pt = (
        _map_point(diag_min, x_min, x_max, x0, x1),
        _map_point(diag_min, y_min, y_max, y0, y1),
    )
    diag_max_pt = (
        _map_point(diag_max, x_min, x_max, x0, x1),
        _map_point(diag_max, y_min, y_max, y0, y1),
    )
    img.draw_line(diag_min_pt[0], diag_min_pt[1], diag_max_pt[0], diag_max_pt[1], (200, 0, 0))
    for t, e in zip(theoretical, empirical):
        x = _map_point(t, x_min, x_max, x0, x1)
        y = _map_point(e, y_min, y_max, y0, y1)
        img.draw_rectangle(x - 2, y - 2, x + 2, y + 2, (0, 0, 200))
    for i in range(5):
        x_val = x_min + i * (x_max - x_min) / 4
        x = _map_point(x_val, x_min, x_max, x0, x1)
        img.draw_line(x, y0, x, y0 + 5, (0, 0, 0))
        y_val = y_min + i * (y_max - y_min) / 4
        y = _map_point(y_val, y_min, y_max, y0, y1)
        img.draw_line(x0 - 5, y, x0, y, (0, 0, 0))
    img.save(path)


def plot_acf(acf_vals: List[float], path: str, conf_level: float) -> None:
    width, height, margin = 800, 500, 60
    img = SimplePNG(width, height)
    x0, x1 = margin, width - margin
    y0, y1 = height - margin, margin
    img.draw_line(x0, y0, x1, y0, (0, 0, 0))
    img.draw_line(x0, y0, x0, y1, (0, 0, 0))
    if len(acf_vals) <= 1:
        img.save(path)
        return
    values = acf_vals[1:]
    lag_max = len(values)
    limit = max(max(values), conf_level)
    limit = max(limit, 0.1)
    y_max = limit * 1.2
    y_min = -y_max
    for i in range(5):
        val = y_min + i * (y_max - y_min) / 4
        y = _map_point(val, y_min, y_max, y0, y1)
        img.draw_line(x0 - 5, y, x0, y, (0, 0, 0))
    conf_y = _map_point(conf_level, y_min, y_max, y0, y1)
    neg_conf_y = _map_point(-conf_level, y_min, y_max, y0, y1)
    img.draw_line(x0, conf_y, x1, conf_y, (200, 0, 0))
    img.draw_line(x0, neg_conf_y, x1, neg_conf_y, (200, 0, 0))
    for idx, value in enumerate(values, start=1):
        x_center = _map_point(idx, 0, lag_max + 1, x0, x1)
        x_left = _map_point(idx - 0.3, 0, lag_max + 1, x0, x1)
        x_right = _map_point(idx + 0.3, 0, lag_max + 1, x0, x1)
        y = _map_point(value, y_min, y_max, y0, y1)
        img.draw_line(x_center, y0, x_center, y, (0, 0, 255))
        if y < y0:
            img.draw_rectangle(x_left, y, x_right, y0, (0, 0, 255))
        else:
            img.draw_rectangle(x_left, y0, x_right, y, (0, 0, 255))
        img.draw_line(x_center, y0, x_center, y0 + 5, (0, 0, 0))
    img.save(path)
