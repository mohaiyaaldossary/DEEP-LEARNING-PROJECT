"""Pure event logic, independent of the detector and GUI."""
from dataclasses import dataclass
from math import isfinite


def validate_polygon(points):
    if len(points) < 3:
        return 'fewer than 3 distinct points'
    if any(len(p) != 2 or not all(isfinite(float(v)) for v in p) for p in points):
        return 'coordinates must be finite pairs'
    points = [tuple(map(float, p)) for p in points]
    if len(set(points)) != len(points):
        return 'duplicate vertices'
    def orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def on(a, b, c):
        return abs(orient(a,b,c)) <= 1e-9 and min(a[0],b[0]) <= c[0] <= max(a[0],b[0]) and min(a[1],b[1]) <= c[1] <= max(a[1],b[1])
    def intersects(a,b,c,d):
        o = [orient(a,b,c),orient(a,b,d),orient(c,d,a),orient(c,d,b)]
        return (o[0]*o[1] < 0 and o[2]*o[3] < 0) or on(a,b,c) or on(a,b,d) or on(c,d,a) or on(c,d,b)
    n = len(points)
    for i in range(n):
        for j in range(i+1,n):
            if j == i+1 or (i == 0 and j == n-1):
                continue
            if intersects(points[i],points[(i+1)%n],points[j],points[(j+1)%n]):
                return 'crossing or touching nonadjacent edges'
    area2 = sum(points[i][0]*points[(i+1)%n][1]-points[(i+1)%n][0]*points[i][1] for i in range(n))
    if abs(area2) <= 1e-9:
        return 'zero area'
    return None


@dataclass
class TrackState:
    inside: bool
    last_seen: int
    out_count: int = 0


class ZoneEvents:
    """Rearm only on consecutive observed outside frames; absence is unknown.

    Missing tracks retain event state for ttl frames but never keep a visible
    warning active. IDs are run-local, not persistent person identities.
    """
    def __init__(self, exit_confirm=15, ttl=90):
        if exit_confirm < 1 or ttl < 1:
            raise ValueError('exit_confirm and ttl must be positive')
        self.exit_confirm, self.ttl = exit_confirm, ttl
        self.states = {}

    def update(self, frame, observations):
        events = []
        for tid in list(self.states):
            if frame-self.states[tid].last_seen > self.ttl:
                del self.states[tid]
        for tid, inside in observations.items():
            s = self.states.get(tid)
            if s is None:
                self.states[tid] = TrackState(bool(inside),frame)
                if inside:
                    events.append((tid,'initial_presence'))
                continue
            if frame != s.last_seen+1:
                s.out_count = 0
            s.last_seen = frame
            if inside:
                if not s.inside:
                    events.append((tid,'entry'))
                    s.inside = True
                s.out_count = 0
            else:
                s.out_count += 1
                if s.out_count >= self.exit_confirm:
                    s.inside = False
        return events, any(observations.values())
