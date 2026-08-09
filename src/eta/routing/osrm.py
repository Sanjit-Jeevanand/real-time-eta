from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import httpx

from eta.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "TURN_MANEUVERS",
    "OsrmClient",
    "RouteError",
    "RouteLeg",
    "parse_route",
]

log = get_logger(__name__)


class RouteError(Exception):
    pass


TURN_MANEUVERS: Final = frozenset(
    {
        "turn",
        "fork",
        "merge",
        "on ramp",
        "off ramp",
        "end of road",
        "roundabout",
        "rotary",
        "roundabout turn",
    }
)

MOTORWAY_CLASS: Final = "motorway"


@dataclass(frozen=True, slots=True)
class RouteLeg:
    distance_m: float
    duration_s: float
    turn_count: int
    highway_fraction: float


def _is_turn(step: dict[str, Any]) -> bool:
    maneuver: dict[str, Any] = step.get("maneuver", {})
    kind = str(maneuver.get("type", ""))
    if kind in TURN_MANEUVERS:
        return str(maneuver.get("modifier", "")) != "straight"
    return False


def _is_motorway(step: dict[str, Any]) -> bool:
    for intersection in step.get("intersections", []):
        classes = intersection.get("classes")
        if classes and MOTORWAY_CLASS in classes:
            return True
    return False


def parse_route(payload: dict[str, Any]) -> RouteLeg | None:
    routes = payload.get("routes") or []
    if not routes:
        return None
    route = routes[0]
    distance = float(route["distance"])
    duration = float(route["duration"])

    turns = 0
    motorway_m = 0.0
    for leg in route.get("legs", []):
        for step in leg.get("steps", []):
            if _is_turn(step):
                turns += 1
            if _is_motorway(step):
                motorway_m += float(step.get("distance", 0.0))

    fraction = motorway_m / distance if distance > 0 else 0.0
    return RouteLeg(distance, duration, turns, min(fraction, 1.0))


class OsrmClient:
    def __init__(
        self,
        base_url: str,
        timeout_s: float = 10.0,
        concurrency: int = 12,
        max_retries: int = 5,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=0)
        self._timeout = timeout_s
        self._max_retries = max_retries
        self._semaphore = asyncio.Semaphore(concurrency)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> OsrmClient:
        self._client = httpx.AsyncClient(
            base_url=self._base_url, timeout=self._timeout, limits=self._limits
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def health(self) -> bool:
        assert self._client is not None
        try:
            r = await self._client.get("/route/v1/driving/-73.985,40.758;-73.968,40.785")
        except httpx.HTTPError:
            return False
        status: int = r.status_code
        return status == 200

    async def route(
        self, origin: tuple[float, float], destination: tuple[float, float]
    ) -> RouteLeg | None:
        assert self._client is not None
        path = (
            f"/route/v1/driving/{origin[1]:.6f},{origin[0]:.6f}"
            f";{destination[1]:.6f},{destination[0]:.6f}"
        )
        params = {"steps": "true", "overview": "false", "annotations": "false"}
        last: Exception | None = None
        async with self._semaphore:
            for attempt in range(self._max_retries):
                try:
                    r = await self._client.get(path, params=params)
                except httpx.HTTPError as exc:
                    last = exc
                    await asyncio.sleep(0.05 * 2**attempt)
                    continue
                if r.status_code != 200:
                    return None
                return parse_route(r.json())
        msg = f"route {origin} -> {destination} failed after {self._max_retries} attempts: {last}"
        raise RouteError(msg)

    async def route_many(
        self, pairs: Sequence[tuple[tuple[float, float], tuple[float, float]]]
    ) -> list[RouteLeg | None | RouteError]:
        tasks = [self.route(o, d) for o, d in pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: list[RouteLeg | None | RouteError] = []
        for r in results:
            if isinstance(r, RouteError):
                out.append(r)
            elif isinstance(r, BaseException):
                raise r
            else:
                out.append(r)
        return out
