"""FastAPI service -- Phase 8.

`POST /eta` returns every trained quantile *and* the conformal interval. Returning
one number would hide the uncertainty that is the entire point of the project, so
the response shape makes that impossible: a caller who wants a single ETA has to
choose which quantile they are choosing, and see the interval while they do it.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

from eta.logging import get_logger

if TYPE_CHECKING:
    from eta.serving.predictor import Predictor

__all__ = ["EtaRequest", "EtaResponse", "build_app"]

log = get_logger(__name__)


class EtaRequest(BaseModel):
    pickup_zone: int = Field(ge=1, le=265)
    dropoff_zone: int = Field(ge=1, le=265)
    request_time: dt.datetime | None = None


class EtaResponse(BaseModel):
    quantiles: dict[str, float]
    served: float
    served_quantile: float
    interval_low: float
    interval_high: float
    interval_coverage: float
    degraded: bool
    total_ms: float


def build_app(predictor: Predictor, served_quantile: float, target_coverage: float) -> FastAPI:
    app = FastAPI(title="eta-system", version="0.1.0")
    served_key = f"p{round(served_quantile * 100)}"

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "served_quantile": served_quantile,
            "store_failure_rate": round(predictor.store.failure_rate, 6),
        }

    @app.post("/eta", response_model=EtaResponse)
    def eta(req: EtaRequest, response: Response) -> EtaResponse:
        when = req.request_time or dt.datetime.now(dt.UTC)
        pred = predictor.predict(req.pickup_zone, req.dropoff_zone, when)

        response.headers["X-Stage-Timings"] = ", ".join(
            f"{k}={v:.2f}ms" for k, v in pred.timings.items()
        )
        if pred.degraded:
            response.headers["X-Degraded"] = "true"

        return EtaResponse(
            quantiles=pred.quantiles,
            served=pred.quantiles[served_key],
            served_quantile=served_quantile,
            interval_low=pred.interval[0],
            interval_high=pred.interval[1],
            interval_coverage=target_coverage,
            degraded=pred.degraded,
            total_ms=pred.total_ms,
        )

    return app
