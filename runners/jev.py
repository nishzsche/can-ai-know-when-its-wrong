"""TypeSafe Jev through OpenRouter's System One endpoint (POST /api/v1/systemone).

    uv run python runners/jev.py --smoke
    uv run python runners/jev.py --split test

Jev reads the identical System One request every decision model gets and answers the `noul`
question with P(yes) directly.

Versioning (PLAN.md: "use jev-latest; record the resolved version"): the runner asks once for
`jev-latest`, records the exact version that answered (e.g. `typesafe/jev-1.13-20260917`), and
then requests that version for every customer. If a response ever comes from a different
version, it stops: one run must not mix two Jevs. Needs OPENROUTER_API_KEY.
"""

from __future__ import annotations

import os
import re
import time

from harness import Prediction, build_parser, http_client, resolve_args, run

BASE_URL = "https://openrouter.ai/api/v1"
REQUESTED = "jev-latest"
QUESTION_ID = "will_default"
PROBE = {
    "state": "This is a connectivity check.",
    "questions": {"probe": {"type": "noul", "instructions": "Is this a connectivity check?"}},
}
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class HTTPStatusError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status


class JevPredictor:
    contestant = "jev"

    def __init__(self, base_url: str = BASE_URL, api_key: str | None = None, tries: int = 5):
        self.base_url = base_url.rstrip("/")
        self.key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.key:
            raise SystemExit("Set OPENROUTER_API_KEY (environment or the git-ignored .env).")
        self.tries = tries
        self.http = http_client(timeout=120)  # kept alive, IPv4 (see harness.http_client)
        # Resolve jev-latest once, then pin the exact version it resolved to.
        first = self._post({"model": REQUESTED, **PROBE})
        self.resolved = first["model"]
        self.request_model = self._pin(self.resolved)
        self.model_version = self.resolved
        self.meta = {"requested": REQUESTED, "resolved": self.resolved,
                     "request_model": self.request_model, "provider": first.get("provider"),
                     "base_url": self.base_url}  # fmt: skip

    def _pin(self, resolved: str) -> str:
        """The model id to send so that every request hits `resolved`."""
        family = re.sub(r"-\d{8}$", "", resolved)  # typesafe/jev-1.13-20260917 -> typesafe/jev-1.13
        for candidate in dict.fromkeys([resolved, family]):
            try:
                if self._post({"model": candidate, **PROBE})["model"] == resolved:
                    return candidate
            except HTTPStatusError as exc:
                if exc.status not in (400, 404):
                    raise
        raise SystemExit(
            f"Can't pin {resolved}: neither it nor {family} is accepted as a model id."
        )

    def _post(self, body: dict) -> dict:
        import httpx2

        headers = {"authorization": f"Bearer {self.key}"}
        for attempt in range(self.tries):
            try:
                resp = self.http.post(f"{self.base_url}/systemone", json=body, headers=headers)
            except httpx2.TransportError:
                if attempt == self.tries - 1:
                    raise
            else:
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code not in RETRY_STATUS or attempt == self.tries - 1:
                    raise HTTPStatusError(resp.status_code, resp.text)
            time.sleep(min(2**attempt, 30))
        raise AssertionError("unreachable")

    def predict(self, request: dict) -> Prediction:
        start = time.perf_counter()
        resp = self._post({"model": self.request_model, **request})
        latency_ms = (time.perf_counter() - start) * 1000
        if resp.get("model") != self.resolved:
            raise SystemExit(
                f"Jev changed version mid-run: expected {self.resolved}, got {resp.get('model')}. "
                f"Stop and decide how to handle it; don't mix versions in one file."
            )
        answer = resp["answers"][QUESTION_ID]
        usage = resp.get("usage") or {}
        return Prediction(
            p_default=float(answer["noul"]),
            latency_ms=latency_ms,
            cost_usd=float(usage.get("cost") or 0.0),
            raw={
                "id": resp.get("id"),
                "model": resp.get("model"),
                "provider": resp.get("provider"),
                "answer": answer,
                "usage": usage,
            },  # fmt: skip
        )


def main() -> None:
    args = resolve_args(build_parser(__doc__).parse_args())
    from env import load_env

    load_env()
    run(JevPredictor(), args)


if __name__ == "__main__":
    main()
