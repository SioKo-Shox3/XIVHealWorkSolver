"""FFLogs API v2 (GraphQL) client with OAuth2 client credentials and local caching."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

TOKEN_URL = "https://www.fflogs.com/oauth/token"
GRAPHQL_URL = "https://www.fflogs.com/api/v2/client"


class FFLogsConfigError(RuntimeError):
    """Raised when FFLogs API credentials are missing or invalid."""


class FFLogsClient:
    """Thin, cache-first GraphQL client for FFLogs API v2."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        cache_dir: str | Path = "data/cache/fflogs",
        min_request_interval_s: float = 0.25,
    ) -> None:
        load_dotenv()
        self.client_id = client_id or os.getenv("FFLOGS_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("FFLOGS_CLIENT_SECRET")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_request_interval_s = min_request_interval_s
        self._access_token: str | None = None
        self._token_expires_at: float = 0.0
        self._last_request_at: float = 0.0
        self._session = requests.Session()

    def _ensure_token(self) -> str:
        if self._access_token and time.time() < self._token_expires_at - 60:
            return self._access_token
        if not self.client_id or not self.client_secret:
            raise FFLogsConfigError(
                "FFLOGS_CLIENT_ID and FFLOGS_CLIENT_SECRET are not configured. "
                "Create a client at https://www.fflogs.com/api/clients/ and put them in .env."
            )
        resp = self._session.post(
            TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(self.client_id, self.client_secret),
            timeout=30,
        )
        if resp.status_code != 200:
            raise FFLogsConfigError(f"FFLogs OAuth2 failed ({resp.status_code}): {resp.text}")
        payload = resp.json()
        self._access_token = payload["access_token"]
        self._token_expires_at = time.time() + float(payload.get("expires_in", 3600))
        return self._access_token

    def _cache_path(self, query: str, variables: dict[str, Any]) -> Path:
        blob = json.dumps({"q": query.strip(), "v": variables}, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(blob).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def query(
        self,
        graphql: str,
        variables: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        variables = variables or {}
        cpath = self._cache_path(graphql, variables)
        if use_cache and cpath.exists():
            return json.loads(cpath.read_text(encoding="utf-8"))

        token = self._ensure_token()
        elapsed = time.time() - self._last_request_at
        if elapsed < self.min_request_interval_s:
            time.sleep(self.min_request_interval_s - elapsed)

        for attempt in range(4):
            self._last_request_at = time.time()
            resp = self._session.post(
                GRAPHQL_URL,
                json={"query": graphql, "variables": variables},
                headers={"Authorization": f"Bearer {token}"},
                timeout=60,
            )
            if resp.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            if "errors" in data:
                raise RuntimeError(f"FFLogs GraphQL error: {data['errors']}")
            result = data["data"]
            if use_cache:
                cpath.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            return result

        raise RuntimeError("FFLogs API rate limit exceeded after retries (HTTP 429).")

    def get_rate_limit(self) -> dict[str, Any]:
        q = """
        query {
          rateLimitData {
            limitPerHour
            pointsSpentThisHour
            pointsResetIn
          }
        }
        """
        return self.query(q, use_cache=False)["rateLimitData"]

    def get_report_overview(self, report_code: str) -> dict[str, Any]:
        q = """
        query ($code: String!) {
          reportData {
            report(code: $code) {
              code
              title
              startTime
              zone { id name }
              fights {
                id
                name
                encounterID
                startTime
                endTime
                kill
                fightPercentage
                friendlyPlayers
              }
              masterData {
                actors {
                  id
                  name
                  type
                  subType
                }
                abilities {
                  gameID
                  name
                  type
                }
              }
            }
          }
        }
        """
        return self.query(q, {"code": report_code})["reportData"]["report"]

    def get_fight_events(
        self,
        report_code: str,
        fight_id: int,
        start_time: float,
        end_time: float,
        data_type: str = "DamageTaken",
        hostility_type: str = "Friendlies",
        max_pages: int = 40,
    ) -> list[dict[str, Any]]:
        """Fetch paginated event stream for one fight."""
        q = """
        query (
          $code: String!,
          $fightIDs: [Int],
          $startTime: Float,
          $endTime: Float,
          $dataType: EventDataType,
          $hostilityType: HostilityType
        ) {
          reportData {
            report(code: $code) {
              events(
                fightIDs: $fightIDs
                startTime: $startTime
                endTime: $endTime
                dataType: $dataType
                hostilityType: $hostilityType
                limit: 10000
              ) {
                data
                nextPageTimestamp
              }
            }
          }
        }
        """
        all_events: list[dict[str, Any]] = []
        cursor = start_time
        for _ in range(max_pages):
            res = self.query(
                q,
                {
                    "code": report_code,
                    "fightIDs": [fight_id],
                    "startTime": cursor,
                    "endTime": end_time,
                    "dataType": data_type,
                    "hostilityType": hostility_type,
                },
            )
            ev_page = res["reportData"]["report"]["events"]
            batch = ev_page.get("data") or []
            all_events.extend(batch)
            nxt = ev_page.get("nextPageTimestamp")
            if nxt is None or nxt >= end_time:
                break
            cursor = float(nxt)
        return all_events

    def get_encounter_ranking_reports(
        self,
        encounter_id: int,
        page: int = 1,
        metric: str = "rdps",
        difficulty: int = 101,
        spec_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch ranked clear reports for an encounter (difficulty=101 for Savage)."""
        q = """
        query (
          $encounterID: Int!,
          $page: Int!,
          $metric: CharacterRankingMetricType,
          $difficulty: Int,
          $specName: String
        ) {
          worldData {
            encounter(id: $encounterID) {
              id
              name
              characterRankings(
                page: $page
                metric: $metric
                difficulty: $difficulty
                specName: $specName
              )
            }
          }
        }
        """
        enc = self.query(
            q,
            {
                "encounterID": encounter_id,
                "page": page,
                "metric": metric,
                "difficulty": difficulty,
                "specName": spec_name,
            },
        )["worldData"]["encounter"]
        rankings = (enc.get("characterRankings") or {}).get("rankings") or []
        out: list[dict[str, Any]] = []
        for r in rankings:
            rep = r.get("report") or {}
            if rep.get("code") and rep.get("fightID"):
                out.append(
                    {
                        "report_code": rep["code"],
                        "fight_id": int(rep["fightID"]),
                        "character_name": r.get("name"),
                        "spec": r.get("spec"),
                        "duration_ms": r.get("duration"),
                        "amount": r.get("amount"),
                    }
                )
        return out

    def get_fight_table(
        self,
        report_code: str,
        fight_id: int,
        data_type: str = "Summary",
    ) -> dict[str, Any]:
        """Fetch FFLogs summary/healing/damage/deaths table for quality curation."""
        q = """
        query ($code: String!, $fightIDs: [Int], $dataType: TableDataType) {
          reportData {
            report(code: $code) {
              table(fightIDs: $fightIDs, dataType: $dataType)
            }
          }
        }
        """
        res = self.query(q, {"code": report_code, "fightIDs": [fight_id], "dataType": data_type})
        return (res["reportData"]["report"].get("table") or {}).get("data") or {}
