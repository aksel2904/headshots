#!/usr/bin/env python3
"""Pull a starter player roster from Wikidata."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("roster")

USER_AGENT = "tennis-wire-headshots/0.1 (https://github.com/tennis-wire/tennis-wire)"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# Every Wikidata id this script depends on. --check resolves them against live
# labels, which is worth doing before trusting a query built on top of them.
ENTITIES = {
    "Q10833314": "tennis player (occupation)",
    "P106": "occupation",
    "P536": "ATP player ID",
    "P597": "WTA player ID",
    "P8618": "ITF player ID",
    "P321": "country for sport",
    "P27": "country of citizenship",
    "P1626": "IOC country code",
    "P569": "date of birth",
    "P18": "image",
    "P166": "award received",
}

# wikibase:sitelinks is materialised on the item, so ranking by it is cheap.
# Counting sitelinks in a subquery instead would time the service out.
QUERY = """
SELECT ?player ?nameEn ?nameRu ?ioc ?born ?image ?atp ?wta ?itf ?sitelinks
WHERE {
  ?player wdt:P106 wd:Q10833314 .
  ?player wikibase:sitelinks ?sitelinks .
  FILTER(?sitelinks >= %(min_sitelinks)d)

  %(tour_filter)s

  ?player rdfs:label ?nameEn . FILTER(lang(?nameEn) = "en")

  OPTIONAL { ?player rdfs:label ?nameRu . FILTER(lang(?nameRu) = "ru") }
  OPTIONAL { ?player wdt:P536 ?atp }
  OPTIONAL { ?player wdt:P597 ?wta }
  OPTIONAL { ?player wdt:P8618 ?itf }
  OPTIONAL { ?player wdt:P569 ?born }
  OPTIONAL { ?player wdt:P18 ?image }

  OPTIONAL { ?player wdt:P321 ?sportCountry }
  OPTIONAL { ?player wdt:P27 ?citizenship }
  BIND(COALESCE(?sportCountry, ?citizenship) AS ?country)
  OPTIONAL { ?country wdt:P1626 ?ioc }
}
ORDER BY DESC(?sitelinks)
LIMIT %(limit)d
"""

TOUR_FILTERS = {
    "atp": "FILTER EXISTS { ?player wdt:P536 [] }",
    "wta": "FILTER EXISTS { ?player wdt:P597 [] }",
    "both": "FILTER(EXISTS { ?player wdt:P536 [] } || EXISTS { ?player wdt:P597 [] })",
}

COLUMNS = (
    "qid", "name_en", "name_ru", "ioc", "born", "commons_file",
    "atp_id", "wta_id", "itf_id", "sitelinks",
)


@dataclass
class Player:
    qid: str
    name_en: str
    name_ru: str
    ioc: str
    born: str
    commons_file: str
    atp_id: str
    wta_id: str
    itf_id: str
    sitelinks: int


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    retry = Retry(
        total=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


HTTP = build_session()


def check_entities() -> int:
    """Resolve every id this script uses and print what it actually is."""
    ids = "|".join(ENTITIES)
    response = HTTP.get(
        WIKIDATA_API,
        params={
            "action": "wbgetentities",
            "ids": ids,
            "props": "labels",
            "languages": "en",
            "format": "json",
            "formatversion": "2",
        },
        timeout=30,
    )
    response.raise_for_status()
    entities = response.json().get("entities", {})

    mismatches = 0
    for entity_id, expected in ENTITIES.items():
        actual = entities.get(entity_id, {}).get("labels", {}).get("en", {}).get("value")
        if actual is None:
            log.error("%-12s MISSING          (expected %s)", entity_id, expected)
            mismatches += 1
            continue
        # Expected strings carry a parenthetical hint; compare on the head only.
        head = expected.split(" (")[0].lower()
        ok = head in actual.lower() or actual.lower() in head
        log.info("%-12s %-26s %s", entity_id, actual, "ok" if ok else f"!= {expected}")
        mismatches += 0 if ok else 1

    if mismatches:
        log.error("%d id(s) do not match their expected meaning", mismatches)
    else:
        log.info("all %d ids resolve as expected", len(ENTITIES))
    return 1 if mismatches else 0


def run_query(tour: str, limit: int, min_sitelinks: int) -> list[Player]:
    query = QUERY % {
        "tour_filter": TOUR_FILTERS[tour],
        "limit": limit,
        "min_sitelinks": min_sitelinks,
    }
    log.info("querying Wikidata (tour=%s, limit=%d, min sitelinks=%d)", tour, limit, min_sitelinks)

    response = HTTP.get(
        SPARQL_ENDPOINT,
        params={"query": query},
        headers={"Accept": "application/sparql-results+json"},
        timeout=180,
    )
    if response.status_code != 200:
        log.error("query service returned %d:\n%s", response.status_code, response.text[:800])
        raise SystemExit(1)

    rows = response.json()["results"]["bindings"]
    log.info("got %d rows", len(rows))

    def value(row: dict, key: str) -> str:
        return row.get(key, {}).get("value", "")

    players: list[Player] = []
    seen: set[str] = set()
    for row in rows:
        qid = value(row, "player").rsplit("/", 1)[-1]
        if qid in seen:  # OPTIONAL clauses can multiply rows
            continue
        seen.add(qid)
        players.append(
            Player(
                qid=qid,
                name_en=value(row, "nameEn"),
                name_ru=value(row, "nameRu"),
                ioc=value(row, "ioc"),
                born=value(row, "born")[:10],
                commons_file=value(row, "image").rsplit("/", 1)[-1].replace("%20", " "),
                atp_id=value(row, "atp"),
                wta_id=value(row, "wta"),
                itf_id=value(row, "itf"),
                sitelinks=int(value(row, "sitelinks") or 0),
            )
        )
    return players


def write_outputs(players: list[Player], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "roster.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for player in players:
            writer.writerow(
                [
                    player.qid, player.name_en, player.name_ru, player.ioc, player.born,
                    player.commons_file, player.atp_id, player.wta_id, player.itf_id,
                    player.sitelinks,
                ]
            )

    # headshots.py reads this directly and skips name resolution when qid is present.
    targets = out_dir / "players.csv"
    with targets.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("qid", "name"))
        for player in players:
            writer.writerow((player.qid, player.name_en))

    with_image = sum(1 for p in players if p.commons_file)
    with_ru = sum(1 for p in players if p.name_ru)
    with_ioc = sum(1 for p in players if p.ioc)
    total = len(players) or 1

    log.info("%s", "-" * 52)
    log.info("players       %d", len(players))
    log.info("with P18      %d (%.0f%%)", with_image, with_image / total * 100)
    log.info("with ru label %d (%.0f%%)", with_ru, with_ru / total * 100)
    log.info("with IOC code %d (%.0f%%)", with_ioc, with_ioc / total * 100)
    log.info("roster:  %s", csv_path)
    log.info("targets: %s", targets)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="roster",
        description="Pull a starter player roster from Wikidata, ranked by Wikipedia reach.",
    )
    parser.add_argument("-o", "--out", type=Path, default=Path("out"),
                        help="output directory (default: out)")
    parser.add_argument("-t", "--tour", choices=("atp", "wta", "both"), default="both",
                        help="require an ATP and/or WTA id (default: both)")
    parser.add_argument("-n", "--limit", type=int, default=400,
                        help="how many players to keep (default: 400)")
    parser.add_argument("--min-sitelinks", type=int, default=5,
                        help="drop players below this many Wikipedia versions (default: 5)")
    parser.add_argument("--check", action="store_true",
                        help="resolve every Wikidata id used here and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    if args.check:
        return check_entities()

    players = run_query(args.tour, args.limit, args.min_sitelinks)
    if not players:
        log.error("no players returned")
        return 1
    write_outputs(players, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())