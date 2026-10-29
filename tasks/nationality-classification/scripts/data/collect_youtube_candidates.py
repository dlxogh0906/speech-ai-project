"""Find Korean speech candidates without downloading audio or assigning labels.

Search-country hints are never treated as verified speaker nationalities.
The review CSV needs identity, country, Korean speech spans, and source-use
evidence before a candidate can become training data. Previous evaluation
videos are marked as reserved, including when a new search rediscovers them.

Run with an environment containing yt_dlp (data utilities may use Python 3.11).
"""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r:
        raise RuntimeError("Repository root not found")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse
import concurrent.futures
import csv
import datetime
import json
import urllib.parse

from shared.paths import NAT_OUTPUTS, NAT_TASK


def dump_json(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def search(job, limit):
    import yt_dlp
    query = job["query"]
    if job["filter"] == "creative_commons":
        url = ("https://www.youtube.com/results?search_query=" +
               urllib.parse.quote(query) + "&sp=EgIwAQ%3D%3D")
    else:
        url = "ytsearch{}:{}".format(limit, query)
    opts = {"quiet": True, "no_warnings": True, "skip_download": True,
            "extract_flat": "in_playlist", "playlistend": limit,
            "socket_timeout": 12, "retries": 1, "extractor_retries": 1}
    with yt_dlp.YoutubeDL(opts) as downloader:
        result = downloader.extract_info(url, download=False)
    entries = []
    for entry in (result or {}).get("entries") or []:
        if not entry or not entry.get("id"):
            continue
        entries.append({key: entry.get(key) for key in
                        ("id", "title", "channel", "channel_id", "uploader", "duration", "view_count")})
    return dict(job, entries=entries)


def details(candidate):
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "skip_download": True,
            "socket_timeout": 12, "retries": 1, "extractor_retries": 1,
            "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as downloader:
        info = downloader.extract_info(candidate["url"], download=False)
    return {key: info.get(key) for key in
            ("id", "title", "channel", "channel_id", "duration", "description", "license",
             "upload_date", "language", "availability")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=_pl.Path, required=True,
                        help="Per-country NPU validation comparison CSV")
    parser.add_argument("--holdout", type=_pl.Path, required=True,
                        help="Previously evaluated video candidates JSON")
    parser.add_argument("--out", type=_pl.Path, default=NAT_OUTPUTS / "youtube_pilot/catalog")
    parser.add_argument("--per-query", type=int, default=5)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--countries", default="", help="Optional comma-separated head label names")
    parser.add_argument("--metadata-limit", type=int, default=20,
                        help="Inspect this many candidate pages; no media downloads")
    parser.add_argument("--inspect-ids", default="",
                        help="Inspect specific comma-separated candidate IDs instead of the default page selection")
    parser.add_argument("--queries-json", type=_pl.Path,
                        help="Optional list of country_hint/query/filter search jobs")
    args = parser.parse_args()
    if args.per_query < 1 or args.workers < 1 or args.metadata_limit < 0:
        parser.error("Invalid collection limits")
    args.out.mkdir(parents=True, exist_ok=True)
    mapping = json.loads((NAT_TASK / "configs/country41_ko2en.json").read_text(encoding="utf-8"))
    names = {en: ko for ko, en in mapping.items()}
    selected = set(filter(None, (s.strip() for s in args.countries.split(","))))
    if selected - set(names):
        parser.error("Unknown head labels: " + ", ".join(sorted(selected - set(names))))
    with args.baseline.open(encoding="utf-8-sig", newline="") as stream:
        baseline = {r["country"]: r for r in csv.DictReader(stream)}
    holdout_rows = json.loads(args.holdout.read_text(encoding="utf-8"))
    reserved = {r["id"] for r in holdout_rows}
    priorities = []
    for country, ko in names.items():
        if country == "Korea" or (selected and country not in selected):
            continue
        row = baseline[country]
        n_spk = int(row["train_speakers"])
        n_valid = int(row["three_validation_speakers"])
        acc = float(row["three_top1"]) if row["three_top1"] else None
        if acc is None:
            reason, order = "no_three_sentence_validation", 0
        elif acc < .65:
            reason, order = "low_validation_accuracy", 1
        elif n_spk < 200:
            reason, order = "few_training_speakers", 2
        else:
            reason, order = "diversity_and_control", 3
        priorities.append({"country": country, "country_ko": ko, "train_speakers": n_spk,
                           "validation_speakers": n_valid, "validation_top1": acc,
                           "priority": order, "reason": reason,
                           "validation_note": "small_sample" if n_valid < 10 else ""})
    priorities.sort(key=lambda r: (r["priority"], r["train_speakers"], r["country"]))
    write_csv(args.out / "country_priorities.csv", priorities, list(priorities[0]))
    jobs = []
    for row in priorities:
        query = row["country_ko"] + " 사람 한국어 자기소개 인터뷰"
        for filter_name in ("any", "creative_commons"):
            jobs.append({"country_hint": row["country"], "query": query, "filter": filter_name})
    if args.queries_json:
        jobs = json.loads(args.queries_json.read_text(encoding="utf-8"))
        if not jobs or any(j.get("country_hint") not in {r["country"] for r in priorities}
                           or not j.get("query") or j.get("filter") not in ("any", "creative_commons")
                           for j in jobs):
            parser.error("Custom jobs need a selected country, query, and supported filter")

    def cache_key(job):
        key = job["country_hint"] + ":" + job["filter"] + ":" + str(args.per_query)
        return key + ":" + job["query"] if args.queries_json else key
    cache_path = args.out / "search_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    pending = []
    for job in jobs:
        key = cache_key(job)
        if key not in cache or cache[key].get("error"):
            pending.append((key, job))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(search, job, args.per_query): (key, job) for key, job in pending}
        for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
            key, job = futures[future]
            try:
                cache[key] = future.result()
            except Exception as exc:
                cache[key] = dict(job, entries=[], error=str(exc)[:500])
            dump_json(cache_path, cache)
            print("[search {}/{}] {} {}: {} candidates".format(
                number, len(pending), job["country_hint"], job["filter"],
                len(cache[key]["entries"])), flush=True)
    candidates = {}
    priority_by_country = {r["country"]: r["priority"] for r in priorities}
    for job in jobs:
        result = cache[cache_key(job)]
        for entry in result["entries"]:
            video_id = entry["id"]
            if video_id not in candidates:
                candidates[video_id] = dict(entry, url="https://www.youtube.com/watch?v=" + video_id,
                                            country_hints=[], search_filters=[], queries=[],
                                            reserved_evaluation_video=video_id in reserved)
            row = candidates[video_id]
            for field, value in (("country_hints", job["country_hint"]),
                                 ("search_filters", job["filter"]), ("queries", job["query"])):
                if value not in row[field]:
                    row[field].append(value)
    meta_path = args.out / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    ordered = sorted(candidates.values(), key=lambda r: (
        r["reserved_evaluation_video"], "creative_commons" not in r["search_filters"],
        min(priority_by_country[c] for c in r["country_hints"]),
        not (r.get("duration") and 40 <= r["duration"] <= 1800), r["id"]))
    inspect_ids = set(filter(None, (s.strip() for s in args.inspect_ids.split(","))))
    if inspect_ids - set(candidates):
        parser.error("Video IDs missing from catalog: " + ", ".join(sorted(inspect_ids - set(candidates))))
    selected_pages = ([r for r in ordered if r["id"] in inspect_ids] if inspect_ids else
                      [r for r in ordered if not r["reserved_evaluation_video"]][:args.metadata_limit])
    todo = [r for r in selected_pages if r["id"] not in metadata or metadata[r["id"]].get("error")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(details, row): row for row in todo}
        for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = futures[future]
            try:
                metadata[row["id"]] = future.result()
            except Exception as exc:
                metadata[row["id"]] = {"id": row["id"], "error": str(exc)[:500]}
            dump_json(meta_path, metadata)
            print("[metadata {}/{}] {} license={}".format(
                number, len(todo), row["id"], metadata[row["id"]].get("license")), flush=True)
    flat = []
    for row in ordered:
        page = metadata.get(row["id"], {})
        row["metadata"] = page
        flat.append({"video_id": row["id"], "url": row["url"],
                     "title": page.get("title") or row.get("title"),
                     "channel": page.get("channel") or row.get("channel") or row.get("uploader"),
                     "duration_sec": page.get("duration") or row.get("duration"),
                     "country_hints": "|".join(row["country_hints"]),
                     "search_filters": "|".join(row["search_filters"]),
                     "license_metadata": page.get("license") or "unknown",
                     "metadata_status": "error" if page.get("error") else "inspected" if page else "pending",
                     "reserved_evaluation_video": int(row["reserved_evaluation_video"]),
                     "verified_country": "", "speaker_id": "", "country_evidence": "",
                     "start_sec": "", "end_sec": "", "korean_speech_verified": "",
                     "single_speaker_verified": "", "use_basis": "", "use_evidence": "",
                     "split": "", "review_status": "pending", "notes": ""})
    dump_json(args.out / "candidates.json", ordered)
    write_csv(args.out / "candidates.csv", flat, list(flat[0]) if flat else ["video_id", "url"])
    # Preserve manual edits when collection is resumed.
    review_path = args.out / "review_segments.csv"
    if not review_path.exists():
        write_csv(review_path, flat, list(flat[0]) if flat else ["video_id", "url"])
    summary = {"collected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "scope": "all_foreign_head_classes" if not selected else sorted(selected),
               "countries_searched": len(priorities), "queries": len(jobs),
               "query_errors": sum(bool(cache[cache_key(j)].get("error")) for j in jobs),
               "unique_candidates": len(ordered),
               "reserved_evaluation_videos_found": sum(r["reserved_evaluation_video"] for r in ordered),
               "cc_search_candidates": sum("creative_commons" in r["search_filters"] for r in ordered),
               "inspected_pages": sum(bool(r["metadata"]) for r in ordered),
               "explicit_cc_license_pages": sum("creative commons" in
                   str(r["metadata"].get("license") or "").lower() for r in ordered),
               "verified_training_speakers": 0, "audio_downloads": 0,
               "note": "Search hints and CC search filters are not verified labels or permissions."}
    dump_json(args.out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
