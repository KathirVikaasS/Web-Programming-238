"""Writes sample_data/sample_farm.geojson. Run once:  python make_sample_geojson.py"""
import json
from pathlib import Path

from planner.sample import sample_geojson_dict

out = Path("sample_data")
out.mkdir(exist_ok=True)
(out / "sample_farm.geojson").write_text(json.dumps(sample_geojson_dict(), indent=2), encoding="utf-8")
Path("reports").mkdir(exist_ok=True)
(Path("reports") / ".gitkeep").touch()
print("Wrote sample_data/sample_farm.geojson")