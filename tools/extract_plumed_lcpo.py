#!/usr/bin/env python3
"""Generate the packaged LCPO lookup table from PLUMED's SAXS.cpp.

This maintainer tool is intentionally not part of the runtime package.  It
keeps the large residue/atom table traceable to the PLUMED implementation
instead of maintaining a second hand-transcribed copy.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

ENTRY = re.compile(
    r'lcpomap\["(?P<key>[^"]+)"\]\s*=\s*\{(?P<values>[^}]+)\};'
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="PLUMED src/isdb/SAXS.cpp")
    parser.add_argument("output", type=Path, help="generated YAML path")
    parser.add_argument("--revision", required=True, help="PLUMED git revision")
    args = parser.parse_args()

    text = args.source.read_text()
    start = text.index("SAXS::setupLCPOparam()")
    end = text.index("// assigns LCPO parameters", start)
    parameters: dict[str, list[float]] = {}
    for match in ENTRY.finditer(text[start:end]):
        values = [float(value.strip()) for value in match.group("values").split(",")]
        if len(values) != 5:
            raise ValueError(f"{match.group('key')}: expected five LCPO values")
        parameters[match.group("key")] = values
    if len(parameters) < 1000:
        raise ValueError(f"only found {len(parameters)} LCPO entries")

    document = {
        "source": {
            "repository": "https://github.com/hmcezar/plumed2",
            "path": "src/isdb/SAXS.cpp",
            "revision": args.revision,
            "function": "SAXS::setupLCPOparam",
        },
        "probe_radius_A": 1.4,
        "parameters": parameters,
    }
    args.output.write_text(
        yaml.safe_dump(document, sort_keys=False, default_flow_style=None)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
