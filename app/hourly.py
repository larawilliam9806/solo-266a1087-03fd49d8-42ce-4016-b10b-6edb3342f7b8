"""Per-file UTC hourly aggregation.

Every parquet file in the lake is reduced, exactly once, to a set of
``(buoy_id, UTC hour) -> (count, salinity sum)`` buckets.  These buckets are
persisted next to the file registry, so querying a day's hourly summary never
has to scan parquet again -- and, more importantly, compaction can move the
same observations from input files to one output file without changing the
summed result (the output is aggregated from the merged table, the inputs are
retired, so every observation is represented exactly once).

The aggregation is computed with Arrow in memory from the same table that is
being written, which is what lets upload ("register file + persist hourly
stats") and compaction ("retire inputs + register output + persist its stats")
be single-transaction operations.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.compute as pc

Bucket = tuple[str, str, int, float]


def hour_key(dt: datetime) -> str:
    """Canonical ``YYYY-MM-DDTHH:00:00Z`` key for a UTC hour."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")


def hourly_buckets(table: pa.Table) -> list[Bucket]:
    """Reduce ``table`` to ``(buoy_id, hour_utc, count, salinity_sum)`` rows.

    Duplicate observations are intentionally preserved: ``count`` counts every
    input row, including identical repetitions.
    """
    if table.num_rows == 0:
        return []

    hours = pc.floor_temporal(table.column("ts_utc"), unit="hour")
    grouped = (
        pa.table(
            {
                "buoy_id": table.column("buoy_id"),
                "hour_utc": hours,
                "salinity": table.column("salinity"),
            }
        )
        .group_by(["buoy_id", "hour_utc"], use_threads=False)
        # Count rows via a non-null key; sum salinity treating nulls as 0 so a
        # bucket with no usable value still counts its observations.
        .aggregate(
            [
                ("buoy_id", "count"),
                ("salinity", "sum"),
            ]
        )
    )
    cols = grouped.to_pydict()
    sums = pc.fill_null(pa.array(cols["salinity_sum"], type=pa.float64()), 0.0).to_pylist()
    return [
        (buoy, hour_key(hour), int(count), float(total))
        for buoy, hour, count, total in zip(
            cols["buoy_id"], cols["hour_utc"], cols["buoy_id_count"], sums
        )
    ]
