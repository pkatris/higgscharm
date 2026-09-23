#!/usr/bin/env python3
"""
Merge CR parquet files per era and category into single files.
Uses pyarrow dataset API for efficiency.

Usage:
    python merge_cr_parquets.py --year 2022postEE
    python merge_cr_parquets.py --year all
    python merge_cr_parquets.py --workflow zzto4l_CR --year all
    python merge_cr_parquets.py --workflow hplusc_mva_4class_CR --year all --force
"""
import argparse
import glob
import awkward as ak
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

ERAS = ["2022postEE", "2022preEE", "2023preBPix", "2023postBPix"]
CATEGORIES = ["CR_3P1F", "CR_2P2F"]


def _normalize_table(table):
    """
    Collapse any list/large_list column to a true scalar via ak.firsts()
    (None for an empty list, the single value otherwise).

    Found 2026-09-18: the raw CR production tree mixes two schema
    generations for the same per-event "best candidate" columns
    (m4l_3p1f, pT_4l_3p1f, z1_mass_3p1f/z2_mass_3p1f, l3_*/l4_* fields,
    etc.) -- files written before the 2026-09-16 parquet_writer.py Arrow
    fix (see hczz_cr_pyarrow_nullable_list_bug_fixed memory) still store
    these as length<=1 large_list<T> (the pre-fix, uncollapsed shape),
    while files reprocessed/resubmitted since that fix store them as a
    plain scalar T (post-_fix_option_in_list() shape). Both shapes carry
    the same physical meaning -- one value per event -- so collapsing the
    list shape here (same ak.firsts() semantics already validated in
    parquet_writer.py) makes every file's schema agree before
    concatenation, without touching the underlying physics. This does NOT
    apply to genuinely multi-valued per-jet arrays (those aren't among the
    columns hitting this conflict -- verified directly: only known
    single-value-per-event columns showed a scalar/list split).
    """
    new_cols = []
    for name in table.column_names:
        col = table.column(name).combine_chunks()
        if pa.types.is_list(col.type) or pa.types.is_large_list(col.type):
            aarr = ak.from_arrow(col)
            # extensionarray=False: plain pyarrow array (matches the other
            # files' already-scalar columns), not awkward's own extension
            # type wrapper -- pa.concat_tables can't merge the two.
            col = ak.to_arrow(ak.firsts(aarr), extensionarray=False)
        new_cols.append(col)
    return pa.table(new_cols, names=table.column_names)


def merge_era_category(era, category, cr_base, merge_out, force=False):
    files = glob.glob(f"{cr_base}/{era}/*/{category}/*.parquet")
    if not files:
        print(f"  No files found for {era}/{category}")
        return

    out_path = Path(merge_out) / era / f"{category}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        print(f"  Already exists, skipping: {out_path}")
        return

    if out_path.exists() and force:
        print(f"  Force-overwriting: {out_path}")

    # Pre-filter corrupted files
    good_files = []
    bad_files = []
    for f in files:
        try:
            pq.read_metadata(f)
            good_files.append(f)
        except Exception as e:
            bad_files.append(f)
            print(f"  Skipping corrupted file: {f} ({e})")
    if bad_files:
        print(f"  Skipped {len(bad_files)} corrupted file(s), using {len(good_files)} good files")
    if not good_files:
        print(f"  No valid files found for {era}/{category}")
        return

    print(f"  Merging {len(good_files)} files -> {out_path}")
    # NOT pyarrow.dataset.to_table() -- that enforces one unified schema
    # across every file up front and crashes (ArrowNotImplementedError:
    # Unsupported cast from large_list<double> to double) on the real
    # scalar/list schema-version split described in _normalize_table's
    # docstring. Read + normalize each file individually instead, then do
    # a genuine schema-union concat.
    tables = [_normalize_table(pq.read_table(f)) for f in good_files]
    table = pa.concat_tables(tables, promote_options="permissive")
    print(f"  Rows: {table.num_rows:,}  Columns: {table.num_columns}")
    # Allow nulls in all fields to avoid schema conflicts across files
    new_schema = pa.schema([f.with_nullable(True) for f in table.schema])
    table = table.cast(new_schema)
    pq.write_table(table, out_path, compression="snappy")
    print(f"  Done: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="all", help="Era or 'all'")
    parser.add_argument("--workflow", default="hplusc_mva_4class_CR",
                        help="CR workflow name (default: hplusc_mva_4class_CR)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing merged files")
    args = parser.parse_args()

    cr_base   = f"/eos/user/s/snandaku/higgscharm/outputs/{args.workflow}"
    merge_out = f"/eos/user/s/snandaku/higgscharm/outputs/{args.workflow}_merged"

    print(f"Workflow:  {args.workflow}")
    print(f"CR base:   {cr_base}")
    print(f"Merge out: {merge_out}")

    eras = ERAS if args.year == "all" else [args.year]

    for era in eras:
        print(f"\n=== {era} ===")
        for cat in CATEGORIES:
            merge_era_category(era, cat, cr_base, merge_out, force=args.force)

    print("\nAll done!")


if __name__ == "__main__":
    main()
