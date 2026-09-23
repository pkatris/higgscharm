import os
import json
import shutil
import pathlib
import awkward as ak
from typing import List, Optional


def _fix_option_in_list(array):
    """
    Collapse awkward field types that deterministically trigger pyarrow's
    "Lists with non-zero length null components are not supported" on
    write -- confirmed 2026-09-16 against the real pyarrow==19.0.1/
    awkward==2.8.1 combo bundled in the coffea-base-almalinux9:0.7.30 grid
    image (the "extensionarray" workaround below doesn't apply there: that
    kwarg isn't present in this ak.to_arrow_table, so the plain
    ak.to_parquet fallback runs instead, and it hits this error for BOTH
    ak.packed() and a from-scratch to_list/from_iter rebuild -- it's a
    genuine type-shape incompatibility, not a buffer-sharing artifact).
    Verified against real hplusc_mva_4class_CR output (GluGluToContinto2Zto2E2Mu,
    2022preEE): every one of ~25 failing columns (m4l_3p1f, z1_mass_2p2f,
    cjets_h_dphi_3p1f, etc.) has one of two shapes, needing two different,
    semantics-preserving treatments:

      "N * option[var * ?T]" -- a per-event list of length<=1 wrapping a
          single "best candidate" value that may be missing (e.g.
          m4l_3p1f: one m4l per event, or None if this event has no 3P1F
          candidate). ak.firsts() collapses this to a true "N * ?T" scalar
          option, which is what these fields are actually meant to be --
          confirmed this alone fixes the write, verified against the real
          failing file.
      "N * option[var * T]"  -- a genuinely multi-valued per-event list
          (e.g. cjets_h_dphi_3p1f: one dphi per c-jet) that can be
          entirely missing for events with no candidate. ak.firsts() here
          would silently keep only the first c-jet and drop the rest --
          wrong. Replacing the missing whole-list with an empty list
          (ak.fill_none(..., [], axis=0)) preserves every real
          multi-element list untouched and also resolves the write.

    Any other type (plain scalars, and genuinely jagged jet-collection
    axes like jets.phi/cjets.phi that are never option-wrapped in the
    first place) passes through unchanged.
    """
    t = str(ak.type(array))
    if "option[var * ?" in t:
        return ak.firsts(array)
    if "option[var *" in t:
        return ak.fill_none(array, [], axis=0)
    return array


def dump_pa_table(
    arrays: dict, fname: str, location: str, subdirs: Optional[List[str]] = None
):
    subdirs = subdirs or []
    xrd_prefix = "root://"
    pfx_len = len(xrd_prefix)
    xrootd = False
    if xrd_prefix in location:
        try:
            import XRootD
            import XRootD.client

            xrootd = True
        except ImportError as err:
            raise ImportError(
                "Install XRootD python bindings with: conda install -c conda-forge xroot"
            ) from err
    local_file = (
        os.path.abspath(os.path.join(".", fname))
        if xrootd
        else os.path.join(".", fname)
    )
    merged_subdirs = "/".join(subdirs) if xrootd else os.path.sep.join(subdirs)
    destination = (
        location + merged_subdirs + f"/{fname}"
        if xrootd
        else os.path.join(location, os.path.join(merged_subdirs, fname))
    )
    out = {}
    for variable, array in arrays.items():
        out[variable] = _fix_option_in_list(array)

    arr = ak.Array(out)
    if len(arr) != 0:  # skip dataframes with empty entries
        # Use ak.to_arrow_table with extensionarray=False to convert option-in-list
        # awkward types to standard Arrow nullable types, avoiding pyarrow
        # ArrowNotImplementedError: "Lists with non-zero length null components".
        # This works for both awkward 1.x and 2.x, unlike passing extensionarray
        # directly to ak.to_parquet (which fails in awk1.x).
        # Some grid worker nodes resolve an awkward build whose to_arrow_table()
        # doesn't accept extensionarray (TypeError: unexpected keyword argument) --
        # inspect the signature at runtime and fall back to the default if absent,
        # rather than crashing the whole job.
        import inspect
        import pyarrow.parquet as _pq

        if "extensionarray" in inspect.signature(ak.to_arrow_table).parameters:
            # awkward >= 2.x: extensionarray=False converts option-in-list types to
            # standard Arrow nullable, avoiding ArrowNotImplementedError on write.
            _pq.write_table(ak.to_arrow_table(arr, extensionarray=False), local_file)
        else:
            # Older awkward: ak.to_arrow_table() uses extension arrays by default and
            # pyarrow can't write null-containing lists from them. Use ak.to_parquet
            # directly, which handles option-in-list types internally.
            ak.to_parquet(arr, local_file)
        if xrootd:
            copyproc = XRootD.client.CopyProcess()
            copyproc.add_job(local_file, destination, force=True)
            copyproc.prepare()
            status, response = copyproc.run()
            if status.status != 0:
                raise Exception(status.message)
            del copyproc
        else:
            dirname = os.path.dirname(destination)
            pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
            shutil.copy(local_file, destination)
            assert os.path.isfile(destination)
        pathlib.Path(local_file).unlink()


def dump_sumw_json(
    sumw: float, dataset: str, fname: str, location: str, subdirs: Optional[List[str]] = None
):
    """
    Persist the pre-selection sum of genWeight for one processed chunk.

    Written unconditionally (even when zero events survive the analysis
    selection) so that downstream xs*lumi normalization can divide by the
    true full-sample sum of weights rather than a post-selection subset.
    """
    subdirs = subdirs or []
    xrd_prefix = "root://"
    xrootd = False
    if xrd_prefix in location:
        try:
            import XRootD
            import XRootD.client

            xrootd = True
        except ImportError as err:
            raise ImportError(
                "Install XRootD python bindings with: conda install -c conda-forge xroot"
            ) from err
    local_file = (
        os.path.abspath(os.path.join(".", fname))
        if xrootd
        else os.path.join(".", fname)
    )
    merged_subdirs = "/".join(subdirs) if xrootd else os.path.sep.join(subdirs)
    destination = (
        location + merged_subdirs + f"/{fname}"
        if xrootd
        else os.path.join(location, os.path.join(merged_subdirs, fname))
    )
    with open(local_file, "w") as f:
        json.dump({"dataset": dataset, "sumw": sumw}, f)
    if xrootd:
        copyproc = XRootD.client.CopyProcess()
        copyproc.add_job(local_file, destination, force=True)
        copyproc.prepare()
        status, response = copyproc.run()
        if status.status != 0:
            raise Exception(status.message)
        del copyproc
    else:
        dirname = os.path.dirname(destination)
        pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
        shutil.copy(local_file, destination)
        assert os.path.isfile(destination)
    pathlib.Path(local_file).unlink()


# Function taken from HiggsDNA
# https://gitlab.cern.ch/HiggsDNA-project/HiggsDNA/-/blob/master/higgs_dna/utils/dumping_utils.py
def dump_ak_array(
    akarr: ak.Array,
    fname: str,
    location: str,
    subdirs: Optional[List[str]] = None,
) -> None:
    """
    Dump an ak array to disk at location/'/'.join(subdirs)/fname.
    """
    subdirs = subdirs or []
    xrd_prefix = "root://"
    pfx_len = len(xrd_prefix)
    xrootd = False
    if xrd_prefix in location:
        try:
            import XRootD  # type: ignore
            import XRootD.client  # type: ignore

            xrootd = True
        except ImportError as err:
            raise ImportError(
                "Install XRootD python bindings with: conda install -c conda-forge xroot"
            ) from err
    local_file = (
        os.path.abspath(os.path.join(".", fname))
        if xrootd
        else os.path.join(".", fname)
    )
    merged_subdirs = "/".join(subdirs) if xrootd else os.path.sep.join(subdirs)
    destination = (
        location + merged_subdirs + f"/{fname}"
        if xrootd
        else os.path.join(location, os.path.join(merged_subdirs, fname))
    )
    ak.to_parquet(ak.fill_none(akarr, [], axis=0), local_file)
    if xrootd:
        copyproc = XRootD.client.CopyProcess()
        copyproc.add_job(local_file, destination, force=True)
        copyproc.prepare()
        status, response = copyproc.run()
        if status.status != 0:
            raise Exception(status.message)
        del copyproc
    else:
        dirname = os.path.dirname(destination)
        pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
        shutil.copy(local_file, destination)
        assert os.path.isfile(destination)
    pathlib.Path(local_file).unlink()
