"""Compatibility imports for archived reproduction scripts.

Independent Python behavior now lives in pamssw.standalone. No native runtime
is used by these descriptor/archive functions.
"""
from pamssw.standalone.legacy_descriptor import (
    cluster_descriptor, descriptor_similarity, same_projection,
    remove_duplicates, merge_archive, energy_window, compete_cumulative,
)
