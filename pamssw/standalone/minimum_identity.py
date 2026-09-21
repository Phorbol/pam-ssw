"""Caller-defined structural identity view for raw SSW observations.

This module deliberately supplies no geometry metric or threshold.  The
caller matcher owns translation, rotation, permutation and PBC semantics.
"""
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class MinimumIdentityView:
    representative_indices: list
    observation_to_representative: list
    failures: list
    match_calls: int = 0
    processed_count: int = 0


def update_identity_view(view, minima, matcher, *, start=None):
    """Process only newly appended raw minima, preserving first representatives."""
    if not callable(matcher):
        raise TypeError('structure_matcher must be callable')
    if start is None:
        start = view.processed_count
    if start != view.processed_count or start < 0 or start > len(minima):
        raise ValueError('identity view processed count is inconsistent')
    if len(view.observation_to_representative) != start:
        raise ValueError('identity view mapping length is inconsistent')
    for observation_index in range(start, len(minima)):
        candidate = minima[observation_index].atoms.copy()
        matched = None
        for representative_index in view.representative_indices:
            try:
                view.match_calls += 1
                same = bool(matcher(candidate.copy(), minima[representative_index].atoms.copy()))
            except Exception as error:
                view.failures.append(dict(observation_index=observation_index,
                                          representative_index=representative_index,
                                          error=f'{type(error).__name__}: {error}'))
                view.observation_to_representative.append(None)
                matched = 'failed'
                break
            if same:
                matched = representative_index
                break
        if matched == 'failed':
            continue
        if matched is None:
            view.representative_indices.append(observation_index)
            view.observation_to_representative.append(observation_index)
        else:
            view.observation_to_representative.append(matched)
    view.processed_count = len(minima)
    return view


def clone_identity_view(view):
    return deepcopy(view)
