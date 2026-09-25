"""Narrow Hookean input handling for the standalone fixed-cell SSW driver."""
import json
import numbers

from .ase_constraints import normalize_constraints, bind_hookean_surface


def prepare_ssw_restraints(atoms, surface, checkpoint=None):
    """Return clean internal atoms, bound objective, and canonical pair specs.

    The standalone SSW geometry supports only nonperiodic atom-pair Hookean
    terms.  Point/plane terms and FixAtoms belong to other driver contracts.
    """
    constraints = normalize_constraints(atoms)
    if constraints.fixed_indices:
        raise NotImplementedError('standalone SSW does not accept FixAtoms constraints')
    for spec in constraints.hookean_specs:
        kwargs = json.loads(spec)['kwargs']
        target = kwargs.get('a2')
        if (isinstance(target, bool) or
                not isinstance(target, (numbers.Integral,))):
            raise NotImplementedError('standalone SSW accepts only atom-pair Hookean constraints')
    specs = tuple(constraints.hookean_specs)
    if specs and atoms.pbc.any():
        raise NotImplementedError('atom-pair Hookean SSW currently requires nonperiodic atoms')
    if checkpoint is not None:
        saved = tuple(getattr(checkpoint, 'hookean_specs', ()))
        if saved != specs:
            raise ValueError('checkpoint Hookean restraint constraints do not match input atoms')
    clean = constraints.clean_atoms(atoms)
    return clean, bind_hookean_surface(surface, specs), specs


def validate_checkpoint_hookean_metadata(checkpoint):
    """Validate the schema/metadata relationship without evaluating a PES."""
    raw_specs = getattr(checkpoint, 'hookean_specs', ())
    if not isinstance(raw_specs, tuple):
        raise ValueError('checkpoint Hookean metadata must be an immutable tuple')
    specs = tuple(raw_specs)
    if checkpoint.schema_version == 6:
        if not specs:
            raise ValueError('schema 6 checkpoint requires Hookean restraint metadata')
        if checkpoint.initial.atoms.pbc.any():
            raise ValueError('schema 6 Hookean checkpoint requires nonperiodic atoms')
        from .ase_constraints import _canonical_hookean, _hookean_from_spec
        for spec in specs:
            if not isinstance(spec, str):
                raise ValueError('checkpoint Hookean metadata must contain canonical specifications')
            try:
                constraint = _hookean_from_spec(spec)
                if _canonical_hookean(constraint, len(checkpoint.initial.atoms)) != spec:
                    raise ValueError('checkpoint Hookean metadata is not canonical')
                target = json.loads(spec)['kwargs']['a2']
                if isinstance(target, list):
                    raise ValueError('checkpoint Hookean metadata must describe atom pairs')
            except (TypeError, KeyError, ValueError, IndexError) as error:
                raise ValueError('invalid checkpoint Hookean metadata') from error
    elif specs:
        raise ValueError('Hookean metadata requires checkpoint schema 6')
