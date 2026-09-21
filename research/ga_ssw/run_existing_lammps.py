"""Process-local launch with broken unrelated LAMMPS plugin discovery disabled.

Set PYTHONPATH/LD_LIBRARY_PATH explicitly to the qualified archived wheel/MPI.
Does not edit installed packages or alter the requested physical calculator.
"""
import importlib.metadata
import runpy
import sys
original = importlib.metadata.entry_points
def filtered(**kwargs):
    if kwargs.get('group') == 'lammps.plugins':
        return importlib.metadata.EntryPoints(())
    return original(**kwargs)
importlib.metadata.entry_points = filtered
try:
    import lammps
finally:
    importlib.metadata.entry_points = original
print('LAMMPS runtime', lammps.__file__, lammps.__version__, flush=True)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
