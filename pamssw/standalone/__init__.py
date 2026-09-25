"""Experimental independent ASE SSW family.

No LASP/Java search runtime dependency. Numerical/release differences and
supported GA branches are explicit in README.md; this is not a claim of
full release parity or established search efficiency. Fixed-cell PBC and
joint log-strain VC have explicit, separate entry-point contracts.
"""
from .surface import ASESurface
from .generalized_numerics import generalized_central_ritz
from .paper_reference import (SSWConfig, LSSettings, SSWCheckpoint, SSWProgress,
                              BiasStageQuenchOutcome, run_ssw, run_ls_ssw,
                              save_ssw_checkpoint, load_ssw_checkpoint)
from .native_mc import NativeMCSettings, NativeMCState
from .paper_ga import PaperGAConfig, run_ga_ssw
from .ga_checkpoint import GACheckpoint
from .minimum_identity import MinimumIdentityView

__all__ = ['ASESurface', 'SSWConfig', 'LSSettings', 'PaperGAConfig',
           'run_ssw', 'run_ls_ssw', 'SSWCheckpoint', 'SSWProgress',
           'BiasStageQuenchOutcome', 'save_ssw_checkpoint',
           'load_ssw_checkpoint', 'run_ga_ssw', 'GACheckpoint', 'MinimumIdentityView', 'generalized_central_ritz']
__all__ += ['NativeMCSettings', 'NativeMCState']

# Experimental joint VC entry point; requires calculator stress and explicit metric.
from .vc_geometry import ASEStressSurface
from .vc_reference import VCSSWConfig, run_vc_ssw
__all__ += ['ASEStressSurface', 'VCSSWConfig', 'run_vc_ssw']

# Experimental sequential cell/atomic escape; same ASE E/F/stress contract.
from .block_ssw import BlockSSWConfig, run_block_ssw
__all__ += ['BlockSSWConfig', 'run_block_ssw']

# Experimental isolated articulated-chain SSW with full Cartesian true quench.
from .rc_reference import RCSSWConfig, run_rc_ssw
__all__ += ['RCSSWConfig', 'run_rc_ssw']

from .ls_native_reference import NativeLSSettings, run_native_ls_ssw
from .rc_forest_reference import RCForestSSWConfig, run_rc_forest_ssw
from .rc_topology import read_rigid_topology
from .periodic_ga_reference import PeriodicGAConfig, run_periodic_ga
__all__ += ['NativeLSSettings', 'run_native_ls_ssw', 'RCForestSSWConfig',
            'run_rc_forest_ssw', 'read_rigid_topology', 'PeriodicGAConfig', 'run_periodic_ga']

from .rc_vc_reference import RCVCSSWConfig, run_rc_vc_ssw
from .constrained_reference import (ConstrainedSSWConfig, ConstrainedSSWResult,
                                    ConstrainedCheckpoint, run_constrained_ssw,
                                    save_constrained_checkpoint, load_constrained_checkpoint)
from .rc_periodic_input import unwrap_rigid_molecules
from .molecular_periodic_ga_reference import run_molecular_periodic_ga
__all__ += ['RCVCSSWConfig', 'run_rc_vc_ssw', 'ConstrainedSSWConfig',
            'ConstrainedSSWResult', 'ConstrainedCheckpoint', 'run_constrained_ssw',
            'save_constrained_checkpoint', 'load_constrained_checkpoint',
            'unwrap_rigid_molecules',
            'run_molecular_periodic_ga']

from .native_height_policy import ConservativeNativeHeightPolicy
from .surface_ga_reference import SurfaceGAConfig, run_surface_ga
__all__ += ['ConservativeNativeHeightPolicy', 'SurfaceGAConfig', 'run_surface_ga']

from .minimal_angle_height import MinimalAngleHeightPolicy
from .initializers import initialize_type2, initialize_type3, initialize_type4
__all__ += ['MinimalAngleHeightPolicy', 'initialize_type2', 'initialize_type3',
            'initialize_type4']

from .initializers import initialize_type1, initialize_type0_regular
from .packing import pack_type0
__all__ += ['initialize_type1', 'initialize_type0_regular', 'pack_type0']

from .cluster_reconnection import ClusterReconnectionResult, ClusterTranslation, reconnect_clusters
__all__ += ['ClusterReconnectionResult', 'ClusterTranslation', 'reconnect_clusters']

from .native_local_pair import LocalPairResult, native_local_pair
__all__ += ['LocalPairResult', 'native_local_pair']

from .broyden_direction import paper_broyden_direction
__all__ += ['paper_broyden_direction']

from .ls_prequench import LSPrequenchSettings
__all__ += ["LSPrequenchSettings"]

from .recovered_rotation import RecoveredRotationSettings
__all__ += ['RecoveredRotationSettings']

from .starter_selection import StarterObservation, StarterPoolSnapshot
__all__ += ['StarterObservation', 'StarterPoolSnapshot']
