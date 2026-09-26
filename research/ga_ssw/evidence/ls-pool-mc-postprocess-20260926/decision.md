# Remove offline MC geometry matching from the GPU worker

Observed: original C60 MC run1500677_3 saved its complete search result and
66 fresh checks, but no summary or offline-identity artifact before the process
wall limit. Search accounting closes at28338, fresh accounting at66. The
source orders the expensive diagnostic mc_archive_summary immediately before
remaining final artifacts; this identifies the likely interruption region,
not an independently profiled timing attribution.

The MC archive is not used for search, fresh qualification, graph discovery,
energy scoring or paid-request comparison. Remove that optional diagnostic
from the GPU worker entirely. Keep active pool matching unchanged. No core
algorithm, calculator, checkpoint format, numerical tolerance or search budget
changes. Offline grouping can be requested on CPU when it answers a question.

CPU1501023: MC EMT end-to-end path completed at67 search+3 fresh calls. Its
scientific result JSON and fresh-check JSON exactly equal the existing qualified
MC fixture fromCPU1500664. Summary, minima and checkpoint are saved; no offline
MC matching file is produced. See check.sbatch and check-1501023.out.
This is implementation verification, not additional scientific evidence.
