# Staged original water workflow evidence

Original run directories are under
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/runs/`.
Review text copies here have trailing whitespace stripped. Untouched original
artifacts, potentials and native logs remain there; source hashes are recorded.
The two main runs both reached their 120-second limit. They remain in the evidence
set. Trial 01 used three initial candidates; trial 02 used one. Both requested
GANum=6, allowing nonzero crossover and each integer-divided mutation allocation.

Trial 01 omitted required common Gaussian auxiliary files and `input/mc`.
Trial 02 retained Gaussian auxiliaries but still lacked `input/mc`; the original
code attempted to write rigidbody there and reported IOExceptions. Before the
staged suffix, an empty `input/mc` directory was created so original code could
write its own rigidbody file. The final preparation script supplies both paths.
No original jar or native binary was patched.

`FinishWaterProbe.java` was compiled against the supplied `sgn.jar` into trial 02's
`soft/` directory with the isolated JDK 17.0.2. From the run directory, the archived
`finish_bounded.py` executed `java -cp .:sgn.jar FinishWaterProbe` with cwd=`soft`,
Intel MPI 2021.13 loaded, OMP/MKL/OPENBLAS threads=1, and
`JAVA_TOOL_OPTIONS=-Xmx512m -XX:ActiveProcessorCount=1`. The suffix had a 90-second
limit and completed in 86.363 seconds. The script is a record of that run, not a
portable launcher from this review folder.

The suffix reconstructs the archive from the completed initial trajectory and
six completed GA optimizations, recomputes original classification using retained
references, then performs quick/fine SSW and export through original classes.
This is staged interface reproduction, not a bitwise continuation: original Java
and LASP random streams are not restored, archive reconstruction can expose order
and serialization differences, and the interrupted quick task is retained as cost.

There are 14 final archive entries. Their lowest stored energy is -220.899429 eV.
Archive size is not a count of independently certified distinct minima. No full
force-call accounting, independent force evaluation or physical validation has
been completed. The actual final SGN/NNA exchange agrees with Python within
8.33e-17 for four projection rows, with matching energy order.
