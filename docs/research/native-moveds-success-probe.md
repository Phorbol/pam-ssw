# Native `moveds` selected displacement and width probe

This bounded Unicorn probe executes uploaded ELF bytes for `n_normal` at
`0x578e20`, the selected-coordinate projection loop at
`0x5c8044–0x5c805b`, and the width store at `0x5c80c4`. Two non-collinear
two/three-atom synthetic cases are used. All pointers, arrays, loop bounds and
XMM initialization are synthetic; no LASP main, PES, protection, or surrounding
`moveds` caller is entered. The JSON records trace addresses and separates
synthetic dependencies from executed native instructions.

The native normalized vector and width projection/store are asserted against
independent NumPy calculations using `delta = selected - center`; a common
translation case is included. An earlier probe reversed the synthetic RCX/RSI
producer pointers and reported negative widths. That was a stub error, not
LASP behavior; its JSON is superseded. This closes only successful local arithmetic;
trajectory selection after retry and post-Allopt behavior remain open.
