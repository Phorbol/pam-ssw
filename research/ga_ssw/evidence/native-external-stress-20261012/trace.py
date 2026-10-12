"""Read-only GDB observer for the external-to-native full-stress handoff."""
import json
import struct
import gdb

EVENTS = 0
CLASS_EVENTS = 0
LAST_OTHERPOT = None
OUTPUT = "native-stress-trace.jsonl"
CALLBACK_LOG = "callback-events.jsonl"
# `module_str_mp_strt_` base from the existing DWARF/source audit; its legacy
# external tensor starts at object/global offset +0x110 (not class_struc +0x110).
LEGACY_STRT = 0x53e88e0
EXTERNAL_DEDCELL = 0x5d71d18


def read_bytes(address, count):
    return bytes(gdb.selected_inferior().read_memory(address, count))


def i32(address):
    return struct.unpack("<i", read_bytes(address, 4))[0]


def f64(address):
    return struct.unpack("<d", read_bytes(address, 8))[0]


def f64_array(address, n=9):
    raw = read_bytes(address, n * 8)
    return list(struct.unpack("<" + "d" * n, raw))


def matrix_views(values):
    return {"fortran_column_major": [[values[i + 3*j] for j in range(3)] for i in range(3)],
            "row_major": [values[3*i:3*i+3] for i in range(3)],
            "raw_values": values}


def latest_callback():
    try:
        with open(CALLBACK_LOG) as stream:
            last = None
            for line in stream:
                if line.strip():
                    last = json.loads(line)
        return last
    except FileNotFoundError:
        return None


def append(row):
    global EVENTS
    EVENTS += 1
    row["trace_event"] = EVENTS
    with open(OUTPUT, "a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def stop(reason):
    with open("trace-stop.json", "w") as stream:
        json.dump({"reason": reason, "events": EVENTS,
                   "class_stress2dedlatt_entries": CLASS_EVENTS}, stream, indent=2)
    gdb.execute("quit")


class FlagGetter(gdb.Breakpoint):
    def stop(self):
        append({"kind": "external_dedcell_getter_return", "pc": int(gdb.parse_and_eval("$pc")),
                "getter_return_eax": int(gdb.parse_and_eval("$eax")),
                "global_flag": i32(EXTERNAL_DEDCELL)})
        return False


class ResetBeforeClear(gdb.Breakpoint):
    def stop(self):
        append({"kind": "module_pes_reset_before_flag_reset",
                "pc": int(gdb.parse_and_eval("$pc")),
                "global_flag_before_reset": i32(EXTERNAL_DEDCELL)})
        return False


class ResetAfterWrite(gdb.Breakpoint):
    def stop(self):
        append({"kind": "module_pes_reset_after_flag_write",
                "pc": int(gdb.parse_and_eval("$pc")),
                "global_flag_after_write": i32(EXTERNAL_DEDCELL),
                "r15d_written": int(gdb.parse_and_eval("$r15d"))})
        return False


class OtherPotReturn(gdb.FinishBreakpoint):
    def __init__(self, frame):
        super().__init__(frame, internal=True)

    def stop(self):
        global LAST_OTHERPOT
        values = f64_array(LEGACY_STRT + 0x110)
        callback = latest_callback()
        LAST_OTHERPOT = {"legacy_tensor": matrix_views(values),
                         "effective_external_dedcell": i32(EXTERNAL_DEDCELL),
                         "callback_request": None if callback is None else callback.get("request")}
        append({"kind": "otherpot_return", "legacy_strt_tensor": matrix_views(values),
                "effective_external_dedcell": i32(EXTERNAL_DEDCELL),
                "latest_callback": callback})
        return False


class OtherPotEntry(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*0x4150a0", internal=False)

    def stop(self):
        OtherPotReturn(gdb.newest_frame())
        return False


class ClassStressEntry(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*0x59cb10", internal=False)

    def stop(self):
        global CLASS_EVENTS, LAST_OTHERPOT
        try:
            descriptor = int(gdb.parse_and_eval("$rdi"))
            obj = struct.unpack("<Q", read_bytes(descriptor, 8))[0]
            if not obj:
                raise ValueError("class stress2dedlatt descriptor has null object")
            stress = matrix_views(f64_array(obj + 0x128))
            cell = matrix_views(f64_array(obj + 0x0e0))
            celli = matrix_views(f64_array(obj + 0x388))
            dedlatt = matrix_views(f64_array(obj + 0x3d0))
            volume = f64(obj + 0x608)
            callback = latest_callback()
            relation = None
            if callback is not None and callback.get("cell_A") is not None:
                cb = callback["cell_A"]
                fc = cell["fortran_column_major"]
                relation = {"same_cell_max_abs": max(abs(cb[i][j]-fc[i][j])
                                                       for i in range(3) for j in range(3)),
                            "transposed_cell_max_abs": max(abs(cb[i][j]-fc[j][i])
                                                            for i in range(3) for j in range(3))}
            CLASS_EVENTS += 1
            append({"kind": "class_struc_mp_stress2dedlatt_entry",
                    "pc": int(gdb.parse_and_eval("$pc")),
                    "descriptor": descriptor, "object": obj,
                    "class_stress": stress, "class_cell": cell,
                    "class_celli": celli, "class_volume_A3": volume,
                    "external_pressure_stored": f64(0x53ed7a0 + 0x2db38),
                    "class_dedlatt_at_entry": dedlatt,
                    "latest_callback": callback,
                    "nearest_otherpot_return": LAST_OTHERPOT,
                    "callback_request_id_matches_otherpot": (
                        callback is not None and LAST_OTHERPOT is not None and
                        callback.get("request") == LAST_OTHERPOT.get("callback_request")),
                    "class_atom_coordinates_read": False,
                    "callback_vs_class_cell": relation,
                    "effective_external_dedcell": i32(EXTERNAL_DEDCELL)})
            LAST_OTHERPOT = None
            if CLASS_EVENTS >= 3:
                stop("third_class_stress2dedlatt_entry")
        except Exception as error:
            with open("trace-error.json", "w") as stream:
                json.dump({"kind": "class_stress_entry_read_error",
                           "error": repr(error), "class_entries": CLASS_EVENTS}, stream, indent=2)
            gdb.execute("quit 2")
        return False


FlagGetter("*0x411445", internal=False)
ResetBeforeClear("*0x414e1f", internal=False)
ResetAfterWrite("*0x414eb7", internal=False)
OtherPotEntry()
ClassStressEntry()
