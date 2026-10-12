"""GDB Python observer. Reads inferior state only; never changes its values."""
import collections
import json
import struct
import gdb

EVENTS = 0
COUNTS = collections.Counter()
PER_STEP = collections.defaultdict(collections.Counter)
OBSERVED_STEPS = set()
PARENT_OBJECT = None
CLIMB_SERIAL = 0
CELL_UPDATE_SERIAL = collections.Counter()
PARA = 0x53ed7a0
CONTROL = 0x53ed5c0
GEN_SERIAL = 0
GEN_OBJECT = None
GEN_MODE_ADDRESS = None
GEN_PHASE_COUNTS = collections.Counter()
SNAPSHOT_PHASES = {'gen_randommode': 'entry',
                   'gen_after_history_scale': 'after_history_scale',
                   'gen_after_coefficient18': 'after_coefficient18',
                   'gen_after_coefficient28': 'after_coefficient28'}


def register(name):
    return int(gdb.parse_and_eval('$' + name))


def read(address, fmt):
    return struct.unpack(fmt, bytes(gdb.selected_inferior().read_memory(address, struct.calcsize(fmt))))[0]


def i32(address):
    return read(address, '<i')


def vector_descriptor(address):
    """Decode the two-dimensional Intel descriptor used by inspected loops.

    Evidence: gen_randommode 0x5ff88e/0x5ff8a4/0x5ff8a8 loads extents
    +0x30/+0x48 and byte stride +0x50. 0x602b82 and 0x6040d6 use
    (column-lower_bound)*stride; descriptor constructors set rank at +0x20,
    first stride at +0x38 and first lower bound at +0x40. Never infer 3xN.
    """
    raw = bytes(gdb.selected_inferior().read_memory(address, 0x60))
    fields = struct.unpack('<12Q', raw)
    pointer, element_bytes, rank = fields[0], fields[1], fields[4]
    result = {'address': address, 'raw_hex': raw.hex(), 'data_pointer': pointer,
              'element_bytes': element_bytes, 'rank': rank}
    if pointer == 0:
        result.update(status='no_data_pointer', dimensions=[], values=None)
        return result
    if element_bytes != 8 or rank != 2:
        raise ValueError('snapshot descriptor is not the inspected float64 rank-2 contract')
    dimensions = []
    for offset in (0x30, 0x48):
        extent, stride, lower = struct.unpack_from('<qqq', raw, offset)
        if extent < 0 or (extent and (stride == 0 or stride % 8)):
            raise ValueError('invalid descriptor extent or float64 byte stride')
        dimensions.append({'extent': extent, 'stride_bytes': stride,
                           'lower_bound': lower, 'upper_bound': lower + extent - 1})
    count = dimensions[0]['extent'] * dimensions[1]['extent']
    if count > 4096:
        raise ValueError('snapshot exceeds 4096-value diagnostic resource cap')
    # Values are [second-dimension index][first-dimension index], traversing
    # actual strides. Data pointer denotes the declared lower-bound element.
    values = [[read(pointer + first * dimensions[0]['stride_bytes'] +
                   second * dimensions[1]['stride_bytes'], '<d')
               for first in range(dimensions[0]['extent'])]
              for second in range(dimensions[1]['extent'])]
    result.update(status='read', dimensions=dimensions, values=values,
                  value_order='second dimension outer, first dimension inner')
    return result


def generator_snapshot(obj, label):
    global GEN_SERIAL, GEN_OBJECT, GEN_MODE_ADDRESS
    if label == 'gen_randommode':
        GEN_SERIAL += 1
        # Entry rdi is parent CSSW descriptor, rsi is ten-coefficient pattern;
        # gen 0x5ff764/0x5ff76e preserves them in r15/r14.
        descriptor = register('rdi')
        GEN_OBJECT = read(descriptor, '<Q')
        if GEN_OBJECT != obj:
            raise ValueError('generator descriptor disagrees with captured parent object')
        GEN_MODE_ADDRESS = register('rsi')
    if GEN_OBJECT != obj or GEN_MODE_ADDRESS is None:
        raise ValueError('generator snapshot has no matching entry context')
    phase = SNAPSHOT_PHASES[label]
    GEN_PHASE_COUNTS[(GEN_SERIAL, phase)] += 1
    mode = [read(GEN_MODE_ADDRESS + 8 * index, '<d') for index in range(10)]
    return {'generator_serial': GEN_SERIAL, 'phase': phase,
            'phase_visit': GEN_PHASE_COUNTS[(GEN_SERIAL, phase)],
            'mode_address': GEN_MODE_ADDRESS, 'mode_coefficients': mode,
            'history_coefficient_offset48': mode[9], 'lrigid_raw': i32(PARA + 0x130),
            # n0 is DWARF object+0x1788. The coefficient loops use this
            # destination and the scratch descriptor object+0x1848. Do not
            # give the latter an unclosed scientific interpretation.
            'destination_object_offset1788': vector_descriptor(obj + 0x1788),
            'scratch_object_offset1848': vector_descriptor(obj + 0x1848)}


def finish(reason):
    with open('trace-stop.json', 'w') as stream:
        json.dump({'reason': reason, 'events': EVENTS, 'counts': dict(COUNTS),
                   'observed_nsswsteps': sorted(OBSERVED_STEPS)}, stream, indent=2)
    gdb.execute('quit')


class Observer(gdb.Breakpoint):
    def __init__(self, address, label, object_source):
        super().__init__('*0x%x' % address, internal=False)
        self.label, self.object_source = label, object_source

    def stop(self):
        global EVENTS, PARENT_OBJECT, CLIMB_SERIAL
        try:
            # Most CSSW entries take a Fortran descriptor as rdi. The random
            # generator can consume a local descriptor: use captured parent.
            if self.object_source == 'descriptor':
                obj = read(register('rdi'), '<Q')
                PARENT_OBJECT = obj
            elif self.object_source == 'rsi':
                obj = register('rsi')
                PARENT_OBJECT = obj
            elif self.object_source == 'rbx':
                obj = register('rbx')
                PARENT_OBJECT = obj
            elif self.object_source == 'r13_descriptor':
                obj = read(register('r13'), '<Q')
                PARENT_OBJECT = obj
            else:
                obj = PARENT_OBJECT
            if obj is None:
                raise RuntimeError('generator reached without a captured CSSW parent')
            step = i32(obj + 0x2a74)
            if self.label == 'ssw_move':
                if step not in OBSERVED_STEPS and len(OBSERVED_STEPS) >= 2:
                    finish('two_distinct_outer_step_limit')
                OBSERVED_STEPS.add(step)
            if self.label == 'climb':
                CLIMB_SERIAL += 1
            cell = i32(obj + 0x2260)
            if self.label in ('update_mode0', 'get_random_after_choice') and cell:
                CELL_UPDATE_SERIAL[step] += 1
            EVENTS += 1
            COUNTS[self.label] += 1
            PER_STEP[step][self.label] += 1
            row = {'event': EVENTS, 'label': self.label, 'pc': register('pc'),
                   'object': obj, 'object_source': self.object_source,
                   'run_type': i32(PARA + 0x100), 'nsswstep': step, 'lcellmove_raw': cell,
                   'step_perssw': i32(CONTROL + 0x6c), 'climb_serial': CLIMB_SERIAL,
                   'cell_mode_event_serial_within_step': CELL_UPDATE_SERIAL[step],
                   'call_count': COUNTS[self.label], 'call_count_within_step': PER_STEP[step][self.label],
                   'ratio_atomcell': i32(PARA + 0x2db58), 'ng': i32(PARA + 0xf4),
                   'ng_cell': i32(PARA + 0xf8)}
            # Preserve raw registers and caller boundaries to distinguish
            # repeated reverse-communication entries during later analysis.
            # No unverified register is labeled a native cycle counter.
            row['registers'] = {name: register(name) for name in
                                ('rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi',
                                 'r12', 'r13', 'r14', 'r15', 'rsp', 'rbp')}
            callers = []
            frame = gdb.newest_frame()
            for _ in range(4):
                if frame is None:
                    break
                callers.append({'pc': int(frame.pc()), 'function': frame.name()})
                frame = frame.older()
            row['frames'] = callers
            if self.label in SNAPSHOT_PHASES:
                row['generator_snapshot'] = generator_snapshot(obj, self.label)
            if self.object_source == 'descriptor' or self.label == 'gen_randommode':
                row['return_address'] = read(register('rsp'), '<Q')
            with open('trace.jsonl', 'a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            if EVENTS >= 500:
                finish('trace_event_limit')
        except Exception as error:
            with open('trace-error.json', 'w') as stream:
                json.dump({'event': EVENTS, 'label': self.label, 'error': repr(error)}, stream)
            gdb.execute('quit 2')
        return False


for address, label, source in (
    (0x5e5de0, 'ssw_move', 'descriptor'),
    (0x5e80d0, 'get_random_mode0', 'descriptor'),
    (0x5e8173, 'get_random_after_choice', 'rsi'),
    (0x5f1d60, 'climb', 'descriptor'),
    (0x5f3750, 'climb_post_dispatch', 'r13_descriptor'),
    (0x5ff2e0, 'update_mode0', 'descriptor'),
    (0x5ff750, 'gen_randommode', 'parent'),
    (0x5ffa5b, 'gen_after_history_scale', 'parent'),
    (0x602d36, 'gen_after_coefficient18', 'parent'),
    (0x604136, 'gen_after_coefficient28', 'parent'),
    (0x608520, 'findmodepattern_cell', 'parent'),
    (0x608370, 'findmodepattern_atom', 'parent'),
):
    Observer(address, label, source)
