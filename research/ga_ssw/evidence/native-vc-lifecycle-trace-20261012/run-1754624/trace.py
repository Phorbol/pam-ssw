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


def register(name):
    return int(gdb.parse_and_eval('$' + name))


def read(address, fmt):
    return struct.unpack(fmt, bytes(gdb.selected_inferior().read_memory(address, struct.calcsize(fmt))))[0]


def i32(address):
    return read(address, '<i')


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
                   'run_type': i32(obj + 0x100), 'nsswstep': step, 'lcellmove_raw': cell,
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
            if self.object_source == 'descriptor' or self.label == 'gen_randommode':
                row['return_address'] = read(register('rsp'), '<Q')
            with open('trace.jsonl', 'a') as stream:
                stream.write(json.dumps(row) + '\n')
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
    (0x5f3750, 'climb_post_dispatch', 'rbx'),
    (0x5ff2e0, 'update_mode0', 'descriptor'),
    (0x5ff750, 'gen_randommode', 'parent'),
    (0x608520, 'findmodepattern_cell', 'parent'),
    (0x608370, 'findmodepattern_atom', 'parent'),
):
    Observer(address, label, source)
