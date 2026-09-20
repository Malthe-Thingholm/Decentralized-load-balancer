"""Phase 1: core engine + TrueModel stub + assertion wrapper.

DEPRECATED: This directory contains the Phase 1 scaffold with alternate
type definitions (dict-based Task/Node). All active development uses the
src/ directory with the canonical type system:

- src/task.py -- Task dataclass (cpu_req, memory_req, network_req, ...)
- src/node.py -- Node dataclass (cpu_cap, memory_cap, network_cap, ...)
- src/event.py -- Simulator event engine
- src/truesim.py -- TrueModel (stochastic black-box cost)
- src/guess.py -- GuessingModel (imperfect estimator)
- src/balancer.py -- Balancer protocol

The sim/ engine uses an enum-based EventType system; src/event.py uses
string-based event types. The src/ version is canonical.

New code should import from src/, not sim/.
"""
