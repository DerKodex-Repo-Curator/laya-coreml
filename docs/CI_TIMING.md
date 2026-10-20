# Idle-window test measurement

The original upstream fixture uses a 200 ms idle-unload threshold, then expects
an observing test thread's `Event.wait(0.1)` to return before unloading. On a busy
macOS runner the observer can resume after the full 200 ms has elapsed. That
observation fails even when unloading happens after the correct deadline.
The same assertion failed against the original upstream implementation in
[the retained comparison run](https://github.com/mizorewww/laya-mlx/actions/runs/39355649931).

The Core ML test adapter now records inference completion on the worker, just
before signaling `finished`. `BlockingRouter.unload` asserts that at least
**0.2 seconds** have actually elapsed before it signals `unloaded`. An early
unload raises before the event is set and therefore also fails the existing
bounded wait for unloading. The two 100 ms waits remain observations rather
than deadlines for when the test thread must be scheduled.

All eight batch/cancellation/failure combinations, the assertion that inference
and unload never overlap, and the existing eventual-unload timeouts remain.
There is no retry, skip, xfail, larger idle threshold, or production runtime change.
Unlike previous namespace-only adaptation, this is an explicit change to where
the timing assertion is measured. Output assertions and numeric model tolerances
are unchanged. The transformer checks the pinned fixture structure and fails
if its expected completion marker or two observations disappear.

`tests/test_idle_test_adapter.py` exercises the transformed upstream router with
a controlled clock: actual unloading at 100/199 ms must fail; at 201 ms or a late
2-second observation it must pass, both after successful and failed inference.
The complete 297-test adapted upstream HTTP suite also runs against the real
thread/event-loop implementation.
