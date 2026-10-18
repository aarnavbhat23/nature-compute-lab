# Give each process its own clock

The learner estimates six relaxation rates from noisy observed input/output pairs. Each process then receives an update interval inversely proportional to its fitted rate. A validation trajectory selects the coarsest multiplier meeting a fixed approximation tolerance. Test inputs and durations are new, but the processes have the same rates estimated during training.

The dense comparison uses exactly the same fitted kinetics and updates at every observation. An approximately event-budget-matched uniform clock is an additional control. Between updates the asynchronous prediction is held constant; no hidden analytic query reconstructs a dense answer. External input changes trigger boundary updates and invalidate old deadlines.

This tests update allocation, not discovery of arbitrary biological time laws. The known relaxation equation is a strong prior. Update counts are not wall-time or FLOP savings: scanning schedules and serving observations has overhead, and small intervals can even require more updates than the dense grid. Report the observed ratio without assuming it exceeds one.

Local five-seed preview before the scheduled checkpoint: asynchronous MSE averaged 6.21e-5 versus 2.63e-5 for dense fitted kinetics and 4.78e-5 for the approximately budget-matched uniform clock. The per-seed dense/asynchronous update ratio averaged 3.44. Fewer updates are possible, but the uniform clock was more accurate at a similar event count. This is not an asynchronous-allocation win against that control.
