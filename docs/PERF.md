# Throughput design

Goal: at least 10× more simulated matches per minute on the same machine. Benchmark with `frsbot.tools.benchmark_engine`
before and after each step.

1. **The driver runs inside the engine process.**
   - Port `bot/driver.py` to JS next to `server.mjs`. Python stays the reference implementation, and a parity test
     compares inputs from both on recorded states.
   - Python sends goals at planner rate, about 4 Hz per robot. Node runs the driver every tick and uses `fs_skip`
     when the inputs are held.
2. **Batched, binary IPC.**
   - `step_until_decision(batch)` advances every match in a worker to the next decision point. It returns a packed
     float32 obs array, plus a separate rewards array and a done mask.
   - Use a length-prefixed binary format (or shared memory) instead of JSON lines.
3. **Vector environments.**
   - `VecWasmEnv(n_workers=nproc, matches_per_worker=k)` gives one Node worker per core, each running k matches
     (separate `fs_new` handles).
   - Policy inference is batched across every match and robot each decision step.
4. **Data.**
   - Decode replays and recordings once, in parallel (`multiprocessing.Pool`), into per-match `.npz` arrays under
     `data/cache/`. They are memory-mapped by analysis and training.
   - Input recovery runs on independent ~2 s segments. Each segment resets the engine to the recorded state
     (`fo_body_set`/`fo_ball_set`), so segments run in parallel across cores.
5. **Recording.** The spectator batch mode runs up to 2–4 concurrent watches (polite cap).
6. **Live.** The JS driver from (1) runs in the page at 60 Hz, and the Python planner sends goals asynchronously.
