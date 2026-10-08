# Live mode: how it works, and why it behaves the way it does

This explains the dashboard's **Live** mode on the MacBook M1 (8 GB): its settings, why real-time mode
takes about 15 s to start, and why the boxes don't update on every frame. All numbers were measured on this machine.

## The settings (config.yaml)

```yaml
detection:
  onnx_providers: [CoreMLExecutionProvider, CPUExecutionProvider]  # where the model runs
live:
  parallel_cams: true    # detect the 3 cameras at the same time: ~25% faster, same results
  detect_every_n: 1      # 1 = detect every frame; 3 = every 3rd frame, reuse in between
  realtime: false        # true = video plays at normal speed, detector runs in the background
```

Restart the dashboard after changing them (`Ctrl+C`, then `venv/bin/python -m dashboard.server`).

| Machine | Setting | What you get |
|---|---|---|
| M1, best quality (default) | `detect_every_n: 1`, `realtime: false` | about 1 fps, every frame detected; also updates the Replay data |
| M1, faster | `detect_every_n: 3` | about 2.6 fps; boxes refresh every 3rd frame |
| M1, smooth | `realtime: true` | about 27 fps, smooth like Replay; boxes and ball update only about every 1–2 s |
| Faster Mac or NVIDIA PC | `realtime: true` | smooth video, with boxes updating far more often |
| NVIDIA GPU | `onnx_providers: [CUDAExecutionProvider, CPUExecutionProvider]`, plus `venv/bin/pip install onnxruntime-gpu` | much faster detection |

The stats strip shows two numbers, **video · detector** fps. When the detector number is close to the video
number, Live looks exactly like Replay.

## Why real-time mode takes about 15 s to start

Real-time mode runs the detector in a **separate process**, a second Python program, so it can't slow down
the video. That program starts from zero every time you click Start:

1. **Start Python and load the libraries** (a few seconds). The detection library is large, a bit like a huge `node_modules`.
2. **Load the model file** from disk.
3. **CoreML compiles the model** for the M1's GPU and Neural Engine. This is the longest part, about 10 s.
   macOS does it once per process before the first detection, like a JIT warm-up.

The normal Live mode does this too, but the dashboard keeps that model loaded after the first run.
Real-time mode starts a fresh process each time, so it pays the cost on every Start.
The log shows "starting the detector process" while this happens, and the clip starts once the detector is ready.

## Why the boxes don't update in real time

The numbers don't fit:

- The video needs a new frame every **33 ms** (30 fps).
- One detection round, all 3 cameras, takes about **1 s** on the M1, and about **1.5–2 s** in real-time mode,
  because the detector runs at low priority so the video stays smooth.

So 30 to 50 video frames go by between two detection results. The video can't wait for the detector, so it
keeps playing and **shows the most recent result it has**. The boxes stay still, then jump when a new result arrives.

They also **lag a little**. A result that arrives now describes the frame the detector started on about
1.5 s ago, so the boxes sit where the players *were*.

It's like a UI animating at 60 fps while it polls a slow API that takes 1.5 s per request:
the screen always shows the last response.

```
video frames:  |||||||||||||||||||||||||||||||||||||||||||||||||||||  30 per second
detections:    ^                         ^                         ^   about 1 every 1.5–2 s
boxes shown:   [ result A ...............][ result B ...............]
```

## Why a separate process, and not a thread

Python lets only one thread run Python code at a time (the "GIL"). When the detector ran in a thread inside
the dashboard, its Python work kept blocking the video, which then played at only 10–17 fps. A separate process
has its own Python, and it runs at low priority, so the video always gets the CPU first. The price is a second
copy of the model in memory, about 0.5–1 GB of extra RAM.

## Other things to know

- Real-time runs **don't update the Replay data**. Only the default setting (`detect_every_n: 1`,
  `realtime: false`) saves a fresh, full-quality detection cache.
- The model runs on CoreML (the M1's GPU and Neural Engine). On the CPU alone it takes about 5.4 s per frame instead of 0.3 s.

## Ideas to make it better (not built yet)

| Idea | Effect on the M1 |
|---|---|
| **Keep the detector process alive** between runs | No 15 s wait after the first Start |
| **Detect the live camera more often**, the other two less often | The camera people are watching updates about 3× more often |
| **Move the ball marker between detections** using the ball's last speed (the tracker already knows it) | The ball glides instead of jumping. It's a guess, not a real detection. |
| **Faster hardware** | The only way to get fresh detections on every frame |
