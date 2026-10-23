# OpenTracer PC

OpenTracer PC is a native desktop editor for Windows and macOS that trims golf videos and creates animated shot tracers.

## Requirements

- Rust stable toolchain
- FFmpeg and FFprobe available on `PATH`
- On macOS, Xcode Command Line Tools

## Install prerequisites

Install Rust from [rustup.rs](https://rustup.rs/).

On macOS, install FFmpeg with Homebrew:

```sh
brew install ffmpeg
```

On Windows, install FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) and ensure `ffmpeg` and `ffprobe` are on `PATH`. Install the Xcode Command Line Tools on macOS with `xcode-select --install`.

## Build and run

Run these commands from the project folder on either Windows or macOS:

```sh
cargo run
```

Build a release executable with `cargo build --release`.

## Editing workflow

Start at the welcome screen and select a video to open the trim screen. Set the kept section with the draggable **IN** and **OUT** handles, then continue to the optional camera-motion analysis step before tracing. Analysis samples the trimmed clip at 10 frames per second and estimates translation only; it does not crop, rotate, or alter the source video. Its progress is reported in analyzed frames, and you can cancel or skip it at any time. When enabled, the tracer follows the estimated scene movement. Skip the step or switch **Motion compensation** off in the tracing toolbar to return to the original behavior. Trimming is disabled in the tracing workspace; select **Back to trim** in the top-left to adjust the range. The editor has frame-by-frame controls, click-to-place keyframes, and tracer styling. Press **Space** or click **Play** to stream the preview at the source frame rate, or pause to inspect a frame. The tracer follows a smooth curved path through its keyframes with a continuous, feathered glow. Portrait videos are centered in the preview.

Frame navigation uses the source video's detected frame rate. Playback controls sit beneath the preview and above the timeline. In the tracing workspace, the timeline is scoped to the trimmed section so the clip being edited fills the navigation range. Tracer width, glow, and smoothness use whole-number 1–10 style scales, with 10 representing 100%. Playback frames are decoded off the UI thread, paced against a fixed frame clock, and previewed at a reduced resolution for smoother playback and scrubbing. The interface uses a dark, glass-inspired palette with blue accents. Export renders a transparent tracer sequence and composites it onto the trimmed source video, showing progress for tracer rendering and MP4 encoding. Save and reopen `.otracer` project files to continue editing later.

## Continuous integration

GitHub Actions checks formatting, tests, Clippy, and native builds on Windows and macOS for pushes and pull requests.
