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

Start at the welcome screen and select a video to open the trim screen. Set the kept section with the draggable **IN** and **OUT** handles, then select **Continue to tracing** to open the keyframe editor. Trimming is disabled in the tracing workspace; select **Back to trim** in the top-left to adjust the range. The editor has frame-by-frame controls, click-to-place keyframes, and tracer styling. Press **Space** or click **Play** to stream the preview at the source frame rate, or pause to inspect a frame. The tracer follows a smooth curved path through its keyframes with a continuous, feathered glow. Portrait videos are centered in the preview.

Frame navigation uses the source video's detected frame rate. Preview frames are decoded in the background at a reduced resolution for responsive scrubbing. Export renders a transparent tracer sequence and composites it onto the trimmed source video, showing progress for tracer rendering and MP4 encoding. Save and reopen `.otracer` project files to continue editing later.

## Continuous integration

GitHub Actions checks formatting, tests, Clippy, and native builds on Windows and macOS for pushes and pull requests.
