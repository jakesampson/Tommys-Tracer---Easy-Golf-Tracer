# Tommy'sTracer - Easy Golf Tracer

Tommy'sTracer - Easy Golf Tracer is a native desktop editor for Windows, macOS, and Linux that trims golf videos and creates animated shot tracers.

The app icon and welcome-screen logo are bundled from `assets/branding`. Additional brand artwork and icon variations are kept in the same folder.

## Requirements

- Rust stable toolchain
- FFmpeg and FFprobe available on `PATH`
- On macOS, Xcode Command Line Tools

## Install prerequisites

Install Rust from [rustup.rs](https://rustup.rs/) to build from source.

On macOS, install FFmpeg with Homebrew:

```sh
brew install ffmpeg
```

On Windows, install FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) and ensure `ffmpeg` and `ffprobe` are on `PATH`. Install the Xcode Command Line Tools on macOS with `xcode-select --install`. On Linux, install FFmpeg and the X11/Wayland, OpenGL, and ALSA development packages for your distribution.

## Build and run

Run these commands from the project folder:

```sh
cargo run
```

Build a release executable with `cargo build --release`.

## Editing workflow

Start at the welcome screen and select a video to open the trim screen. Set the kept section with the draggable **IN** and **OUT** handles, then continue to the optional camera-motion analysis step before tracing. Analysis samples the trimmed clip at 10 frames per second and estimates translation only; it does not crop, rotate, or alter the source video. Its progress is reported in analyzed frames, and you can cancel or skip it at any time. When enabled, the tracer follows the estimated scene movement. Skip the step or switch **Motion compensation** off in the tracing toolbar to return to the original behavior. Trimming is disabled in the tracing workspace; select **Back to trim** in the top-left to adjust the range. The editor has frame-by-frame controls, click-to-place keyframes, and tracer styling. In the tracing workspace, use the zoom buttons or scroll over the preview to zoom in at the pointer; **right-click-drag** to pan and **Reset** to return to fit. Click the preview to activate keyboard editing. A single left-click places a keyframe; left-click-drag the tracer point at the current frame to move its keyframe. Use **Left/Right Arrow** to step frames; when stepping away from an existing keyframe, a matching keyframe is added on the adjacent frame for continued editing. Use **WASD** to nudge by 10 preview pixels per press, or 5 pixels when zoomed past 400%; hold **Shift** to nudge by 1 pixel. The tracer line, glow, and endpoint marker all become slimmer as you zoom in to keep the ball visible. Press **Space** or click **Play** to stream the preview at the source frame rate, or pause to inspect a frame. The tracer follows a smooth curved path through its keyframes with a continuous, feathered glow. Portrait videos are centered in the preview.

Frame navigation uses the source video's detected frame rate. Playback controls sit beneath the preview and above the timeline. In the tracing workspace, the timeline is scoped to the trimmed section so the clip being edited fills the navigation range. Tracer width, glow, and smoothness use whole-number 1–10 style scales, with 10 representing 100%; choose the tracer and glow colors independently beside their respective controls. The trail-duration slider sets how long the tracer draws before its oldest end begins to disappear (default 5 seconds). The trail-pace slider controls how quickly that end advances after the duration: 0.25x, 0.5x, or 1.0x of timeline speed. The default 0.5x makes the old end recede at half the rate the tracer is drawn. The keyframe list scrolls independently so style controls stay accessible during longer edits. Playback frames are decoded off the UI thread, paced against a fixed frame clock, and previewed at a reduced resolution for smoother playback and scrubbing. The interface uses a dark, glass-inspired palette with blue accents. Export renders a transparent tracer sequence and composites it onto the trimmed source video, showing progress for tracer rendering and MP4 encoding. Save and reopen `.otracer` project files to continue editing later.

## Continuous integration

GitHub Actions checks formatting, tests, Clippy, and native builds on Windows, macOS, and Linux for pushes and pull requests.

## GitHub releases

Push a version tag to build and publish downloadable packages automatically:

```sh
git tag v0.1.0
git push origin v0.1.0
```

If a release run fails after the tag has been pushed, first push the workflow fix to `main`, then run **Build release packages** from the Actions tab with the existing version tag as the `release_tag` input. This rebuilds from that tag and publishes without moving or recreating it.

The workflow creates a Windows x86_64 ZIP, a universal macOS `.app` ZIP for Intel and Apple Silicon, and a Linux x86_64 tarball, then attaches them to a GitHub Release with generated notes. The macOS app bundles universal LGPL-only FFmpeg and FFprobe builds and uses Apple's VideoToolbox H.264 encoder for export; the corresponding FFmpeg source archive is published alongside the app. Windows and Linux packages still require FFmpeg and FFprobe to be installed separately and available on `PATH`. The Windows executable is not code-signed, so SmartScreen may show an unverified-app warning. The macOS app is not signed or notarized, so macOS may require users to Control-click and choose **Open** the first time.
