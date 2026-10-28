Tommy'sTracer - Easy Golf Tracer

INSTALLATION

Windows:
  Extract this archive and run tommys-tracer.exe.
  FFmpeg and FFprobe are required to open videos and export MP4 files.
  Download a Windows build from https://www.gyan.dev/ffmpeg/builds/,
  extract it, add its bin folder to your PATH, then restart the app.
  If Windows shows an unverified-app warning, this download is not
  code-signed. Only run it if you trust the source of the download.

macOS:
  Move Tommy's Tracer.app to Applications and open it. This app is not
  signed or notarized yet. If macOS blocks it, Control-click the app,
  choose Open, then confirm.
  FFmpeg and FFprobe are bundled in the app; no separate install is needed.
  The included FFmpeg license and build details are in the app's Resources.

Linux:
  Extract the archive, open a terminal in the extracted folder, and run:
    ./tommys-tracer

REQUIREMENTS

Windows and Linux require FFmpeg and FFprobe to be installed; they are used
to read videos and create MP4 exports and are not included in those packages.
Install FFmpeg from https://ffmpeg.org/download.html or your operating
system's package manager. Ensure both tools are available on PATH and restart
Tommy's Tracer after installation. The macOS package bundles universal
FFmpeg and FFprobe executables.

Linux builds use the system's standard X11/Wayland, OpenGL, and ALSA
libraries. Most desktop Linux distributions already include the runtime
libraries.

PROJECTS

Open existing .otracer project files from the app. Your source video must
remain available at its saved path.
