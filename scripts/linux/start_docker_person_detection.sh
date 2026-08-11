#!/bin/bash
set -e

# Erlaube Docker den Zugriff auf das lokale X11-Display (für GUI)
xhost +local:root || true

echo "Starte Docker Container mit Webcam-Unterstützung..."

# Mount /dev/video0 für die Webcam (Logitech C922)
docker run --rm -it \
    --ipc=host \
    --device /dev/video0:/dev/video0 \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e GTK_A11Y=none \
    -e LIBGL_ALWAYS_SOFTWARE=1 \
    -e GALLIUM_DRIVER=llvmpipe \
    -e GSK_RENDERER=cairo \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$(pwd):/workspace" \
    -w /workspace \
    ghcr.io/kataglyphis/kataglyphis_beschleuniger:latest-cross \
    bash -lc '
    set -e
    git config --global --add safe.directory /workspace || true
    # Fix für Bibliotheken aus /opt, da diese priorisiert geladen werden müssen
    export GDK_BACKEND=x11
    
    # Füge alle Library-Pfade aus /opt hinzu (z.B. OpenCV, FFmpeg, GStreamer)
    for libdir in $(find /opt ! -name "android*" -type d \( -name "lib" -o -name "lib64" -o -name "x86_64-linux-gnu" \)); do
        if [ -d "$libdir" ]; then
            export LD_LIBRARY_PATH="$libdir:$LD_LIBRARY_PATH"
        fi
    done
    export LD_LIBRARY_PATH="/opt/gstreamer/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

    # Only what the image genuinely does NOT ship. This list used to name 24
    # packages; 22 of them are already installed by ContainerHub'"'"'s
    # linux/scripts/03-media/runtime/install-deps.sh, so it was a stale copy of
    # that list which would drift every time the image changed.
    #
    # libgtk-4-dev is the real gap, and a deliberate one: ContainerHub excludes
    # it because the foreign-arch GTK dev chain pulls target-side Python and
    # breaks cross builds on python3-minimal'"'"'s postinst. Installing it here,
    # at runtime, in a throwaway container, is the right place for it - not in
    # the image. Keep this list minimal for the same reason; if something else
    # turns out to be missing, check install-deps.sh first, it probably is not.
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        libgtk-4-dev libavfilter9 || true

    bash scripts/linux/run_person_detection.sh
    '
