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
    -v $(pwd):/workspace \
    -w /workspace \
    ghcr.io/kataglyphis/kataglyphis_beschleuniger:latest \
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

    # Installiere die für GStreamer-Plugins zur Laufzeit fehlenden System-Bibliotheken nach,
    # die im Dockerfile im finalen Stage nicht übernommen/installiert wurden.
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        libunwind-dev libdw-dev libgtk-4-dev libv4l-0 libjson-glib-1.0-0 dbus-x11 \
        libopenexr-dev libx264-dev libcdio-dev libspeex-dev libopenh264-dev libsrtp2-dev \
        libtwolame-dev libgsm1-dev libdav1d-dev libwavpack-dev libx265-dev libdc1394-dev \
        libvpx-dev libavcodec-dev libcsound64-dev libtbb12 libavfilter9 libavfilter-dev libavformat-dev || true

    bash Scripts/Linux/run_person_detection.sh
    '
