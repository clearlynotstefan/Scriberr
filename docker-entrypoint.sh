#!/bin/bash
set -e

# Default values
PUID=${PUID:-1000}
PGID=${PGID:-1000}

echo "=== Scriberr Container Setup ==="
echo "Requested UID: $PUID, GID: $PGID"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/
# echo "LD_LIBRARY_PATH is: $LD_LIBRARY_PATH"

# Function to setup user if needed
setup_user() {
    local target_uid=$1
    local target_gid=$2
    local current_uid
    local current_gid

    current_uid=$(id -u appuser)
    current_gid=$(id -g appuser)

    echo "Current appuser UID: $current_uid, GID: $current_gid"

    # Ensure appuser's primary group has the requested GID. If that GID already
    # exists (common on Unraid, e.g. users=100), simply attach appuser to it.
    if [ "$current_gid" != "$target_gid" ]; then
        if getent group "$target_gid" >/dev/null 2>&1; then
            echo "Using existing group with GID=$target_gid"
            usermod -g "$target_gid" appuser
        else
            local current_group
            current_group=$(id -gn appuser)
            echo "Changing group $current_group from GID=$current_gid to GID=$target_gid"
            groupmod -g "$target_gid" "$current_group"
            usermod -g "$target_gid" appuser
        fi
    fi

    # Dockerfile creates appuser with UID 10001. Always reconcile the actual
    # account UID with PUID, including the nominal default PUID=1000.
    if [ "$current_uid" != "$target_uid" ]; then
        echo "Changing appuser UID from $current_uid to $target_uid"
        usermod -u "$target_uid" appuser
    fi

    # Update ownership of app directory. Bind-mounted paths are explicitly
    # chowned again below after their directories are created.
    chown -R "$target_uid:$target_gid" /app 2>/dev/null || true
}

# Setup the user (only if running as root)
if [ "$(id -u)" = "0" ]; then
    setup_user "$PUID" "$PGID"

    # Set up directories with proper ownership
    echo "Setting up data directories..."
    mkdir -p /app/data/uploads /app/data/transcripts /app/whisperx-env
    chown -R "$PUID:$PGID" /app/data /app/whisperx-env

    echo "=== Setup Complete ==="
    echo "Switching to user appuser (UID=$(id -u appuser), GID=$(id -g appuser)) and starting application..."

    # Switch to the appuser and execute the command
    exec gosu appuser "$@"
else
    echo "Running as non-root user UID=$(id -u), GID=$(id -g)"

    # Just ensure directories exist
    mkdir -p /app/data/uploads /app/data/transcripts /app/whisperx-env 2>/dev/null || true

    echo "=== Setup Complete ==="
    echo "Starting Scriberr application..."

    # Execute directly
    exec "$@"
fi
