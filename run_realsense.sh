#!/bin/bash
#
# This script launches the ORB-SLAM3 stereo example for the RealSense D435i.
#

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Set the root directory for your project
PROJECT_ROOT="/workspace/ORB_SLAM3"

# Set the paths to your files
EXECUTABLE="${PROJECT_ROOT}/Examples/Stereo/stereo_realsense_D435i"
VOCAB_FILE="${PROJECT_ROOT}/Vocabulary/ORBvoc.txt"
SETTINGS_FILE="${PROJECT_ROOT}/realsense_d435i.yaml"

# --- Pre-flight Checks ---

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "❌ ERROR: Executable not found at $EXECUTABLE"
    echo "Please compile the project first. (e.g., run 'make' in the 'build' directory)"
    exit 1
fi

# Check if vocabulary file exists
if [ ! -f "$VOCAB_FILE" ]; then
    echo "❌ ERROR: Vocabulary file not found at $VOCAB_FILE"
    exit 1
fi

# Check if settings file exists
if [ ! -f "$SETTINGS_FILE" ]; then
    echo "❌ ERROR: Settings file not found at $SETTINGS_FILE"
    exit 1
fi

# --- Run ---

# Navigate to the project root directory. This is important
# as the program may expect to be run from here.
cd "$PROJECT_ROOT"

echo "🚀 Launching ORB-SLAM3 RealSense Stereo..."
echo "------------------------------------------"

# Run the command with all arguments
"$EXECUTABLE" "$VOCAB_FILE" "$SETTINGS_FILE"

echo "------------------------------------------"
echo "✅ ORB-SLAM3 session finished."
