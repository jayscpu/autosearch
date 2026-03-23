#!/bin/bash
# pod_download.sh — Download all 5 Bellevue intersections from Google Drive.
# Uses gdown with --remaining-ok so partial downloads can be resumed.
set -e

cd /workspace/autosearch/pod/data

echo "=== Downloading Bellevue Traffic Video Dataset ==="
echo "  5 intersections, ~101 hours total"
echo ""

# Google Drive folder IDs for each intersection
declare -A FOLDERS
FOLDERS[Bellevue_150th_Eastgate]="1cR1VwoAvEjFLRaUzeYph-bxx4LoM6pOH"
FOLDERS[Bellevue_150th_Newport]="1irB6XKu2iM3BSJ2AEYH4kJl9nfG9j-yy"
FOLDERS[Bellevue_150th_SE38th]="1IN6kwywddO3B3uHyC5S18vqf0KEWToJ_"
FOLDERS[Bellevue_Bellevue_NE8th]="17bn7l7Qm5s-r5DYoFQPhviFZ0jWY9qk5"
FOLDERS[Bellevue_116th_NE12th]="16coOR8PlNzvmUm1vsaYJVF_bAOQGySa8"

for name in "${!FOLDERS[@]}"; do
    fid="${FOLDERS[$name]}"
    if [ -d "$name" ] && [ "$(find "$name" -name '*.mp4' -size +1M | wc -l)" -gt 0 ]; then
        n_valid=$(find "$name" -name '*.mp4' -size +1M | wc -l)
        echo "  SKIP $name — $n_valid valid video files already exist"
        continue
    fi
    echo "  Downloading $name ..."
    gdown --folder "https://drive.google.com/drive/folders/$fid" \
        -O "$name" --remaining-ok || true
done

echo ""
echo "=== Checking for corrupted downloads ==="
# Files under 1MB are likely HTML error pages, not videos
bad_files=$(find . -name "*.mp4" -size -1M 2>/dev/null)
if [ -n "$bad_files" ]; then
    echo "  WARNING: These files are suspiciously small (likely quota-blocked):"
    echo "$bad_files"
    echo "  Delete them and re-run this script later to retry."
else
    echo "  All video files look valid."
fi

echo ""
echo "=== Download summary ==="
for name in "${!FOLDERS[@]}"; do
    if [ -d "$name" ]; then
        n=$(find "$name" -name '*.mp4' -size +1M | wc -l)
        size=$(du -sh "$name" 2>/dev/null | cut -f1)
        echo "  $name: $n videos, $size"
    else
        echo "  $name: NOT DOWNLOADED"
    fi
done

echo ""
echo "=== Done. Now run: python pod_pipeline.py ==="
