#!/bin/bash

# Get the directory of the bash script
script_dir=$(dirname "$0")

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <folder_path>"
  exit 1
fi

folder_path="$1"

# Check if the folder exists
if [ ! -d "$folder_path" ]; then
  echo "Folder '$folder_path' does not exist."
  exit 1
fi

# Find all .bag files in the folder and process each one
for bag_file in "$folder_path"/*.bag; do
#   if [ -f "$bag_file" ]; then
    echo "Processing $bag_file"
    python "$script_dir/readbag_instant_cot.py" "$bag_file"
#   fi
done
