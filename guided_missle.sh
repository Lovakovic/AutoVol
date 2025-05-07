#!/usr/bin/env bash

# Script: guided-missle.sh
# Description: Concatenates specified text files into a single file, now with directory support
# Author: Enhanced by Claude
# Version: 1.0.2

set -euo pipefail  # Enable strict mode
IFS=$'\n\t'       # Set safe IFS

# Script metadata
readonly VERSION="1.0.2"
readonly SCRIPT_NAME=$(basename "${BASH_SOURCE[0]}")
readonly DEFAULT_OUTPUT="concatenated.txt"

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m' # No Color

# Default configuration
output_file="$DEFAULT_OUTPUT"
declare -a input_paths=()

# Non-text file extensions to skip (simplified matching like nuke.sh)
readonly SKIP_EXTENSIONS="png jpg jpeg gif svg ico webp bmp tiff ttf woff woff2 eot otf mp3 mp4 wav ogg webm avi mov zip tar gz rar 7z pdf doc docx xls xlsx ppt pptx map min.js min.css exe dll so dylib"

# Directories to exclude (common in Python proje cts)
readonly EXCLUDED_DIRS=".venv __pycache__"

# Help message
show_help() {
    cat << EOF
Usage: ${SCRIPT_NAME} [OPTIONS] PATH1 [PATH2 ...]

Concatenates specified text files or all text files in specified directories into a single file.

Options:
    --output=FILE   Output file name (default: ${DEFAULT_OUTPUT})
    -h, --help      Show this help message
    -v, --version   Show version information

Examples:
    ${SCRIPT_NAME} file1.txt file2.txt file3.txt
    ${SCRIPT_NAME} --output=combined.txt *.js
    ${SCRIPT_NAME} /path/to/directory
    ${SCRIPT_NAME} /path/to/file1.txt /path/to/directory /path/to/file2.txt

Notes:
    - The script automatically skips binary files and common non-text formats
    - Files are processed in the order they are specified
    - When processing directories, hidden files and node_modules, ${EXCLUDED_DIRS}, are skipped
EOF
}

# Version information
show_version() {
    echo "${SCRIPT_NAME} version ${VERSION}"
}

# Error handling
error() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

# Warning messages
warn() {
    echo -e "${YELLOW}Warning: $1${NC}" >&2
}

# Success messages
success() {
    echo -e "${GREEN}$1${NC}"
}

# Check if file should be skipped based on extension
should_skip_file() {
    local file="$1"
    local extension="${file##*.}"
    extension=$(echo "$extension" | tr '[:upper:]' '[:lower:]')

    [[ " $SKIP_EXTENSIONS " == *" $extension "* ]] && return 0
    return 1
}

# Check if a directory should be excluded
should_exclude_dir() {
    local dir="$1"
    for excluded in $EXCLUDED_DIRS; do
        if [[ "$dir" == *"/$excluded" || "$dir" == "$excluded" ]]; then
            return 0
        fi
    done
    return 1
}

# Process a single file
process_file() {
    local file="$1"
    local base_name=$(basename "$file")
    local dir_name=$(dirname "$file")

    # Skip conditions
    [[ "$base_name" == "$SCRIPT_NAME" ]] && return
    [[ "$base_name" == "$output_file" ]] && return
    [[ "$base_name" == .* ]] && return
    [[ "$dir_name" == *"/."* ]] && return
#    [[ "$file" == *"node_modules"* ]] && return

    # Skip additional excluded directories
    if should_exclude_dir "$dir_name"; then
        warn "Skipping excluded directory: $dir_name"
        return
    fi

    # Check if file is readable
    if [[ ! -r "$file" ]]; then
        error "File is not readable: $file"
    fi

    # Skip if it's a binary file
    if should_skip_file "$file"; then
        warn "Skipping binary file: $file"
        return
    fi

    {
        echo -e "\n// $file"
        cat "$file"
        echo -e "\n"
    } >> "$output_file"
}

# Process a directory
process_directory() {
    local dir="$1"

    # Check if directory exists and is readable
    if [[ ! -d "$dir" ]] || [[ ! -r "$dir" ]]; then
        error "Directory is not accessible: $dir"
    fi

    # Find and process all files in directory, excluding specified directories
    find "$dir" -type f \
        -not -path '*/.venv/*' \
        -not -path '*/__pycache__/*' \
        -not -path '*/\.*/*' \
        -not -path '*/node_modules/*' \
        -exec bash -c 'process_file "$0"' {} \;
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --output=*)
                output_file="${1#*=}"
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--version)
                show_version
                exit 0
                ;;
            -*)
                error "Unknown option: $1"
                ;;
            *)
                input_paths+=("$1")
                ;;
        esac
        shift
    done

    # Check if any input paths were specified
    if [ ${#input_paths[@]} -eq 0 ]; then
        error "No input paths specified. Use -h or --help for usage information."
    fi
}

# Main execution
main() {
    parse_arguments "$@"

    # Remove existing output file
    [[ -f "$output_file" ]] && rm "$output_file"

    # Export required variables and functions for subshell execution
    export output_file
    export SCRIPT_NAME
    export SKIP_EXTENSIONS
    export EXCLUDED_DIRS
    export -f should_skip_file
    export -f should_exclude_dir
    export -f process_file
    export -f warn
    export -f error

    # Process each input path
    for path in "${input_paths[@]}"; do
        if [[ -d "$path" ]]; then
            process_directory "$path"
        elif [[ -f "$path" ]]; then
            process_file "$path"
        else
            error "Path does not exist: $path"
        fi
    done

    success "Files have been concatenated into $output_file"
}

# Execute main function with all arguments
main "$@"
