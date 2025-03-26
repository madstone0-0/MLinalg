#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="build"

# Usage function prints help message
usage() {
	echo "Usage: $0 [debug|release|profile|asm|all]"
	exit 1
}

# Run Doxygen and log output; exit if it fails.
build_doxygen() {
	local out_dir="$1"
	echo "Running Doxygen..."
	if ! doxygen Doxyfile &>"$out_dir/doxygen.log"; then
		echo "Doxygen failed. Check $out_dir/doxygen.log for details."
		exit 1
	fi
}

# Configure and build with CMake and Ninja.
# Arguments: build directory, build type, compiler
build_with_cmake() {
	local build_dir="$1"
	local build_type="$2"
	local compiler="$3"

	echo "Configuring and building ${build_type} build in ${build_dir} using ${compiler}..."
	mkdir -p "$build_dir"

	if [ $build_type == "Release" ]; then
		build_doxygen "$build_dir"
	fi

	if ! CXX="$compiler" cmake -G Ninja -S . -B "$build_dir" -DCMAKE_BUILD_TYPE="$build_type"; then
		echo "CMake configuration failed for ${build_dir}."
		exit 1
	fi

	if ! ninja -C "$build_dir" &>"$build_dir/build.log"; then
		echo "Ninja build failed for ${build_dir}. See $build_dir/build.log for details."
		exit 1
	fi

	# compdb --use-arguments -p build/ list >compile_commands.json
	echo "${build_type} build completed successfully in ${build_dir}."
}

build_debug() {
	build_with_cmake "$OUT_DIR" "Debug" "clang++"
}

build_release() {
	local release_dir="${OUT_DIR}-release"
	build_with_cmake "$release_dir" "Release" "g++"
}

build_profile() {
	local profile_dir="${OUT_DIR}-profile"
	build_with_cmake "$profile_dir" "Profile" "g++"
}

build_asm() {
	local profile_dir="${OUT_DIR}-asm"
	build_with_cmake "$profile_dir" "Asm" "g++"
}

# Determine build mode from first argument (default is debug)
if [ $# -eq 0 ]; then
	MODE="debug"
else
	MODE="$1"
fi

case "$MODE" in
debug)
	build_debug
	;;
release)
	build_release
	;;
profile)
	build_profile
	;;
asm)
	build_asm
	;;
all)
	echo "Building all configurations concurrently..."
	build_debug &
	build_release &
	build_profile &
	build_asm &
	wait
	;;
*)
	usage
	;;
esac
