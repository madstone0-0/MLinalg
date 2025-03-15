#!/usr/bin/env bash

OUT_DIR="build"

build_release() {
	echo "Building Release"
	LOCAL_OUT_DIR="$OUT_DIR-release"
	mkdir -p "$LOCAL_OUT_DIR"
	doxygen Doxyfile &>"$LOCAL_OUT_DIR"/doxygen.log
	CXX=g++ cmake -G 'Ninja' -S . -B "$LOCAL_OUT_DIR" -DCMAKE_BUILD_TYPE=Release && ninja -C $LOCAL_OUT_DIR/ &>"$LOCAL_OUT_DIR".log
}

build_debug() {
	echo "Building Debug"
	mkdir -p "$OUT_DIR"
	doxygen Doxyfile &>./doxygen.log
	CXX=clang++ cmake -G 'Ninja' -S . -B "$OUT_DIR" -DCMAKE_BUILD_TYPE=Debug && ninja -C $OUT_DIR/ &>"$OUT_DIR".log
}

build_profile() {
	echo "Building Profile"
	LOCAL_OUT_DIR="$OUT_DIR-profile"
	mkdir -p "$LOCAL_OUT_DIR"
	CXX=g++ cmake -G 'Ninja' -S . -B "$LOCAL_OUT_DIR" -DCMAKE_BUILD_TYPE=Profile && ninja -C $LOCAL_OUT_DIR/ &>"$LOCAL_OUT_DIR".log
}

MODE="$1"

if [ -z "$MODE" ]; then
	MODE="debug"
fi

case $MODE in
"debug")
	build_debug
	;;
"release")
	build_release
	;;
"profile")
	build_profile
	;;
esac
