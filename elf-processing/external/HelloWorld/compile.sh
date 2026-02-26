#!/bin/bash

# Remove the existing executable if it exists
if [ -f hello_world.elf ]; then
	rm hello_world.elf
fi

gcc hello_world.c -o hello_world.elf -O0 -fno-inline
cp hello_world.elf ../../data/hello_world.elf
