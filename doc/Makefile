CURRENT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
ROOT ?= $(realpath $(CURRENT_DIR)/..)

INCLUDE ?= $(CURRENT_DIR)/include

all: $(ROOT)/README.md

$(ROOT)/README.md: README.in.md .version release.yaml
	$(INCLUDE) "$<" "$@"

.PHONY: all
