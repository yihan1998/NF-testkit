SUBDIRS := ./bindings/c/

bindings:
	cd bindings/net && LIBRARY_PATH=../bindings/c cargo build && cd ../../

all: $(SUBDIRS) bindings
$(SUBDIRS):
	$(MAKE) -C $@

.PHONY: all $(SUBDIRS)