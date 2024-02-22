SUBDIRS	:= ./bindings/c/

all: $(SUBDIRS) bindings

$(SUBDIRS):
	@echo "Calling makefile in $@ ..."
	$(MAKE) -C $@

clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done

bindings:
	cd bindings/net && LIBRARY_PATH=../bindings/c cargo build && cd ../../

.PHONY: all $(SUBDIRS)
