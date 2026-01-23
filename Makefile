CC ?= cc
CFLAGS ?= -O2 -std=c11

INCLUDES = -Icsrc -Iassets -Itests
LIBS = -lm

.PHONY: test_conv0 test_c3 test_c3_debug test_sppf_gen test_sppf clean

test_conv0:
	$(CC) $(CFLAGS) csrc/test_conv0.c csrc/operations/conv2d.c csrc/operations/bn_silu.c $(INCLUDES) $(LIBS) -o tests/test_conv0
	./tests/test_conv0

test_c3:
	$(CC) $(CFLAGS) csrc/test_c3.c csrc/blocks/c3.c csrc/operations/conv2d.c csrc/operations/bn_silu.c csrc/operations/bottleneck.c csrc/operations/concat.c $(INCLUDES) $(LIBS) -o tests/test_c3
	./tests/test_c3

test_c3_debug:
	$(CC) $(CFLAGS) csrc/test_c3_debug.c csrc/operations/conv2d.c csrc/operations/bn_silu.c csrc/operations/bottleneck.c csrc/operations/concat.c $(INCLUDES) $(LIBS) -o tests/test_c3_debug
	./tests/test_c3_debug

test_sppf_gen:
	python tools/gen_sppf_test_vectors.py --pt assets/yolov5n.pt --out tests/test_vectors_sppf.h --h 8 --w 8

test_sppf:
	$(CC) $(CFLAGS) csrc/test_sppf.c csrc/blocks/sppf.c csrc/operations/conv2d.c csrc/operations/bn_silu.c csrc/operations/maxpool2d.c csrc/operations/concat.c $(INCLUDES) $(LIBS) -o tests/test_sppf
	./tests/test_sppf

clean:
	rm -f tests/test_conv0 tests/test_c3 tests/test_c3_debug tests/test_sppf

