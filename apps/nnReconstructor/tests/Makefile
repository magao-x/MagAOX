# App-local CPU tests; SDK doubles do not validate the production GPU ABI.
CXX = clang++
CXXFLAGS = -std=c++20 -pthread -g -O1 -Wall -Wextra -Wno-unused-parameter
all: test
.build/nnReconstructor_test: nnReconstructor_test.cpp nnTestSupport.hpp ../nnReconstructor.hpp
	mkdir -p .build
	$(CXX) $(CXXFLAGS) -DNNRECONSTRUCTOR_TEST -DCATCH_CONFIG_NO_POSIX_SIGNALS $< -o $@
test: .build/nnReconstructor_test
	./.build/nnReconstructor_test
.build/nnReconstructor_optimized_test: nnReconstructor_test.cpp nnTestSupport.hpp ../nnReconstructor.hpp
	mkdir -p .build
	$(CXX) $(CXXFLAGS) -O3 -ffast-math -DNNRECONSTRUCTOR_TEST -DCATCH_CONFIG_NO_POSIX_SIGNALS $< -o $@
optimized: .build/nnReconstructor_optimized_test
	./.build/nnReconstructor_optimized_test
asan:
	$(MAKE) clean
	$(MAKE) CXXFLAGS='$(CXXFLAGS) -fsanitize=address,undefined -fno-omit-frame-pointer' test
tsan:
	$(MAKE) clean
	$(MAKE) CXXFLAGS='$(CXXFLAGS) -fsanitize=thread' test
clean:
	rm -rf .build
.PHONY: all test optimized asan tsan clean
