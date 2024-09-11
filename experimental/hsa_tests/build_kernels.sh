~/src/iree-build/llvm-project/bin/clang \
  -x cl -Xclang -cl-std=CL2.0 \
  -target amdgcn-amd-amdhsa -march=gfx1100 \
  -cl-no-stdinc \
  -nogpulib \
  -fgpu-rdc \
  -fno-short-wchar \
  -fno-ident \
  -Xclang -finclude-default-header \
  -fvisibility=hidden \
  -O3 \
  kernels.cl \
  -c -emit-llvm -o kernels_cl.bc

~/src/iree-build/llvm-project/bin/llvm-link \
  -internalize \
  -only-needed \
  kernels_cl.bc \
  /opt/rocm/lib/llvm/lib/clang/17/lib/amdgcn/bitcode/ockl.bc \
  -o kernels_cl_linked.bc
  # ~/src/iree-build/tools/iree_platform_libs/rocm/ockl.bc \

~/src/iree-build/llvm-project/bin/lld \
  -flavor gnu \
  -m elf64_amdgpu \
  --build-id=none \
  --no-undefined \
  -shared \
  -plugin-opt=mcpu=gfx1100 \
  -plugin-opt=O3 \
  --lto-CGO3 \
  --no-whole-archive \
  -o kernels_cl.elf \
  kernels_cl_linked.bc
  # -save-temps \

~/src/iree-build/llvm-project/bin/llvm-readelf \
  kernels_cl.elf --all \
  >kernels_cl.txt

rm kernels_cl_linked.bc

# ~/src/iree-build/llvm-project/bin/clang \
#   -x hip \
#   --offload-device-only \
#   --offload-arch=gfx1100 \
#   -fuse-cuid=none \
#   -nogpulib \
#   -fgpu-rdc \
#   -fvisibility=hidden \
#   -O3 \
#   kernels.cpp \
#   -c -emit-llvm -o kernels_hip.bc

# ~/src/iree-build/llvm-project/bin/llvm-link \
#   -internalize \
#   -only-needed \
#   kernels_hip.bc \
#   /opt/rocm/lib/llvm/lib/clang/17/lib/amdgcn/bitcode/ockl.bc \
#   -o kernels_hip_linked.bc
#   # ~/src/iree-build/tools/iree_platform_libs/rocm/ockl.bc \

# ~/src/iree-build/llvm-project/bin/lld \
#   -flavor gnu \
#   -m elf64_amdgpu \
#   --build-id=none \
#   --no-undefined \
#   -shared \
#   -plugin-opt=mcpu=gfx1100 \
#   -plugin-opt=O3 \
#   --lto-CGO3 \
#   --no-whole-archive \
#   -o kernels_hip.elf \
#   kernels_hip_linked.bc
#   # -save-temps \

# ~/src/iree-build/llvm-project/bin/llvm-readelf \
#   kernels_hip.elf --all \
#   >kernels_hip.txt

# rm kernels_hip_linked.bc
