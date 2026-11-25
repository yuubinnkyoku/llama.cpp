# Phase 64 - rebuild the standard CPU backend WITH dotprod and i8mm.
#
# Every build from Phase 62 onward used GGML_CPU_ARM_ARCH=armv8 instead of
# armv8.6-a+dotprod+i8mm: the -D value was mangled on the way through the shell,
# so CMake fell back to the plain armv8 default. That silently removed SDOT from
# libggml-cpu.so, and the tiled microkernel needs SDOT. With dotprod the tiled
# path runs at 36.0 s for a 128-token prefill; without it the kernel falls back
# to the scalar microtile and the same run takes 114 s.
#
# A script is used instead of an inline command specifically because the inline
# form is what silently dropped the flag.
param(
    [string]$Dir  = 'build/c64-dot',
    [string]$Arch = 'armv8.6-a+dotprod+i8mm'
)

$ErrorActionPreference = 'Stop'
Set-Location 'D:\ghq\github.com\yuubinnkyoku\llama.cpp-c51'

$ndkRoot   = 'D:/android-sdk/ndk/29.0.14206865'
$toolchain = "$ndkRoot/build/cmake/android.toolchain.cmake"
$ninja     = (Get-Command ninja).Source

if (Test-Path $Dir) { Remove-Item $Dir -Recurse -Force }

& cmake -B $Dir -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" "-DCMAKE_TOOLCHAIN_FILE=$toolchain" `
    '-DANDROID_ABI=arm64-v8a' '-DANDROID_PLATFORM=android-28' '-DCMAKE_BUILD_TYPE=Release' `
    '-DBUILD_SHARED_LIBS=ON' '-DGGML_HEXAGON=OFF' '-DGGML_OPENMP=OFF' '-DGGML_LLAMAFILE=ON' `
    '-DGGML_CPU_REPACK=ON' '-DGGML_NATIVE=OFF' '-DGGML_BUILD_EXAMPLES=OFF' '-DGGML_BUILD_TOOLS=OFF' `
    "-DGGML_CPU_ARM_ARCH=$Arch"
if ($LASTEXITCODE -ne 0) { throw 'configure failed' }

& cmake --build $Dir --target ggml-cpu -- -j 16
if ($LASTEXITCODE -ne 0) { throw 'build failed' }

$cache = Get-Content "$Dir/CMakeCache.txt" | Where-Object { $_ -match '^GGML_CPU_ARM_ARCH:' }
Write-Host "=== cache check: $cache"
$f = "$Dir/bin/libggml-cpu.so"
$h = (Get-FileHash $f -Algorithm SHA256).Hash.ToLower()
$len = (Get-Item $f).Length
Write-Host "=== $h  $len  $f"

# Assert the flag actually took. The arch option was silently dropped for every
# build from Phase 62 to Phase 64, which cost the tiled path 3.5x, so check the
# cache and the instruction count instead of trusting the command line.
if ($cache -notmatch 'dotprod') { throw "GGML_CPU_ARM_ARCH did not take: $cache" }

$objdump = 'D:\android-sdk\ndk\29.0.14206865\toolchains\llvm\prebuilt\windows-x86_64\bin\llvm-objdump.exe'
$sdot = (& $objdump -d --no-show-raw-insn $f 2>$null | Select-String -Pattern '\bsdot\b' | Measure-Object).Count
Write-Host "=== sdot count: $sdot"
if ($sdot -eq 0) { throw 'no sdot in libggml-cpu.so: dotprod is not in the binary' }

