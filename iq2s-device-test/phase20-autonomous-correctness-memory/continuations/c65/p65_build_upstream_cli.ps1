# Phase 65 - build upstream llama-cli with the Hexagon backend for the Bonsai test.
#
# Why upstream and not the fork: the fork has the Q2_0 type and the FWHT kernel,
# but no Prism Bonsai model support at all (0 PRISM_HADAMARD keys in
# src/llama-arch.h). Upstream master has it. PrismML's own preset says outright
# that without their unreleased fix the output is wrong on the Hexagon NPU, so
# the point here is to measure splits, speed and memory in that state, not to get
# a correct answer.
#
# Hexagon is ON so the splits number is comparable with the 262 measured for
# plain upstream on the 27B. dotprod is ON and checked, per the Phase 64 rule.
param(
    [string]$Tree = 'D:\ghq\github.com\yuubinnkyoku\llama-upstream-c63',
    [string]$Dir  = 'build\c65-upcli',
    [string]$Arch = 'armv8.6-a+dotprod+i8mm'
)

$ErrorActionPreference = 'Stop'
Set-Location $Tree

$sdk       = 'C:/Qualcomm/Hexagon_SDK/6.6.0.0'
$cmake     = "$sdk/tools/cmake-3.28.3-windows-x86_64/bin/cmake.exe"
$tools     = "$sdk/tools/HEXAGON_Tools/19.0.07"
$toolchain = 'D:/android-sdk/ndk/29.0.14206865/build/cmake/android.toolchain.cmake'
$ninja     = (Get-Command ninja).Source

if (-not (Test-Path $cmake)) { throw "SDK cmake missing: $cmake" }

& $cmake -B $Dir -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" "-DCMAKE_TOOLCHAIN_FILE=$toolchain" `
    '-DANDROID_ABI=arm64-v8a' '-DANDROID_PLATFORM=android-28' '-DCMAKE_BUILD_TYPE=Release' `
    '-DBUILD_SHARED_LIBS=ON' '-DGGML_NATIVE=OFF' '-DGGML_OPENMP=OFF' '-DGGML_CPU_REPACK=ON' `
    '-DGGML_HEXAGON=ON' "-DHEXAGON_SDK_ROOT=$sdk" "-DHEXAGON_TOOLS_ROOT=$tools" `
    '-DPREBUILT_LIB_DIR=toolv19_v81' "-DGGML_CPU_ARM_ARCH=$Arch" `
    '-DGGML_BUILD_EXAMPLES=OFF' '-DGGML_BUILD_TESTS=OFF'
if ($LASTEXITCODE -ne 0) { throw 'configure failed' }

& $cmake --build $Dir --target llama-cli -- -j 16
if ($LASTEXITCODE -ne 0) { throw 'build failed' }

$cache = (Get-Content "$Dir/CMakeCache.txt" | Where-Object { $_ -match '^GGML_CPU_ARM_ARCH:' })
Write-Host "=== cache: $cache"
if ($cache -notmatch 'dotprod') { throw "arch flag did not take: $cache" }

$objdump = 'D:\android-sdk\ndk\29.0.14206865\toolchains\llvm\prebuilt\windows-x86_64\bin\llvm-objdump.exe'
$cpu = Get-ChildItem "$Dir" -Recurse -Filter 'libggml-cpu.so' | Select-Object -First 1
if ($cpu) {
    $sdot = (& $objdump -d --no-show-raw-insn $cpu.FullName 2>$null | Select-String -Pattern '\bsdot\b' | Measure-Object).Count
    Write-Host "=== sdot in $($cpu.FullName): $sdot"
    if ($sdot -eq 0) { throw 'no sdot: dotprod missing' }
}

Write-Host '=== artifacts'
Get-ChildItem "$Dir\bin" -Filter '*.so' | ForEach-Object { "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(), $_.Name }
Get-ChildItem "$Dir\bin" -Filter 'llama-cli*' | ForEach-Object { "{0}  {1}  {2}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(), $_.Length, $_.Name }
