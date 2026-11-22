# Phase 61 - build the ARM CPU backends to match the device's existing libraries.
#
# The device carries libggml-base.so d950aa96 (7,802,768 B) which has NO libomp
# dependency, so it was built with GGML_OPENMP=OFF. Every libggml-cpu.so built
# with GGML_OPENMP=ON links libomp and therefore runs against a different
# ggml-base than the one on the device. That mismatch alone moves the replay
# hash, so the A/B has to be done with a matched set.
#
# This script builds the whole set (base + cpu) with GGML_OPENMP=OFF so the
# device can be given a consistent trio.
#
# Usage: pwsh -File p61_build_matched.ps1 [-Which neon|scalar|all]
param(
    [string]$Which = 'all'
)

$ErrorActionPreference = 'Stop'
$repo = 'D:\ghq\github.com\yuubinnkyoku\llama.cpp-c51'
Set-Location $repo

$ndkRoot   = 'D:/android-sdk/ndk/29.0.14206865'
$toolchain = "$ndkRoot/build/cmake/android.toolchain.cmake"
$ninja     = (Get-Command ninja).Source

if (-not (Test-Path $toolchain)) { throw "toolchain not found: $toolchain" }

$common = @(
    '-G', 'Ninja',
    "-DCMAKE_MAKE_PROGRAM=$ninja",
    "-DCMAKE_TOOLCHAIN_FILE=$toolchain",
    '-DANDROID_ABI=arm64-v8a',
    '-DANDROID_PLATFORM=android-28',
    '-DCMAKE_BUILD_TYPE=Release',
    '-DBUILD_SHARED_LIBS=ON',
    '-DGGML_HEXAGON=OFF',
    '-DGGML_OPENMP=OFF',
    '-DGGML_LLAMAFILE=ON',
    '-DGGML_CPU_REPACK=ON',
    '-DGGML_NATIVE=OFF',
    '-DGGML_BUILD_EXAMPLES=OFF',
    '-DGGML_BUILD_TOOLS=OFF'
)

function Build-One {
    param([string]$Dir, [string]$Arch)

    if (Test-Path $Dir) { Remove-Item $Dir -Recurse -Force }

    $args = @('-B', $Dir) + $common + @("-DGGML_CPU_ARM_ARCH=$Arch")
    Write-Host "=== configure $Dir (arch=$Arch)"
    & cmake @args
    if ($LASTEXITCODE -ne 0) { throw "configure failed for $Dir" }

    Write-Host "=== build $Dir"
    & cmake --build $Dir --target ggml-cpu -- -j 16
    if ($LASTEXITCODE -ne 0) { throw "build failed for $Dir" }
}

if ($Which -eq 'neon' -or $Which -eq 'all') {
    Build-One -Dir 'build/c61m-neon' -Arch 'armv8.6-a+dotprod+i8mm'
}
if ($Which -eq 'scalar' -or $Which -eq 'all') {
    Build-One -Dir 'build/c61m-scalar' -Arch 'armv8-a'
}

Write-Host '=== sha256'
foreach ($d in @('build/c61m-neon', 'build/c61m-scalar')) {
    foreach ($n in @('libggml-cpu.so', 'libggml-base.so')) {
        $f = "$d/bin/$n"
        if (Test-Path $f) {
            $h = (Get-FileHash $f -Algorithm SHA256).Hash.ToLower()
            $len = (Get-Item $f).Length
            Write-Host "$h  $len  $f"
        }
    }
}
