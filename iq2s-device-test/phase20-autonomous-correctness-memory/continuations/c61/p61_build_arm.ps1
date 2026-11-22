# Phase 61 - build the ARM CPU backends with the SAME options as the baseline
# (build/p53-android), so an A/B changes only the tiled kernel.
#
# The Phase 60 builds used GGML_OPENMP=OFF, which is not what the baseline
# 0a3db9c5 was built with. That difference alone moves the replay hash, so it
# has to be removed before any end-to-end comparison.
#
# Usage: pwsh -File p61_build_arm.ps1 [-Which neon|scalar|all]
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

# matches build/p53-android: GGML_OPENMP=ON, GGML_LLAMAFILE=ON, GGML_CPU_REPACK=ON,
# GGML_NATIVE=OFF, no GGML_CPU_ARM_ARCH (so the stock armv8-a code path)
$common = @(
    '-G', 'Ninja',
    "-DCMAKE_MAKE_PROGRAM=$ninja",
    "-DCMAKE_TOOLCHAIN_FILE=$toolchain",
    '-DANDROID_ABI=arm64-v8a',
    '-DANDROID_PLATFORM=android-28',
    '-DCMAKE_BUILD_TYPE=Release',
    '-DBUILD_SHARED_LIBS=ON',
    '-DGGML_HEXAGON=OFF',
    '-DGGML_OPENMP=ON',
    '-DGGML_LLAMAFILE=ON',
    '-DGGML_CPU_REPACK=ON',
    '-DGGML_NATIVE=OFF',
    '-DGGML_BUILD_EXAMPLES=OFF',
    '-DGGML_BUILD_TOOLS=OFF'
)

function Build-One {
    param([string]$Dir, [string]$Arch, [string]$Target)

    if (Test-Path $Dir) { Remove-Item $Dir -Recurse -Force }

    $args = @('-B', $Dir) + $common + @("-DGGML_CPU_ARM_ARCH=$Arch")
    Write-Host "=== configure $Dir (arch=$Arch)"
    & cmake @args
    if ($LASTEXITCODE -ne 0) { throw "configure failed for $Dir" }

    Write-Host "=== build $Dir target=$Target"
    & cmake --build $Dir --target $Target -- -j 16
    if ($LASTEXITCODE -ne 0) { throw "build failed for $Dir" }
}

if ($Which -eq 'neon' -or $Which -eq 'all') {
    Build-One -Dir 'build/c61-neon' -Arch 'armv8.6-a+dotprod+i8mm' -Target 'ggml-cpu'
}
if ($Which -eq 'scalar' -or $Which -eq 'all') {
    Build-One -Dir 'build/c61-scalar' -Arch 'armv8-a' -Target 'ggml-cpu'
}

Write-Host '=== sha256'
foreach ($f in @('build/c61-neon/bin/libggml-cpu.so', 'build/c61-scalar/bin/libggml-cpu.so')) {
    if (Test-Path $f) {
        $h = (Get-FileHash $f -Algorithm SHA256).Hash.ToLower()
        $len = (Get-Item $f).Length
        Write-Host "$h  $len  $f"
    }
}
