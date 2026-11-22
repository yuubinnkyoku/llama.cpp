# Build the two ARM CPU backends for phase 60.
# Usage: pwsh -File p60_build_arm.ps1 [-Which neon|scalar|armtest|all]
param(
    [string]$Which = 'all'
)

$ErrorActionPreference = 'Stop'
$repo = 'D:\ghq\github.com\yuubinnkyoku\llama.cpp-c51'
Set-Location $repo

$ndkRoot  = 'D:/android-sdk/ndk/29.0.14206865'
$toolchain = "$ndkRoot/build/cmake/android.toolchain.cmake"
$ninja    = (Get-Command ninja).Source

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
    param([string]$Dir, [string]$Arch, [string]$Target, [bool]$Tests)

    if (Test-Path $Dir) { Remove-Item $Dir -Recurse -Force }

    # this fork has no ggml/tests, so the tests come from LLAMA_BUILD_TESTS
    $args = @('-B', $Dir) + $common + @(
        "-DGGML_CPU_ARM_ARCH=$Arch",
        "-DLLAMA_BUILD_TESTS=$(if ($Tests) { 'ON' } else { 'OFF' })"
    )
    Write-Host "=== configure $Dir (arch=$Arch tests=$Tests)"
    & cmake @args
    if ($LASTEXITCODE -ne 0) { throw "configure failed for $Dir" }

    Write-Host "=== build $Dir target=$Target"
    & cmake --build $Dir --target $Target -- -j 16
    if ($LASTEXITCODE -ne 0) { throw "build failed for $Dir" }
}

if ($Which -eq 'neon' -or $Which -eq 'all') {
    Build-One -Dir 'build/c60-neon' -Arch 'armv8.6-a+dotprod+i8mm' -Target 'ggml-cpu' -Tests $false
}
if ($Which -eq 'scalar' -or $Which -eq 'all') {
    Build-One -Dir 'build/c60-scalar' -Arch 'armv8-a' -Target 'ggml-cpu' -Tests $false
}
if ($Which -eq 'armtest' -or $Which -eq 'all') {
    Build-One -Dir 'build/c60-armtest' -Arch 'armv8.6-a+dotprod+i8mm' -Target 'test-tiled-mulmat' -Tests $true
}

Write-Host '=== sha256'
foreach ($f in @('build/c60-neon/bin/libggml-cpu.so', 'build/c60-scalar/bin/libggml-cpu.so', 'build/c60-armtest/bin/test-tiled-mulmat')) {
    if (Test-Path $f) {
        $h = (Get-FileHash $f -Algorithm SHA256).Hash.ToLower()
        Write-Host "$h  $f"
    }
}
