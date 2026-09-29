param(
    [string]$Target = "adb",
    [string]$Devices = "HTP0:0",
    [string]$LogDir = ".\iq2s-device-test"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $LogDir | Out-Null

function Invoke-Iq2sCase {
    param(
        [string]$Name,
        [string]$Mode,
        [string]$Params,
        [int]$MmSelect = -1,
        [switch]$Profile
    )

    $runArgs = @(
        ".\scripts\snapdragon\run.py",
        "--target", $Target,
        "--devices", $Devices
    )

    if ($MmSelect -ge 0) {
        $runArgs += @("--hex-mm-select", "$MmSelect")
    }
    if ($Profile) {
        $runArgs += @("--hex-profile", "1")
    }

    $runArgs += @(
        "--",
        "test-backend-ops",
        $Mode,
        "-b", $Devices,
        "-o", "MUL_MAT",
        "-p", $Params
    )

    $log = Join-Path $LogDir "$Name.txt"
    Write-Host "== $Name =="
    & python @runArgs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        throw "IQ2_S device test '$Name' failed with exit code $LASTEXITCODE. See $log"
    }
}

$baseDecode   = '^type_a=iq2_s,type_b=f32,m=16,n=1,k=256,bs=\[1,1\],nr=\[1,1\]'
$basePrefill  = '^type_a=iq2_s,type_b=f32,m=16,n=10,k=256,bs=\[1,1\],nr=\[1,1\]'
$kDecode      = '^type_a=iq2_s,type_b=f32,m=16,n=1,k=(512|1024|4096|8192),bs=\[1,1\],nr=\[1,1\]'
$kPrefill     = '^type_a=iq2_s,type_b=f32,m=16,n=10,k=(512|1024|4096|8192),bs=\[1,1\],nr=\[1,1\]'
$rowDecode    = '^type_a=iq2_s,type_b=f32,m=(31|32|33|63|64|65),n=1,k=1024,bs=\[1,1\],nr=\[1,1\]'
$rowPrefill   = '^type_a=iq2_s,type_b=f32,m=(31|32|33|63|64|65),n=10,k=1024,bs=\[1,1\],nr=\[1,1\]'
$batchDecode  = '^type_a=iq2_s,type_b=f32,m=33,n=1,k=1024,bs=\[2,3\],nr=\[1,1\]'
$batchPrefill = '^type_a=iq2_s,type_b=f32,m=33,n=10,k=1024,bs=\[2,3\],nr=\[1,1\]'
$largeHmx     = '^type_a=iq2_s,type_b=f32,m=64,n=32,k=4096,bs=\[2,1\],nr=\[1,1\]'

Invoke-Iq2sCase -Name "00-support-base" -Mode "support" -Params 'type_a=iq2_s,type_b=f32,m=16,n=(1|10),k=256'

Invoke-Iq2sCase -Name "01-hvx-base"        -Mode "test" -Params $baseDecode   -MmSelect 1 -Profile
Invoke-Iq2sCase -Name "02-hmx-base"        -Mode "test" -Params $basePrefill  -MmSelect 2 -Profile
Invoke-Iq2sCase -Name "03-hvx-larger-k"    -Mode "test" -Params $kDecode      -MmSelect 1
Invoke-Iq2sCase -Name "04-hmx-larger-k"    -Mode "test" -Params $kPrefill     -MmSelect 2
Invoke-Iq2sCase -Name "05-hvx-row-boundary" -Mode "test" -Params $rowDecode    -MmSelect 1
Invoke-Iq2sCase -Name "06-hmx-row-boundary" -Mode "test" -Params $rowPrefill   -MmSelect 2
Invoke-Iq2sCase -Name "07-hvx-batched"     -Mode "test" -Params $batchDecode  -MmSelect 1
Invoke-Iq2sCase -Name "08-hmx-batched"     -Mode "test" -Params $batchPrefill -MmSelect 2
Invoke-Iq2sCase -Name "09-hmx-large-batch" -Mode "test" -Params $largeHmx     -MmSelect 2 -Profile

Write-Host "All IQ2_S Hexagon device tests passed."
