# run_reference_or.ps1 — launch the REFERENCE (Fortran) OpenRadioss starter /
# engine with the full runtime env. Distilled from the machine's verified
# launch environment; do NOT rediscover DLL paths by trial and error.
# Usage:
#   powershell -File tools\run_reference_or.ps1 -Deck <path\model_0000.rad>          # starter (.k also works)
#   powershell -File tools\run_reference_or.ps1 -Deck <path\model_0001.rad>          # engine (mpiexec when -Np > 1)
#   powershell -File tools\run_reference_or.ps1 -Deck <path\model_0000.rad> -Both    # starter, then engine if clean
#   -Np <n>  SPMD domains (default 1)   -Nt <n>  OpenMP threads (default 6)
# Exit code: the solver's own exit code (0 = clean; run summary printed either way).
param(
    [Parameter(Mandatory = $true)][string]$Deck,
    [int]$Np = 1,
    [int]$Nt = 6,
    [switch]$Both
)

$ErrorActionPreference = "Stop"
$OR     = "C:\OpenRadioss"
$ONEAPI = "C:\Program Files (x86)\Intel\oneAPI"

# Runtime env — every entry is load-bearing:
#   intelOneAPI_runtime\win64 : libiomp5md.dll (starter+engine die 0xC0000135 without it)
#   hm_reader\win64           : native LS-DYNA .k reading
#   h3d\lib\win64             : H3D output
#   mpi\latest\bin(+libfabric): impi.dll — engine_win64.exe is the Intel-MPI build and
#                               needs it even at -Np 1 (MPI singleton init)
$env:PATH = "$OR\extlib\intelOneAPI_runtime\win64;$OR\extlib\hm_reader\win64;" +
            "$OR\extlib\h3d\lib\win64;$ONEAPI\mpi\latest\bin;" +
            "$ONEAPI\mpi\latest\libfabric\bin;" + $env:PATH
$env:RAD_CFG_PATH = "$OR\hm_cfg_files"
# Hybrid P/E-core CPU: park spinning threads or the engine livelocks mid-iteration
# under desktop load.
$env:KMP_BLOCKTIME    = "0"
$env:OMP_WAIT_POLICY  = "PASSIVE"

$deckItem = Get-Item $Deck
Set-Location $deckItem.DirectoryName
$name = $deckItem.Name

$isEngine = $name -match "_[0-9]{3}[1-9]\.rad$"   # _0001.rad and later restarts
if ($isEngine -and $Both) { throw "-Both expects the starter deck (_0000.rad)" }

# NB: solver stdout is piped to Out-Host inside the functions — a PS function
# returns EVERYTHING left in its pipeline, so uncaptured stdout would pollute
# the returned exit code.
function Invoke-Starter([string]$file) {
    Write-Host "== starter: $file (np=$Np)"
    & "$OR\exec\starter_win64.exe" -i $file -np $Np -nt $Nt | Out-Host
    $code = $LASTEXITCODE
    $out = $file -replace "_0000\.rad$", "_0000.out" -replace "\.k$", "_0000.out"
    if (Test-Path $out) {
        Select-String -Path $out -Pattern "ERROR\(S\)|WARNING\(S\)|TERMINATION" |
            Select-Object -Last 4 | ForEach-Object { Write-Host $_.Line.Trim() }
    }
    return $code
}

function Invoke-Engine([string]$file) {
    Write-Host "== engine: $file (np=$Np, nt=$Nt)"
    if ($Np -gt 1) {
        & "$ONEAPI\mpi\latest\bin\mpiexec.exe" -np $Np "$OR\exec\engine_win64.exe" -i $file -nt $Nt | Out-Host
    } else {
        & "$OR\exec\engine_win64.exe" -i $file -nt $Nt | Out-Host
    }
    $code = $LASTEXITCODE
    $out = $file -replace "\.rad$", ".out"
    if (Test-Path $out) {
        Select-String -Path $out -Pattern "TERMINATION|ERROR" |
            Select-Object -Last 3 | ForEach-Object { Write-Host $_.Line.Trim() }
    }
    return $code
}

if ($isEngine) {
    $code = Invoke-Engine $name
} else {
    $code = Invoke-Starter $name
    if ($Both -and $code -eq 0) {
        $engineDeck = $name -replace "_0000\.rad$", "_0001.rad"
        if (Test-Path $engineDeck) { $code = Invoke-Engine $engineDeck }
        else { Write-Host "no $engineDeck next to the starter deck - engine skipped" }
    }
}
Write-Host "== exit code: $code"
exit $code
