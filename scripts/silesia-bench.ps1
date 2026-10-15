param(
    [string]$MainBin = "target\release\aet-main.exe",
    [string]$OptBin = "target\release\aet-opt.exe",
    [string]$SilesiaDir = "silesia"
)

$ErrorActionPreference = "Continue"

$files = Get-ChildItem -Path $SilesiaDir -File
Write-Host "Found $($files.Count) Silesia files for benchmark."
Write-Host ""
Write-Host ("{0,-10} {1,8} | {2,8} {3,8} {4,8} | {5,8} {6,8} {7,8} | {8,8}" -f "File", "Size", "MainRatio", "MainComp", "MainDec", "OptRatio", "OptComp", "OptDec", "RatioDiff")
Write-Host ("-" * 95)

$tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("silesia-bench-" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tempDir | Out-Null

try {
    foreach ($file in $files) {
        $filePath = $file.FullName
        $fileName = $file.Name
        $fileSize = $file.Length

        # --- Benchmark Main ---
        $arcMain = Join-Path $tempDir ($fileName + "-main.aet")
        $extMain = Join-Path $tempDir ($fileName + "-main-out")
        
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        & $MainBin compress $filePath --output $arcMain --predictor ssm --profile archival --force 2>$null
        $mainCompTime = $sw.Elapsed.TotalSeconds
        $mainArcSize = (Get-Item $arcMain).Length
        $mainRatio = ($mainArcSize / $fileSize) * 100.0

        $sw.Restart()
        & $MainBin extract $arcMain --output $extMain 2>$null
        $mainDecTime = $sw.Elapsed.TotalSeconds

        Remove-Item -Path $arcMain -Force -ErrorAction SilentlyContinue
        Remove-Item -Path $extMain -Recurse -Force -ErrorAction SilentlyContinue

        # --- Benchmark Opt ---
        $arcOpt = Join-Path $tempDir ($fileName + "-opt.aet")
        $extOpt = Join-Path $tempDir ($fileName + "-opt-out")

        $sw.Restart()
        & $OptBin compress $filePath --output $arcOpt --predictor ssm --profile archival --force 2>$null
        $optCompTime = $sw.Elapsed.TotalSeconds
        $optArcSize = (Get-Item $arcOpt).Length
        $optRatio = ($optArcSize / $fileSize) * 100.0

        $sw.Restart()
        & $OptBin extract $arcOpt --output $extOpt 2>$null
        $optDecTime = $sw.Elapsed.TotalSeconds

        Remove-Item -Path $arcOpt -Force -ErrorAction SilentlyContinue
        Remove-Item -Path $extOpt -Recurse -Force -ErrorAction SilentlyContinue

        $mainCompSpeed = ($fileSize / 1MB) / [Math]::Max($mainCompTime, 0.001)
        $mainDecSpeed = ($fileSize / 1MB) / [Math]::Max($mainDecTime, 0.001)
        $optCompSpeed = ($fileSize / 1MB) / [Math]::Max($optCompTime, 0.001)
        $optDecSpeed = ($fileSize / 1MB) / [Math]::Max($optDecTime, 0.001)
        $ratioDiff = $optRatio - $mainRatio

        Write-Host ("{0,-10} {1,9:N1}M | {2,7:N2}% {3,7:N1}M {4,7:N1}M | {5,7:N2}% {6,7:N1}M {7,7:N1}M | {8,7:N2}%" -f `
            $fileName, ($fileSize / 1MB), `
            $mainRatio, $mainCompSpeed, $mainDecSpeed, `
            $optRatio, $optCompSpeed, $optDecSpeed, `
            $ratioDiff)
    }
}
finally {
    Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
}
