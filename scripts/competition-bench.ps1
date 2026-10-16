param(
    [string]$OptBin = "target\release\aet-opt.exe",
    [string]$MainBin = "target\release\aet-main.exe",
    [string]$Bzip2Bin = "C:\Program Files\Git\usr\bin\bzip2.exe",
    [string]$GzipBin = "C:\Program Files\Git\usr\bin\gzip.exe",
    [string]$SilesiaDir = "silesia"
)

$ErrorActionPreference = "Continue"

$files = Get-ChildItem -Path $SilesiaDir -File
Write-Host "=========================================================================================="
Write-Host "Benchmark: AET (Optimized vs Main) vs Competition (bzip2 -9, gzip -9)"
Write-Host "Corpus: Silesia ($($files.Count) files)"
Write-Host "=========================================================================================="
Write-Host ""
Write-Host ("{0,-10} {1,7} | {2,8} {3,7} {4,7} | {5,8} {6,7} {7,7} | {8,8} {9,7} | {10,8} {11,7}" -f `
    "File", "Size", `
    "OptRatio", "OptC", "OptD", `
    "MainRatio", "MainC", "MainD", `
    "bz2Ratio", "bz2C", `
    "gzRatio", "gzC")
Write-Host ("-" * 115)

$tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("comp-bench-" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tempDir | Out-Null

$totOrig = 0
$totOpt = 0
$totMain = 0
$totBz2 = 0
$totGz = 0

try {
    foreach ($file in $files) {
        $filePath = $file.FullName
        $fileName = $file.Name
        $fileSize = $file.Length
        $totOrig += $fileSize

        # --- Aether Opt ---
        $arcOpt = Join-Path $tempDir ($fileName + "-opt.aet")
        $extOpt = Join-Path $tempDir ($fileName + "-opt-out")
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        & $OptBin compress $filePath --output $arcOpt --predictor ssm --profile archival --force 2>$null
        $optCTime = [Math]::Max($sw.Elapsed.TotalSeconds, 0.001)
        $optSize = (Get-Item $arcOpt).Length
        $totOpt += $optSize
        $optRatio = ($optSize / $fileSize) * 100.0

        $sw.Restart()
        & $OptBin extract $arcOpt --output $extOpt 2>$null
        $optDTime = [Math]::Max($sw.Elapsed.TotalSeconds, 0.001)
        Remove-Item -Path $arcOpt -Force -ErrorAction SilentlyContinue
        Remove-Item -Path $extOpt -Recurse -Force -ErrorAction SilentlyContinue

        # --- Aether Main ---
        $arcMain = Join-Path $tempDir ($fileName + "-main.aet")
        $extMain = Join-Path $tempDir ($fileName + "-main-out")
        $sw.Restart()
        & $MainBin compress $filePath --output $arcMain --predictor ssm --profile archival --force 2>$null
        $mainCTime = [Math]::Max($sw.Elapsed.TotalSeconds, 0.001)
        $mainSize = (Get-Item $arcMain).Length
        $totMain += $mainSize
        $mainRatio = ($mainSize / $fileSize) * 100.0

        $sw.Restart()
        & $MainBin extract $arcMain --output $extMain 2>$null
        $mainDTime = [Math]::Max($sw.Elapsed.TotalSeconds, 0.001)
        Remove-Item -Path $arcMain -Force -ErrorAction SilentlyContinue
        Remove-Item -Path $extMain -Recurse -Force -ErrorAction SilentlyContinue

        # --- bzip2 -9 ---
        $arcBz2 = Join-Path $tempDir ($fileName + ".bz2")
        $sw.Restart()
        & $Bzip2Bin -k -9 -f -c $filePath > $arcBz2
        $bz2CTime = [Math]::Max($sw.Elapsed.TotalSeconds, 0.001)
        $bz2Size = (Get-Item $arcBz2).Length
        $totBz2 += $bz2Size
        $bz2Ratio = ($bz2Size / $fileSize) * 100.0
        Remove-Item -Path $arcBz2 -Force -ErrorAction SilentlyContinue

        # --- gzip -9 ---
        $arcGz = Join-Path $tempDir ($fileName + ".gz")
        $sw.Restart()
        & $GzipBin -k -9 -f -c $filePath > $arcGz
        $gzCTime = [Math]::Max($sw.Elapsed.TotalSeconds, 0.001)
        $gzSize = (Get-Item $arcGz).Length
        $totGz += $gzSize
        $gzRatio = ($gzSize / $fileSize) * 100.0
        Remove-Item -Path $arcGz -Force -ErrorAction SilentlyContinue

        $mb = $fileSize / 1MB
        $optCSpeed = $mb / $optCTime
        $optDSpeed = $mb / $optDTime
        $mainCSpeed = $mb / $mainCTime
        $mainDSpeed = $mb / $mainDTime
        $bz2CSpeed = $mb / $bz2CTime
        $gzCSpeed = $mb / $gzCTime

        Write-Host ("{0,-10} {1,6:N1}M | {2,7:N2}% {3,6:N1}M {4,6:N1}M | {5,7:N2}% {6,6:N1}M {7,6:N1}M | {8,7:N2}% {9,6:N1}M | {10,7:N2}% {11,6:N1}M" -f `
            $fileName, $mb, `
            $optRatio, $optCSpeed, $optDSpeed, `
            $mainRatio, $mainCSpeed, $mainDSpeed, `
            $bz2Ratio, $bz2CSpeed, `
            $gzRatio, $gzCSpeed)
    }

    Write-Host ("-" * 115)
    $totMb = $totOrig / 1MB
    Write-Host ("{0,-10} {1,6:N1}M | {2,7:N2}%                 | {3,7:N2}%                 | {4,7:N2}%         | {5,7:N2}%" -f `
        "TOTAL", $totMb, `
        (($totOpt / $totOrig) * 100.0), `
        (($totMain / $totOrig) * 100.0), `
        (($totBz2 / $totOrig) * 100.0), `
        (($totGz / $totOrig) * 100.0))
}
finally {
    Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
}
