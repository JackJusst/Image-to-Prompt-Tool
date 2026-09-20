$ErrorActionPreference = 'Stop'

$Root = Join-Path $env:USERPROFILE 'Documents\JusstBildTool'
$Venv = Join-Path $Root '.venv'
$App = Join-Path $Root 'JusstBildTool.py'
$Tools = Join-Path $Root 'tools'
$Esr = Join-Path $Tools 'realesrgan'

New-Item -ItemType Directory -Force -Path $Root,$Tools | Out-Null
Write-Host '=== Jusst BildTool Setup ===' -ForegroundColor Cyan

$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Host 'Python 3.12 wird installiert...'
    winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'Python konnte nicht installiert oder gefunden werden.'
}

if (-not (Test-Path (Join-Path $Venv 'Scripts\python.exe'))) {
    py -3.12 -m venv $Venv
}

$Python = Join-Path $Venv 'Scripts\python.exe'
& $Python -m pip install --upgrade pip
& $Python -m pip install pillow opencv-python numpy

Write-Host 'Lade JusstBildTool.py...'
Invoke-WebRequest -UseBasicParsing -Uri 'https://raw.githubusercontent.com/JackJusst/Image-to-Prompt-Tool/main/downloads/JusstBildTool.py' -OutFile $App

if (-not (Test-Path (Join-Path $Esr 'realesrgan-ncnn-vulkan.exe'))) {
    Write-Host 'Lade Real-ESRGAN fuer KI-Upscaling...'
    $zip = Join-Path $Tools 'realesrgan.zip'
    try {
        Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip' -OutFile $zip
        New-Item -ItemType Directory -Force -Path $Esr | Out-Null
        Expand-Archive -Path $zip -DestinationPath $Esr -Force
        Remove-Item $zip -Force
        $exe = Get-ChildItem -Path $Esr -Filter 'realesrgan-ncnn-vulkan.exe' -Recurse | Select-Object -First 1
        if ($exe -and $exe.DirectoryName -ne $Esr) {
            Get-ChildItem $exe.DirectoryName | Move-Item -Destination $Esr -Force
        }
    } catch {
        Write-Host 'Real-ESRGAN konnte nicht geladen werden. Das Tool funktioniert trotzdem ohne KI-Upscaling.' -ForegroundColor Yellow
    }
}

$Start = @"
@echo off
cd /d "$Root"
"$Venv\Scripts\pythonw.exe" "$App"
"@
Set-Content -Path (Join-Path $Root 'START_JusstBildTool.cmd') -Value $Start -Encoding ASCII

$Desktop = [Environment]::GetFolderPath('Desktop')
$Shortcut = Join-Path $Desktop 'Jusst BildTool.lnk'
$Shell = New-Object -ComObject WScript.Shell
$Lnk = $Shell.CreateShortcut($Shortcut)
$Lnk.TargetPath = Join-Path $Root 'START_JusstBildTool.cmd'
$Lnk.WorkingDirectory = $Root
$Lnk.Save()

Write-Host ''
Write-Host 'FERTIG. Das Tool wird jetzt gestartet.' -ForegroundColor Green
Start-Process (Join-Path $Root 'START_JusstBildTool.cmd')
