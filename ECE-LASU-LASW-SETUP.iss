; ECE-LASU-LASW.iss
; ---------------------------------------------------------
; Inno Setup script for Expert-Level Radiology AI Assistant
;
; Usage:
;   1. Build the PyInstaller output first:
;        pyinstaller ECE-LASU-LASW.spec
;      This produces: dist\ECE-LASU-LASW\  (the folder this script installs)
;   2. Open this file in Inno Setup Compiler (or run: iscc ECE-LASU-LASW.iss)
;   3. Output installer lands in: Output\ECE-LASU-LASW-Setup.exe
; ---------------------------------------------------------

#define MyAppName "ECE-LASU-LASW"
#define MyAppDisplayName "Expert-Level Radiology AI Assistant"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Lateef Akinyemi, Sebastine Waniko,"
#define MyAppExeName "ECE-LASU-LASW.exe"
; Path to the PyInstaller COLLECT output — adjust if your dist folder lives elsewhere
#define MyDistDir "dist\ECE-LASU-LASW"

[Setup]
AppId={B3F2A9C4-6D1E-4A7B-9F3C-1E2D3C4B5A6F}
AppName={#MyAppDisplayName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\{#MyAppName}
DefaultGroupName={#MyAppDisplayName}
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename={#MyAppName}-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
SetupIconFile={#MyDistDir}\Profile.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
; Installing under LocalAppData (not Program Files) avoids UAC prompts and
; write-permission issues, since the app writes to its own folder at runtime
; (ct_uploads, radiology_faiss_index, output cache, Ollama models, etc.)
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
; Recursively pull in the entire PyInstaller output folder — the .exe,
; every bundled DLL/dependency, TotalSegRunner, ollama, sdxl_model, pdf,
; fontawesome, and anything else COLLECT placed there.
Source: "{#MyDistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppDisplayName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppDisplayName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppDisplayName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppDisplayName} now"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up runtime-generated folders the app creates on first use, so a
; reinstall starts fresh rather than inheriting old scan/cache data.
Type: filesandordirs; Name: "{app}\ct_uploads"
Type: filesandordirs; Name: "{app}\radiology_faiss_index"
Type: filesandordirs; Name: "{app}\image_output_cache"