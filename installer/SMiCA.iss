; Inno Setup script for SMiCA (Statistical Microstructure Characterisation & Analysis).
; Packages the PyInstaller onedir build in dist\SMiCA\ into a single Setup.exe:
; Start Menu entry, optional desktop shortcut, uninstaller, and the Microsoft
; VC++ Redistributable as a prerequisite (see [Run] below - the frozen build's
; main.py deliberately does NOT bundle its own copy of those runtime DLLs,
; since doing so crashes Qt6Core.dll on startup; the installer is what
; guarantees the target machine has a working copy instead).

#define MyAppName "SMiCA"
#define MyAppVersion "0.3.0"
#define MyAppPublisher "Hamed Amiri"
#define MyAppExeName "SMiCA.exe"

[Setup]
AppId={{A5008495-320B-439F-8A11-EAA6DAB42A42}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL=https://github.com/hamediut/SMiCA
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
; Explicitly show the "Select Destination Location" page rather than relying on
; Inno Setup's default behavior for it, so the user can always pick where SMiCA
; installs instead of it silently going to Program Files.
DisableDirPage=no
; Bundling the VC++ redistributable installer requires it to run elevated,
; so this installer always requests admin rights (the Inno Setup default).
OutputDir=output
OutputBaseFilename=SMiCA_Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "..\dist\SMiCA\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion
Source: "redist\vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Microsoft's own installer detects an existing up-to-date install and exits
; quickly as a no-op, so this always runs rather than trying to detect that
; ourselves - simpler and just as correct.
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "Installing Visual C++ Runtime (skipped automatically if already present)..."; Flags: waituntilterminated
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
