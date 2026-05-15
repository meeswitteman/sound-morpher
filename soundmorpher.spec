# PyInstaller spec for Sound Morpher
# Build with: pyinstaller soundmorpher.spec
# Requires: pip install pyinstaller

import sys
from pathlib import Path

block_cipher = None

ROOT = Path(SPECPATH)

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        (str(ROOT / "resources"), "resources"),
        (str(ROOT / "plugins"), "plugins"),
    ],
    hiddenimports=[
        # scipy sub-modules not auto-detected
        "scipy.signal",
        "scipy.signal._peak_finding",
        "scipy._lib.array_api_compat.numpy.fft",
        # librosa
        "librosa",
        "librosa.core",
        "librosa.effects",
        "librosa.feature",
        "librosa.filters",
        "librosa.util",
        "numba",
        "numba.core",
        # sounddevice / soundfile
        "sounddevice",
        "soundfile",
        "cffi",
        "_cffi_backend",
        # PySide6 extras
        "PySide6.QtSvg",
        "PySide6.QtSvgWidgets",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SoundMorpher",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # no console window on Windows
    icon=str(ROOT / "resources" / "icons" / "app.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="SoundMorpher",
)

# macOS .app bundle
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="SoundMorpher.app",
        icon=str(ROOT / "resources" / "icons" / "app.svg"),
        bundle_identifier="com.soundmorpher.app",
        info_plist={
            "NSMicrophoneUsageDescription": "Sound Morpher needs microphone access for recording.",
            "CFBundleShortVersionString": "1.0.0",
        },
    )
