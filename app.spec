# -*- mode: python ; coding: utf-8 -*-

import os

# Path to the extracted embedded Python folder
embedded_python_path = os.path.abspath("embedded_python")

# Path to the MediaPipe folder
mediapipe_path = os.path.abspath("D:\\New folder\\Lib\\site-packages\\mediapipe")

a = Analysis(
    ['app.py'],  # Entry-point script
    pathex=[],  # Optional additional paths
    binaries=[],  # Custom binaries if needed
    datas=[
        # Bundle the extracted Embedded Python runtime
        (os.path.join(embedded_python_path, '*'), 'embedded_python'),
        # Bundle the pretrained model folder
        ('pretrained_models/pm_37vtrain_mp.pkl', 'pretrained_models'),
        # Bundle the entire mediapipe folder
        (mediapipe_path, 'mediapipe'),  # This will include the whole mediapipe library and submodules
    ],
    hiddenimports=[
        'mediapipe',
        'mediapipe.python',
        'mediapipe.python.solutions.pose',
        'mediapipe.python.solutions.drawing_utils',
        'mediapipe.python.solutions.holistic',
        'tensorflow.lite',
        'cv2',
        'numpy',
        'tkinter',
        'sklearn',
	'cvzone', 
	'opencv-python', 
	'tensorflow'# Add any missing modules here

        'sklearn.neural_network',  # Include the neural network module
        'sklearn.base',  # Include base if necessary
        'sklearn.preprocessing',  # Add other sklearn submodules as needed
        'sklearn.ensemble',  # Example: include other sklearn models
        'pandas',
    ],  # Include necessary hidden imports
    hookspath=[],  # Add custom hooks if required
    hooksconfig={},
    runtime_hooks=[],  # Add any runtime hooks here if needed
    excludes=[],  # Exclude unnecessary packages if needed
    noarchive=False,  # Keep in mind that some of the package's files will not be packed into archives
    optimize=0,  # No optimization; can be set to 2 for max optimization (only if needed)
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Set to False if you encounter issues with UPX
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False if you want to suppress the console
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,  # Set to False if you encounter issues with UPX
    name='app',
)
