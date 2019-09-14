# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['process_tiff.py'],
             pathex=['C:\\code\\angiogenesis-analyzer'],
             binaries=[],
             datas=[],
             hiddenimports=['scipy._lib.messagestream', 'scipy', 'scipy.signal', 'scipy.signal.bsplines', 'scipy.special', 'scipy.special._ufuncs_cxx',
                            'scipy.linalg.cython_blas',
                            'scipy.linalg.cython_lapack',
                            'scipy.integrate',
                            'scipy.integrate.quadrature',
                            'scipy.integrate.odepack',
                            'scipy.integrate._odepack',
                            'scipy.integrate.quadpack',
                            'scipy.integrate._quadpack',
                            'scipy.integrate._ode',
                            'scipy.integrate.vode',
                            'scipy.integrate._dop', 'scipy._lib', 'scipy._build_utils','scipy.__config__',
                            'scipy.integrate.lsoda', 'scipy.cluster', 'scipy.constants','scipy.fftpack','scipy.interpolate','scipy.io','scipy.linalg','scipy.misc','scipy.ndimage','scipy.odr','scipy.optimize','scipy.setup','scipy.sparse','scipy.spatial','scipy.special','scipy.stats','scipy.version'],

             hookspath=[],
             runtime_hooks=[],
             excludes=[],
            )

a.datas += Tree('./scipy-extra-dll', prefix=None)

pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='process_tiff',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )

app = BUNDLE(exe,
         name='process_tiff.app',
         icon=None,
         bundle_identifier=None,
         info_plist={
            'NSHighResolutionCapable': 'True'
            },
         )
