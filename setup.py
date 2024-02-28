from distutils.core import setup
setup(
  name = 'phph',
  packages = ['phph'],
  version = '0.1',
  license='MIT',
  description = 'A Python package for PH/PH/c queueing systems',
  author = 'Anders Reenberg Andersen',
  author_email = 'andersra@live.dk',
  url = 'https://github.com/areenberg/phph',
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    #<<<--- NEED TO UPDATE
  keywords = ['queueing-theory','queueing','phase-type','waiting-time','service','probability'],
  install_requires=[
          'numpy',
          'scipy',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.10',
  ],
)