
conda update conda

conda create -n epymarl_env  python=3.8 jupyter

conda activate epymarl_env

conda install -y clang_osx-arm64 clangxx_osx-arm64 gfortran_osx-arm64

MACOSX_DEPLOYMENT_TARGET=12.3 CC=clang CXX=clang++ python -m pip --no-cache-dir   install torch torchvision torchaudio

python -c "import torch; print(torch.__version__)"  #---> (Confirm the version is 1.11.0)

MACOSX_DEPLOYMENT_TARGET=12.3 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+${cpu}.html

MACOSX_DEPLOYMENT_TARGET=12.3 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+${cpu}.html

MACOSX_DEPLOYMENT_TARGET=12.3 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-geometric