# Installation Guide


First do not forget to update submodules required by this repo:

```shell
git submodule update --init
```

Check system dependencies:

```shell
sudo apt install git cmake ninja-build build-essential ccache
sudo apt install libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev
sudo apt install libboost-program-options-dev libboost-graph-dev libboost-system-dev libboost-filesystem-dev libflann-dev libfreeimage-dev libmetis-dev libgtest-dev libgmock-dev libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev
```

It is recommended to compile following tools under gcc/g++ 10:

```shell
sudo apt install gcc-10 g++-10
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDAHOSTCXX=/usr/bin/g++-10
```

It is also recommended to compile with cuda toolkit on nvidia GPUs for accelerated experience, though fully compatible with CPU.

### [CuDSS](https://developer.nvidia.com/cudss-downloads) 0.3.0

Optional, speed up sparse linear solver (Ceres-solver) on nvidia GPUs.
Skip this step if not required or run in pure CPU mode.

```shell
wget https://developer.download.nvidia.com/compute/cudss/0.3.0/local_installers/cudss-local-repo-ubuntu2204-0.3.0_0.3.0-1_amd64.deb
sudo dpkg -i cudss-local-repo-ubuntu2204-0.3.0_0.3.0-1_amd64.deb
sudo cp /var/cudss-local-repo-ubuntu2204-0.3.0/cudss-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install cudss

sudo ln -s /etc/alternatives/libcudss.so /usr/lib/x86_64-linux-gnu/libcudss.so.0.3.0
sudo ln -s /etc/alternatives/libcudss_commlayer_openmpi.so /usr/lib/x86_64-linux-gnu/libcudss_commlayer_openmpi.so.0.3.0
sudo ln -s /etc/alternatives/libcudss_commlayer_nccl.so /usr/lib/x86_64-linux-gnu/libcudss_commlayer_nccl.so.0.3.0
```

### [Ceres-solver](https://github.com/ceres-solver/ceres-solver) 2.3.0

```shell
cd ceres-solver
git submodule update --init
cmake -S . -B build -G Ninja -DCMAKE_CXX_STANDARD=17 -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=TRUE
sudo cmake --build build --target install/strip
```

Please refer to Ceres-solver [official installation guide](http://ceres-solver.org/installation.html) for more information.

### [COLMAP](https://github.com/colmap/colmap) 3.11.0

```shell
cd colmap
cmake -S . -B build -G Ninja -DCMAKE_CXX_STANDARD=17 -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=TRUE
sudo cmake --build build --target install/strip
```

Please refer to COLMAP [official installation guide](https://colmap.github.io/install.html) for more information.


### [GLOMAP](https://github.com/colmap/glomap) 1.0.0

```shell
cd glomap
cmake -S . -B build -G Ninja -DCMAKE_CXX_STANDARD=17 -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=TRUE
cmake --build build
sudo ln -sf $(pwd)/build/glomap/glomap /usr/local/bin/glomap
```

As mentioned in [official repo](https://github.com/colmap/glomap?tab=readme-ov-file#getting-started), auto fetch dependencies for GLOMAP requires CMake>=3.28, you can install CMake either by snap/conda or from source for newer versions.

### [FVD](https://github.com/zhizhou57/FVD)

```shell
pip install FVD/fvdcal-1.0-py3-none-any.whl
```