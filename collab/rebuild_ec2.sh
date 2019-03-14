# Install system dependencies
sudo apt-get install openjdk-8-jdk
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install bazel
sudo apt-get install -y \
    xorg-dev \
    libglu1-mesa \
    libgl1-mesa-dev \
    xvfb \
    libxinerama1 \
    libxcursor1 \
    gfortran
pip install imutils
conda install opencv

# Install SUMO
cd ~
rm -rf sumo
git clone https://github.com/eclipse/sumo.git
cd sumo
git checkout cbe5b73d781
cd build
mkdir cmake-build
cd cmake-build
cmake ../..
make -j8

# Install ray
cd ~
rm -rf ray_results
rm -rf ray
git clone https://github.com/ray-project/ray.git
cd ray/python
pip install -e . --verbose
cd ~

# Install flow meng-master branch
cd ~
pip uninstall flow
rm -rf flow
git clone https://github.com/flow-project/flow.git
cd flow
git checkout meng_master
pip install -e .

# Run script.py in the background.
#nohup xvfb-run -a -s  "-screen 0 1400x900x24 +extension RANDR" -- python script.py > script.log 2>&1 &
