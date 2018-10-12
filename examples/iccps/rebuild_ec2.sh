cd ~
rm -rf sumo
git clone https://github.com/eclipse/sumo.git
cd sumo
git checkout 1d4338ab80
make -f Makefile.cvs
./configure
make -j4
cd ~
rm -rf ray
git clone https://github.com/eugenevinitsky/ray.git
cd ray/python
python setup.py develop
cd ~
sudo apt-get install -y \
    xorg-dev \
    libglu1-mesa libgl1-mesa-dev \
    xvfb \
    libxinerama1 libxcursor1 \
cd ~/flow/examples/iccps/
# Running script in the background.
#nohup xvfb-run -a -s \
#    "-screen 0 1400x900x24 +extension RANDR"\
#     -- python script.py > script.out 2>&1 &
