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
    libxinerama1 libxcursor1
pip install imutils
cd ~/flow/examples/iccps/
# Running script in the background.
#nohup xvfb-run -a -s \
#    "-screen 0 1400x900x24 +extension RANDR"\
#     -- python script.py > script.out 2>&1 &
