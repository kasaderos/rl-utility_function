#!/bin/bash
# gcc 9.4.0
apt-get install gcc
apt-get install make
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install pandas==2.0.3
pip install numpy==1.24.4
pip install TA-Lib== 0.4.28
pip install scikit-learn==1.3.2
pip install matplotlib==3.7.4
pip install lightgbm==4.2.0