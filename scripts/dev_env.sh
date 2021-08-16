conda create -n asdnet -c pytorch pytorch=1.3.0 torchvision python=3.7 -y

source activate asdnet

pip uninstall Pillow
pip install Pillow-SIMD
pip install python_speech_features
pip install natsort
pip install scipy
pip install sklearn
pip install pandas
