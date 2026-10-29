python3 -m pip install pip==26.2.2
python3 -m pip install --uploaded-prior-to=P7D -r requirements.txt
python3 -m pip install --uploaded-prior-to=P7D -r requirements-dev.txt
cd abides-core
python3 setup.py install
cd ../abides-markets
python3 setup.py install
cd ../abides-gym
python3 setup.py install
cd ..
