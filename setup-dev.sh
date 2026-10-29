python -m pip install pip==26.2.2
python -m pip install --uploaded-prior-to=P7D -r requirements.txt
python -m pip install --uploaded-prior-to=P7D -r requirements-dev.txt
cd abides-core
python setup.py develop
cd ../abides-markets
python setup.py develop
cd ..