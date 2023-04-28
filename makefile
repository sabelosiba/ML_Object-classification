# ML Assignment makefile

install: venv
	. venv/bin/activate; pip3 install -Ur requirements.txt

venv :
	test -d venv || python3 -m venv venv
	virtualenv -p python3 venv

clean:
	rm -rf venv
	rm -rf data
	find -iname "*.pyc" -delete

runMLP:
	python3 MLP.py

runCNN:
	python3 CNN.py

runRES:
	python3 RESNET.py