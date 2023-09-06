## chrislib

#### to install
* create a python3 virtualenv (`python3 -m venv venv`) and start it (`source venv/bin/activate`)
* run `python setup.py install` (or `python setup.py develop` for developer mode) to install chrislib as a library

#### to set up documentation
###### from scratch
* `pip install sphinx`
* `mkdir docs`
* `cd docs`
* `sphinx-quickstart`
###### build html
* `cd docs`
* `make clean && make html`
* in your file browser, navigate to _build/html/index.html and double-click. This should open a browser with the HTML loaded.

