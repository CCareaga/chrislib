## chrislib

#### to install
* create a python3 virtualenv (`python3 -m venv venv`) and start it (`source venv/bin/activate`)
* install chrislib dependencies with `pip install -r requirements.txt`
* run `python setup.py install` (or `python setup.py develop` for developer mode) to install chrislib as a library
* to use methods and classes from "altered_midas," run `pip install https://github.com/CCareaga/MiDaS/archive/master.zip`
* (Note: omnidata_tools is installed by default in requirements.txt) to use methods and classes from "omnidata_tools," (used in normal_util.py), run `pip install https://github.com/CCareaga/omnidata/archive/main.zip`

#### to run pylint
* with the virtualenv activated, `pip install pylint`
* run pylint with `pylint --rcfile .pylintrc chrislib/*.py`

#### to run pytests
* with the virtualenv activated, `pip install pytest`
* run tests:
    * to run all tests in the "tests" directory, simply run the command `pytest tests`
    * to run all tests in one file, e.g. "filename.py," run `pytest tests/filename.py`
    * to only run tests whose names match a string, e.g. "test_loss," run `pytest -k test_loss`

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
