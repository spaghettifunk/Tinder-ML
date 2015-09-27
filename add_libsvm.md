# how to add libsvm support?

* `https://github.com/cjlin1/libsvm`
* `cd libsvm`
* `make`
* `cd python/`
* `make`
* `pwd` -> copy the path somewhere, I call it (libsvmpath)
* edit `virtualenvironment/bin/activate/`
* add `PYTHONPATH="$PYTHONPATH:(libsvmpath)"
	* and `export PYTHONPATH`

* execute `source virtualenvironment/bin/activate` 
