# ALPAO-interface

To compile with the ASDK and milk [ImageStreamIO](https://github.com/milk-org/ImageStreamIO) libraries:
	
	gcc -o build/runALPAO runALPAO.c -lImageStreamIO -lasdk -lpthread
	
Before running, set the path to the ALPAO configuration files and copy \<serial\>_userconfig.txt to the same directory:
	
	export ACECFG=$HOME/ALPAO/Config
	
To ensure drivers are loaded (if exao0 has been recently rebooted, for example), run with root privileges:

	/usr/src/interface_alpao/diobminsmod && /usr/src/interface_alpao/util/dpg0101 -s 2x72c

------------------------

To enter the DM control loop with default settings:

	./runALPAO <serialnumber>

`ctrl+c` will exit the loop and safely reset and release the DM. To run with bias and normalization conventions disabled and inputs in fractional stroke:

	./runALPAO <serialnumber> --nobias --nonorm --fractional

For help:

	./runALPAO --help

Once the control loop is running, the DM can be commanded by writing a 97x1 image to shared memory using your tool of choice. By default, double-precision inputs are expected in microns of stroke. A few examples using the milk package are provided:

	./loadfits <path-to-fits-file> <serial>
	./setpix <value> <actuator-number> <serial>
	
Alternatively, [PyImageStreamIO](https://github.com/milk-org/pyImageStreamIO) provides a simple interface in Python.
