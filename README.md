## OpenCL-JPEG(Beta)

#### Intro

	Implement of JPEG codec with OpenCL fork from IJG libjpeg (v8)

#### Install

* CPU Version 

	./configure && make && make install

then you can find **cjpeg djpeg jpegtran rdjpgcom wrjpgcom** in the source directory

* GPU Version

	* OSX
	
		./configure --enable-opencl --enable-shared=no && make && make install


	then you can find **jpegresize** in the source directory


	* Linux

	because I do not have an Linux Computer supported OpenCL, so I do not test this branch


#### TODO

	* Benchmark

#### LICENSE

	MIT License
