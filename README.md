## OpenCL-JPEG(Beta)

#### Intro

Implement of JPEG codec with OpenCL fork from IJG libjpeg (v8)
	
* Decode/Encode JPEG Image
* Resize JPEG Image
* Do not support libjpeg(v8) smart scale, because I do not understand the code :D
* Do not support other transform now, such as rotate, crop, etc :D

I recommend to use this lib with resize feature, because data transform between CPU and GPU is very expensive.

#### Install

* CPU Version 

	./configure && make && make install

then you can find **cjpeg djpeg jpegtran rdjpgcom wrjpgcom** in the source directory

* GPU Version

	* OSX
		
		./configure --enable-opencl --enable-shared=no && make && make install


	then you can find **jpegresize** in the source directory. It will be some mistakes when compile shared library, so you need to disable shared library


	* Linux

		./configure --enable-opencl --with-openclincludedir=xxx --with-opencllibdir=xxx
		
	because I do not have an Linux Computer supported OpenCL, so I do not test this branch, if you find any problem please let me know

#### Usage

	./jpegresize -w 500 -h 300 -q 80 -outfile han.jpg testimg.jpg
	
The command will resize testimg.jpg to 500x300 image use your GPU device.

#### TODO

* Code & Performance
* Benchmark
* Http Server

#### LICENSE

MIT License
