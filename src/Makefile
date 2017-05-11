CXX=g++
CFLAGS=-I/home/kaushik/numberPlate/src/mser/MSER/extrema-edu/extrema -I/home/kaushik/numberPlate/src/mser/MSER/extrema-edu/imageutls -I/home/kaushik/numberPlate/src/mser/MSER/extrema-edu/LL -I/home/kaushik/numberPlate/src/mser/MSER/extrema-edu/optionGM -I/home/kaushik/numberPlate/src/mser/MSER/extrema-edu/utls -I/usr/local/include -I/usr/local/include/opencv2 -I/usr/local/cuda/include -I/home/kaushik/caffe/include 
LIBS=-L/usr/local/lib -L/home/kaushik/numberPlate/src/mser/MSER/extrema-edu/extrema -L/home/kaushik/numberPlate/src/mser/MSER/extrema-edu/LL -L/home/kaushik/numberPlate/src/mser/MSER/extrema-edu/utls -L/home/kaushik/numberPlate/src/mser/MSER/extrema-edu/optionGM -L/usr/local/cuda/lib -L/home/kaushik/caffe/build/lib
all:
	        $(CXX) -std=c++11 -o findNumPlate main.cpp Detector.cpp Reader.cpp mser.cpp Convnet.cpp $(CFLAGS) $(LIBS) -lcaffe -lglog -lopencv_core -lopencv_highgui -lopencv_imgproc -lExtrema -lLL -lutls -lOptionGM -lpng -ljpeg -lz -lrt -lm -lpthread -lboost_system -lboost_filesystem
	        $(CXX) -std=c++11 -o findNumPlate_folder main_folder.cpp Detector.cpp Reader.cpp mser.cpp Convnet.cpp $(CFLAGS) $(LIBS) -lcaffe -lglog -lopencv_core -lopencv_highgui -lopencv_imgproc -lExtrema -lLL -lutls -lOptionGM -lpng -ljpeg -lz -lrt -lm -lpthread -lboost_system -lboost_filesystem

debug:
	        $(CXX) -std=c++11 -g -D DEBUG -o findNumPlate main.cpp Detector.cpp Reader.cpp mser.cpp Convnet.cpp $(CFLAGS) $(LIBS) -lcaffe -lglog -lopencv_core -lopencv_highgui -lopencv_imgproc -lExtrema -lLL -lutls -lOptionGM -lpng -ljpeg -lz -lrt -lm -lpthread -lboost_system -lboost_filesystem
	        $(CXX) -std=c++11 -g -D DEBUG -o findNumPlate_folder main_folder.cpp Detector.cpp Reader.cpp mser.cpp Convnet.cpp $(CFLAGS) $(LIBS) -lcaffe -lglog -lopencv_core -lopencv_highgui -lopencv_imgproc -lExtrema -lLL -lutls -lOptionGM -lpng -ljpeg -lz -lrt -lm -lpthread -lboost_system -lboost_filesystem

profile:
	        $(CXX) -std=c++11 -g -pg -o findNumPlate main.cpp Detector.cpp Reader.cpp mser.cpp Convnet.cpp $(CFLAGS) $(LIBS) -lcaffe -lglog -lopencv_core -lopencv_highgui -lopencv_imgproc -lExtrema -lLL -lutls -lOptionGM -lpng -ljpeg -lz -lrt -lm -lpthread -lboost_system -lboost_filesystem
	        $(CXX) -std=c++11 -g -pg -o findNumPlate_folder main_folder.cpp Detector.cpp Reader.cpp mser.cpp Convnet.cpp $(CFLAGS) $(LIBS) -lcaffe -lglog -lopencv_core -lopencv_highgui -lopencv_imgproc -lExtrema -lLL -lutls -lOptionGM -lpng -ljpeg -lz -lrt -lm -lpthread -lboost_system -lboost_filesystem

