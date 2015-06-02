CC=gcc
CXX=g++

################################################################################
# OpenCV settings
# run following command first:
#   EXPORT PKG_CONFIG_PATH=/opt/OpenCV-2.4/lib/pkgconfig
################################################################################
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv)
OPENCV_LIBS   = $(shell pkg-config --libs opencv)

################################################################################
# OpenGL settings
################################################################################
OPENGL_CFLAGS = 
OPENGL_LIBS   = -lGL -lGLU -lglut

################################################################################
# FFTW settings
################################################################################
FFTW_CFLAGS = -I./fftw
FFTW_LIBS   = -L./fftw -lfftw


################################################################################
# Overall settings
################################################################################
ALL_CFLAGS = $(OPENCV_CFLAGS) $(OPENGL_CFLAGS) $(FFTW_CFLAGS)
ALL_LIBS   = $(OPENCV_LIBS)   $(OPENGL_LIBS)   $(FFTW_LIBS)


src-cxx-all := $(wildcard *.cpp)
src-c-all := $(wildcard *.c)
bin-all := $(patsubst %.cpp,%.e,$(src-cxx-all)) $(patsubst %.c,%.e,$(src-c-all))
obj-all := $(patsubst %.cpp,%.o,$(src-cxx-all)) $(patsubst %.c,%.o,$(src-c-all))



all : lfd.e

lfd.e : $(obj-all)
	$(CXX) -o $@ $? $(ALL_LIBS)

%.e:%.cpp
	$(CXX) $< -g -o $@ $(ALL_CFLAGS) $(ALL_LIBS)

%.e:%.c
	$(CXX) $< -g -o $@ $(ALL_CFLAGS) $(ALL_LIBS)

%.o:%.cpp
	$(CXX) -g -c $< -o $@ $(ALL_CFLAGS)

%.o:%.c
	$(CXX) -g -c $< -o $@ $(ALL_CFLAGS)

clean :
	rm -f *.e *.o

