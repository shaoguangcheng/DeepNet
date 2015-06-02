CC=gcc
CXX=g++
AR=ar


src-all := $(wildcard *.c)
obj-all := $(patsubst %.c,%.o,$(src-all))
inc-all := $(wildcard *.h)
bin-all := $(patsubst %.c,%.e,$(src-all))

target  := libfftw.a


all : $(target)

$(target) : $(obj-all) $(inc-all)
	$(AR) rcs $@ $(obj-all) 

%.e:%.cpp $(inc-y)
	$(CXX) $< -o $@ -lm

%.o:%.cpp
	$(CXX) -c $< -o $@ 

clean :
	rm -f *.e *.o *.a

