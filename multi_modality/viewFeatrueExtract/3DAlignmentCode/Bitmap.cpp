//#include <windows.h>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Bitmap.h"


using namespace cv;

// the input "Buff" has "no" pad after every row
// so, it must add "pad" after every row
void WriteBitmap(unsigned char *Buff, int x, int y, char *filename)
{
    Mat             img;
    unsigned char   *p1, *p2;
    int             i;

    img.create(y, x, CV_8UC3);
    p2 = img.data;
    p1 = Buff;

    for(i=0; i<3*x*y; i++) p2[i] = p1[i];

    imwrite(filename, img);
}

void WriteBitmap8(unsigned char *sBuff, int x, int y, char *filename)
{
    Mat             img;
    unsigned char   *p1, *p2;
    int             i;

    img.create(y, x, CV_8UC1);
    p2 = img.data;
    p1 = sBuff;

    for(i=0; i<x*y; i++) p2[i] = p1[i];

    imwrite(filename, img);
}

void WriteRAW(unsigned char *Buff, int x, int y, char *filename)
{
	FILE	*fpt;
	int		i;

	fpt=fopen(filename, "wb");
	// write width and height
	fwrite(&x, 4, 1, fpt);
	fwrite(&y, 4, 1, fpt);
	// write only 1 byte per pixel
	for(i=0; i<3*x*y; i+=3)
		fprintf(fpt, "%c", Buff[i]);
	fclose(fpt);
}
