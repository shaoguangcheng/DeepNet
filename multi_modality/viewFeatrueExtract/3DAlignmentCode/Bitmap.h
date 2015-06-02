#ifndef __BITMAP_H__
#define __BITMAP_H__

#ifdef __cplusplus
extern "C" {
void WriteBitmap(unsigned char *Buff, int x, int y, char *filename);
void WriteBitmap8(unsigned char *sBuff, int x, int y, char *filename);
void WriteRAW(unsigned char *Buff, int x, int y, char *filename);

}
#endif

#endif // end of __BITMAP_H__
