#define GLUT_DISABLE_ATEXIT_HACK

#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include <stdio.h>
#include <malloc.h>
#include <memory.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <limits.h>

#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>

#include "win_types.h"
#include "Ds.h"
#include "RWObj.h"
#include "Bitmap.h"
#include "TranslateScale.h"
#include "Rotate.h"
#include "RegionShape.h"
#include "RecovAffine.h"
#include "Refine.h"
#include "Edge.h"
#include "convert.h"
#include "ColorDescriptor.h"
#include "Circularity.h"
#include "FourierDescriptor.h"
#include "Eccentricity.h"

#define abs(a) (a>0)?(a):-(a)

#define	QUANT8				256		// 2^8
#define FD_SCALE			2		// *2 first, and then quantization
#define CIR_SCALE			2.318181818		// the range of circularity is [0~110], so *2.318 to be [0~255]
#define ECC_SCALE			25.5			// the range of circularity is [0~10], so *25.5 to be [0~255]

unsigned char	CamMap[CAMNUM_2]={0,1,2,3,4,5,6,7,8,9,5,6,7,8,9,2,3,4,0,1};

char srcfn[100];
char destfn[100];

int			winw = WIDTH, winh = HEIGHT;

pVer		vertex=NULL;
pTri		triangle=NULL;
int			NumVer=0, NumTri=0;		// total number of vertex and triangle.

pVer		vertex1, vertex2;
pTri		triangle1, triangle2;
int			NumVer1, NumTri1, NumVer2, NumTri2;		// total number of vertex and triangle.

// translate and scale of model 1 and 2
Ver				Translate1, Translate2;
double			Scale1, Scale2;

void FindCenter(unsigned char *srcBuff, int width, int height, double *CenX, double *CenY)
{
    int					x, y, count;
    unsigned char		*pImage;
    int					maxX, minX, maxY, minY;
    int					MeanX, MeanY;

    count = 0;
    pImage = srcBuff;

    // ***********************************************************************************************
    // if use "mean" to each 2D shape independnetly, the origin will be moved a lot in 3D
    // if ues "center" to each 2D shape independnetly, the origin will be moved only a little in 3D
    // if center can be defined in 3D, the origin will not be moved any more.
    // But this will not very robust in 3D similarity transformation
    // In addition, to make center of each 2D shape more close to user drawn 2D shapes,
    // it's better to define center for each 2D shape independently

    // uee center of max and min to be center
    maxX = maxY = -1;
    minX = minY = INT_MAX;
    for (y=0 ; y<height ; y++)
    for (x=0 ; x<width; x++)
    {
        if( *pImage < 255 )
        {
            if( x > maxX ) maxX = x;
            if( x < minX ) minX = x;
            if( y > maxY ) maxY = y;
            if( y < minY ) minY = y;
        }
        pImage++;
    }

    if( maxX > 0 )
    {
        *CenX = (maxX+minX) / 2.0;
        *CenY = (maxY+minY) / 2.0;
    }
    else
        *CenX = *CenY = -1;		// nothing to be rendered
}

void display(void)
{
    int				i, j;
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glPushMatrix();
//		glColor3f((GLfloat)0.0, (GLfloat)0.0, (GLfloat)0.0);
        for(i = 0; i<NumTri; i++)
        {
            glColor3f((GLfloat)triangle[i].r, (GLfloat)triangle[i].g, (GLfloat)triangle[i].b);
            glBegin(GL_POLYGON);
                for(j=0; j<triangle[i].NodeName; j++)
                    glVertex3d(vertex[triangle[i].v[j]].coor[0], vertex[triangle[i].v[j]].coor[1], vertex[triangle[i].v[j]].coor[2]);
            glEnd();
        }
    glPopMatrix();

    glutSwapBuffers();
}

void RenderToMem(unsigned char *bmBits, unsigned char *bmColor, pVer CamVertex, pVer v, pTri t, int nv, int nt)
{
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    gluLookAt(CamVertex->coor[0], CamVertex->coor[1], CamVertex->coor[2],
                0, 0, 0,
                0, 1, 0);

    vertex = v;
    triangle = t;
    NumVer = nv;
    NumTri = nt;
    display();

    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, winw, winh, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, bmBits);
    if( bmColor )
        glReadPixels(0, 0, winw, winh, GL_RGB, GL_UNSIGNED_BYTE, bmColor);
}


char* labelIndexFile = NULL;
char* objFileList = NULL;
char* dirToSaveImg = NULL;
char* imgFileList = NULL;
char* templateObj = NULL;

void saveDepthImage(unsigned char key, int x, int y)
{
    unsigned char	*srcBuff[CAMNUM], *EdgeBuff, *ColorBuff[CAMNUM];
    char			filename[400],fname[300],srcName[500],edgeName[500],currentDir[500],currentModelName[20];
    pVer			CamVertex[ANGLE];
    pTri			CamTriangle[ANGLE];
    int				CamNumVer[ANGLE], CamNumTri[ANGLE];		// total number of vertex and triangle.
    int				i,srcCam,destCam,Count,lineNumber=0,num = 0;
    int             preModelName = 1;
    FILE  *depthImageList,*modelNames,*fpt1;

    switch (key)
    {
    case 27:
        exit(0);
        break;

    case 'n':
        GenerateBasisLUT();

        for(destCam=0; destCam<ANGLE; destCam++){
            sprintf(filename, "%s/12_%d", templateObj, destCam);
            ReadObj(filename, CamVertex+destCam, CamTriangle+destCam, CamNumVer+destCam, CamNumTri+destCam);
        }

        for(i=0; i<CAMNUM; i++){
            srcBuff[i] = (unsigned char *) malloc (winw * winh * sizeof(unsigned char));
            ColorBuff[i] = (unsigned char *) malloc (3 * winw * winh * sizeof(unsigned char));
        }

        EdgeBuff = (unsigned char *) malloc (winw * winh * sizeof(unsigned char));

//        fpt1          = fopen("modelNameData/PSB_OBJList.txt", "r");
//       modelNames    = fopen("modelNameData/PSB_labelIndex.txt","r");
//        depthImageList = fopen("featureData/PSB_depthImageList.txt","w");

        fpt1          = fopen(objFileList, "r");
        modelNames    = fopen(labelIndexFile,"r");
        depthImageList = fopen(imgFileList,"w");

//        getcwd(currentDir,sizeof(currentDir));
//        strcat(currentDir,"/PSB_image");
        mkdir(dirToSaveImg,S_IRWXU);

        Count = 1;

        while( !feof(fpt1) )
        {
            if( NULL == fgets(fname, 300, fpt1) ) break;

            fname[strlen(fname)-1] = '\0';
            fgets(currentModelName,20,modelNames);
            currentModelName[strlen(currentModelName)-1] = '\0';

            num++;
            if(atoi(currentModelName) != preModelName)
                num = 1;
            preModelName = atoi(currentModelName);

            printf("%d.", Count++);
            printf("process model: %s ...\n\n",fname);
            if( ReadObj(fname, &vertex1, &triangle1, &NumVer1, &NumTri1) == 0 ){
                printf("can not open %s\n",fname);
                continue;
            }

            TranslateScale(vertex1, NumVer1, triangle1, NumTri1, fname, &Translate1, &Scale1);

            for(srcCam=0; srcCam<ANGLE; srcCam++){
                for(i=0; i<CAMNUM; i++)
                    RenderToMem(srcBuff[i], NULL, CamVertex[srcCam]+i, vertex1, triangle1, NumVer1, NumTri1);

                for(i=0; i<CAMNUM; i++){
                    sprintf(srcName,"%s/%s_%d_%d_%d_src.bmp",dirToSaveImg,currentModelName,num,srcCam,i);
 //                   sprintf(edgeName,"%s/%s_%d_%d_%d_edge.bmp",currentDir,currentModelName,num,srcCam,i);
 //                   EdgeDetect(EdgeBuff, srcBuff[i], winw, winh);
                    WriteBitmap8(srcBuff[i], winw, winh, srcName);
 //                   WriteBitmap8(EdgeBuff, winw, winh, edgeName);
                    fprintf(depthImageList,"%d ",++lineNumber);
                    fprintf(depthImageList,"%s ",currentModelName);
                    fprintf(depthImageList,"%d ",num);
                    fprintf(depthImageList,"%s\n",srcName);
                }
        }

            free(vertex1);
            free(triangle1);

        }

        for(i=0; i<CAMNUM; i++) {
            free(srcBuff[i]);
            free(ColorBuff[i]);
        }

        free(EdgeBuff);
        fclose(fpt1);
        fclose(depthImageList);
        fclose(modelNames);

        for(destCam=0; destCam<ANGLE; destCam++)
        {
            free(CamVertex[destCam]);
            free(CamTriangle[destCam]);
        }
        NumTri = 0;
        break;

    default:
        break;
    }

    exit(0);
}

void init(void)
{
    glClearColor (1.0, 1.0, 1.0, 0.0);
    glClearDepth(1.0);
    glEnable(GL_DEPTH_TEST);
}

void reshape (int w, int h)
{
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
//	gluPerspective(90.0, (GLfloat) winw/(GLfloat) winh, 0.1, 10.0);
    glOrtho(-1, 1, -1, 1, 0.0, 2.0);
//	glOrtho(-1, 1, -1, 1, 0.5, 1.5);
    glViewport (0, 0, (GLsizei) winw, (GLsizei) winh);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(1,0,0,0,0,0,0,1,0);
//	gluLookAt(CAMSCALE*-0.85065, CAMSCALE*0.61803, CAMSCALE*-0.20081, 0,0,0,0,1,0);
//	gluLookAt(-0.85065, 0.61803, -0.20081, 0,0,0,0,1,0);
}

int main(int argc, char** argv)
{
    if(argc != 6){
        printf("usage : %s objFileList labelIndexFile imgFileList dirToSaveImg templateObjDir\n", argv[0]);
        return -1;
    }
    else{
        objFileList    = argv[1];
        labelIndexFile = argv[2];
        imgFileList    = argv[3];
        dirToSaveImg   = argv[4];
        templateObj    = argv[5];
    }

	glutInit(&argc, argv);
	glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize (WIDTH, HEIGHT); 
	glutInitWindowPosition (100, 100);
	glutCreateWindow (argv[0]);
	init ();
    glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	//   glutMouseFunc(mouse);
	//   glutMotionFunc(motion);
    glutKeyboardFunc(saveDepthImage);

	glutMainLoop();

	return 0;
}
