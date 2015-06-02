#!/bin/bash
dirToSaveViewFeature=../data/viewFeatureData
prog=./lfd.e

#viewImgFileList=$(ls $dirToSaveViewFeature/*_viewListImage.txt)
echo $viewImgFileList
echo ${#viewImgFileList[@]}
#if [ ${#viewImgFileList[@]}-1 -eq 0 ]
#then
	
    for file in $(ls $dirToSaveViewFeature/*_objList.txt)
    do
	prefix=${file%_*}
        $prog $file $prefix"_labelIndex.txt" $prefix"_viewImage.txt" $prefix"_image" ./

    done
#fi

