#!/bin/bash

# the diectory of 3d path database
shapePath=/home/cheng/work_shop/myProjects/3D_data/xie_data/

# where to save the temporary files
dirToSaveViewFeature=/home/cheng/work_shop/code/multi_modality/data/viewFeatureData/xie_data

# tool to generate file list
makeFileList=tools/test_dir.e

# convert shape format
convertShape=tools/tools_meshsimplify.e

# generate views
generateViews=tools/lfd.e

# generate bow
generateBow=tools/extractFeature

# generate npy data 

# generate off file list and then conver the off format to obj format
generateObjFile(){
	echo "generate obj files ... "
	fileListName=$(ls $shapePath)
	objList=$(ls $dirToSaveViewFeature/*obj*)

#	if [ ${#objList[@]} -eq 0 ]
#	then
		if [ ! -d $dirToSaveViewFeature ]
		then
			mkdir -p ${dirToSaveViewFeature}
		fi

		for file in $fileListName
		do
			echo $shapePath$file
			fileToSave=$dirToSaveViewFeature/$file"_offList.txt"
			$makeFileList $shapePath$file | grep ".off$" > $fileToSave
		done

		export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:tools
		for file in $(ls $shapePath)
		do
			temp=$dirToSaveViewFeature/$file
			fileToSave=$temp"_objList.txt"

			$convertShape off2obj $temp"_offList.txt" > $temp"_obj"

			echo "generating $fileToSave"
			for line in $(more $temp"_obj")
			do 
				echo ${line%".obj"} >> $fileToSave
			done	
	
			rm $temp"_offList.txt" $temp"_obj"
		done
#	fi
}

# generate label index and label map for shape dataset
generateLabelFile(){
	echo "generate label files ... "

	labelList=$(ls $dirToSaveViewFeature/*_labelMap.txt)

	if [ ${#labelList[@]} -eq 0 ]
	then
		objList=$(ls $dirToSaveViewFeature)
		for file in $objList
		do
			prefix=${file%_*}
			for line in $(more $dirToSaveViewFeature/$file)
			do
				labelFile=$dirToSaveViewFeature/$prefix"_label.txt"
				tempLine=${line%/*}
				echo ${tempLine##*/} >> $labelFile
			done
		done

		labelList=$(ls $dirToSaveViewFeature/*_label.txt)
		for file in $labelList
		do
			prefix=${file%_*}
			labelMap=$prefix"_labelMap.txt"
			labelIndex=$prefix"_labelIndex.txt"
			label=0
			lastLine=""
			for line in $(more $file)
			do
				if [ "$line" != "$lastLine" ]
				then
					label=$(($label+1))
					lastLine=$line
					echo $label $line >> $labelMap
				fi

				echo $label >> $labelIndex
			done

			rm $file
		done
	fi
}

generateViews(){
	echo "generate views ... "	

#	sudo chmod -R +r $dirToSaveViewFeature
#	viewImgFileList=$(ls $dirToSaveViewFeature/*_viewListImage.txt)
#	if [ ${#viewImgFileList[@]} -eq 0 ]
#	then
	
		for file in $(ls $dirToSaveViewFeature/*_objList.txt)
		do
			postfix=${file##*/}		
			prefix=${file%_*}
			$generateViews $file $prefix"_labelIndex.txt" $prefix"_viewListImage.txt" $prefix"_image" tools
			echo $file
		done
#	fi
}

generateFeature()
{
	echo "generate bow ..." 
#	bowFileList=$(ls $dirToSaveViewFeature/*_BOW.txt)
#	if [ ${#bowFileList[@]} -eq 0 ]
#	then
		for file in $(ls $dirToSaveViewFeature/*_viewListImage.txt)
		do
			postfix=${file##*/}
			case "$postfix" in 
				"SHREC_2014_Real_viewListImage.txt" | "SHREC_2014_Synthetic_viewListImage.txt" )
					nWords=1000
					;;
#				 "dl_test_viewListImage.txt" )
#					nWords=1500
#					;;
				 "xie_data_viewListImage.txt" )
					nWords=1000
					;;							
			esac

			prefix=${file%_*}
			$generateBow $file 1000 $prefix 	
		done		
#	fi
}

generateObjFile
generateLabelFile
generateViews
generateFeature

exit 0
