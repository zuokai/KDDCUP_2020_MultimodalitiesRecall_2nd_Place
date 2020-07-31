import tensorflow as tf
import os
from tqdm import tqdm
import time

def WritingFiles(ReadFilePath,WriteFileList):
	with tf.gfile.FastGFile(ReadFilePath) as reader:
		isheadline = True
		nowitem = 0
		OpenFileList = []
		Totalnum = 3000001
		pbar = tqdm(total=Totalnum)
		Nownum = 0
		while Nownum<Totalnum:
			line = reader.readline()
			if isheadline:
				for WriteFile in WriteFileList:
					OpenFileList.append(open(WriteFile,'wb'))
				for OpenFile in OpenFileList:
					OpenFile.write(line)
				isheadline = False
				Nownum += 1
				pbar.update(1)
				continue
			if not line:
				Nownum += 1
				pbar.update(1)
				continue
			OpenFileList[nowitem].write(line)
			nowitem += 1
			if nowitem == len(WriteFileList):nowitem = 0
			pbar.update(1)
		for OpenFile in OpenFileList:
			OpenFile.close()
	print("Success!")
	
if __name__ == '__main__':
	FileList = []
	for i in range(5):
		FileList.append("train%d.tsv"%i)
	WritingFiles('../train.tsv',FileList)



