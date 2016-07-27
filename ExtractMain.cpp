
#include "extractTable.hpp"
#include <stdio.h>
#include <io.h>
#include <fstream>
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

void getAllFiles(string path, vector<string>& files);

int main(int argc, char** argv)
{
	Mat src, dst;
	string path = "G:\\change\\to\\your\\directory";
	vector<string> files;
	getAllFiles(path,files);

	for (auto file : files)
	{
		extractTable(file);
	}
	return 0;
}

void getAllFiles(string path, vector<string>& files)
{
	//文件句柄  
	__int64 hFile = 0;
	//文件信息  
	struct __finddata64_t  fileinfo;  //很少用的文件信息读取结构
	string p;  //string类很有意思的一个赋值函数:assign()，有很多重载版本

	
	if ((hFile = _findfirst64(p.assign(path).append("\\*.jpg").c_str(), &fileinfo)) == -1)
	{
		cout << "No file is found\n" << endl;
	}
	else
	{
		do
		{
			files.push_back(p.assign(path).append("\\").append(fileinfo.name));
		} while (_findnext64(hFile, &fileinfo) == 0);  //寻找下一个，成功返回0，否则-1
		_findclose(hFile);
	}
	
}