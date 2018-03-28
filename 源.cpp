#include <iostream>
#include <cv.h>
#include <opencv2\opencv.hpp>
#include <cmath>

using namespace std;
using namespace cv;

/*
 * 图像增强，将灰度映射到一个范围
 */
void Imadjust(Mat& inout,
	double low_in, double high_in, 
	double low_out, double high_out, double gamma) {
	
	double val;
	uchar* p;
	for (int y = 0; y < inout.rows; ++y) {
		p = inout.ptr<uchar>(y);
		for (int x = 0; x < inout.cols; ++x) {
			val = pow((p[x] - low_in) / (high_in - low_in), gamma)*(high_out - low_out) + low_out;
			if (val > 255) val = 255;
			if (val < 0) val = 0;
			inout.at<uchar>(y, x) = (uchar)val;
		}
	}
	

}

/*
 * 提取视频中的帧，20帧保存一张
 */
void VedioFrameExtraction(VideoCapture& capture) {
	int frame_num = capture.get(CV_CAP_PROP_FRAME_COUNT);
	Mat frame;
	int cur_frame = 0;
	cout << "开始读视频帧.每20帧写入一张." << endl;
	while (true) {
		capture.read(frame);
		if (cur_frame % 20 == 0) {
			cout << "正在写入第" << cur_frame << "帧" << endl;
			imwrite("D:\\Pictures\\" + to_string(cur_frame) + ".png", frame);
		}
		if (cur_frame >= frame_num)
			break;
		++cur_frame;
	}
	cout << "读取已完成." << endl;
}

/*
 * 输入图片，如果存在字幕，返回true，同时把处理后的图片传入outputImg
 * 如果不存在字幕，返回false
 */
bool SubtitleExtraction(const Mat& inputImg, Mat& outputImg) {
	Mat dst_img, temp, show;
	/*灰度化*/
	cvtColor(inputImg, temp, CV_RGB2GRAY);
	Imadjust(temp, 50, 200, 0, 255, 1);

	/*形态学开操作*/
	Mat ele = getStructuringElement(MORPH_ELLIPSE, Size(12, 12));
	morphologyEx(temp, dst_img, MORPH_OPEN, ele);
	/*差分*/
	dst_img = temp - dst_img;

	/*切割*/
	dst_img = dst_img.rowRange(dst_img.rows * 3 / 4, dst_img.rows);
	/*二值化*/
	threshold(dst_img, show, 230, 255, THRESH_BINARY);
	
	/*形态学操作，使字幕区域连通*/
	ele = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
	morphologyEx(show, dst_img, MORPH_OPEN, ele);
	morphologyEx(dst_img, dst_img, MORPH_CLOSE, ele);

	ele = getStructuringElement(MORPH_ELLIPSE, Size(1, 150));
	morphologyEx(dst_img, dst_img, MORPH_CLOSE, ele);
	ele = getStructuringElement(MORPH_ELLIPSE, Size(30, 5));
	morphologyEx(dst_img, dst_img, MORPH_CLOSE, ele);

	/*二值化*/
	threshold(dst_img, dst_img, 200, 255, THRESH_BINARY);
	/*形态学闭操作，去掉一些尖刺*/
	ele = getStructuringElement(MORPH_ELLIPSE, Size(4, 4));
	morphologyEx(dst_img, dst_img, MORPH_OPEN, ele);

	bool has_caption = false;
	vector<vector<Point>> contours;
	/*提取出连通区域*/
	findContours(dst_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	int size = contours.size();
	int width, height, x, y;
	vector<Rect> boundRect(size);

	/*遍历各个连通区域*/
	for (int i = 0; i < size; ++i) {
		/*连通区域的外接矩形*/
		boundRect[i] = boundingRect(Mat(contours[i]));
		width = boundRect[i].width + 20;
		height = boundRect[i].height + 10;
		x = boundRect[i].x - 10;
		y = boundRect[i].y - 10;

		/*防止越界*/
		if (y < 0)
			y = 0;
		if (y + height > dst_img.rows)
			height -= 10;
		if (x < 0)
			x = 0;
		if (x + width > dst_img.cols)
			width -= 20;

		/*字幕存在条件*/
		if (abs(x + width / 2 - dst_img.cols / 2) < 50 && abs(dst_img.rows - y - height) < 30 && width*height >= 4000) {
			has_caption = true;
			cout << width*height << endl;
			show(Rect(x, y, width, height)).copyTo(dst_img(Rect(x, y, width, height)));
		}
		else {
			/*不存在字幕的区块涂成黑色*/
			dst_img(Rect(x, y, width, height)) = 0;
		}
	}

	/*将处理后的图片传给outputImg*/
	dst_img.copyTo(outputImg);
	return has_caption;
}


int main(int argc, char** argv)
{
	//VideoCapture capture(argv[1]);
	Mat img, dst_img;
	bool has_caption;
	
	for (int i = 0; i < 1500; ++i) {
		img = imread("D:\\Pictures\\" + to_string(i * 20) + ".png");
		has_caption = SubtitleExtraction(img, dst_img);
		cout << i * 20 << endl;
		if(has_caption)
			imwrite("D:\\Pictures\\captions\\" + to_string(i * 20) + ".png", dst_img);
	}
	
	system("pause");
	return 0;
}