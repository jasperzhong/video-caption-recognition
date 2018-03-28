#include <iostream>
#include <cv.h>
#include <opencv2\opencv.hpp>
#include <cmath>

using namespace std;
using namespace cv;

/*
 * ͼ����ǿ�����Ҷ�ӳ�䵽һ����Χ
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
 * ��ȡ��Ƶ�е�֡��20֡����һ��
 */
void VedioFrameExtraction(VideoCapture& capture) {
	int frame_num = capture.get(CV_CAP_PROP_FRAME_COUNT);
	Mat frame;
	int cur_frame = 0;
	cout << "��ʼ����Ƶ֡.ÿ20֡д��һ��." << endl;
	while (true) {
		capture.read(frame);
		if (cur_frame % 20 == 0) {
			cout << "����д���" << cur_frame << "֡" << endl;
			imwrite("D:\\Pictures\\" + to_string(cur_frame) + ".png", frame);
		}
		if (cur_frame >= frame_num)
			break;
		++cur_frame;
	}
	cout << "��ȡ�����." << endl;
}

/*
 * ����ͼƬ�����������Ļ������true��ͬʱ�Ѵ�����ͼƬ����outputImg
 * �����������Ļ������false
 */
bool SubtitleExtraction(const Mat& inputImg, Mat& outputImg) {
	Mat dst_img, temp, show;
	/*�ҶȻ�*/
	cvtColor(inputImg, temp, CV_RGB2GRAY);
	Imadjust(temp, 50, 200, 0, 255, 1);

	/*��̬ѧ������*/
	Mat ele = getStructuringElement(MORPH_ELLIPSE, Size(12, 12));
	morphologyEx(temp, dst_img, MORPH_OPEN, ele);
	/*���*/
	dst_img = temp - dst_img;

	/*�и�*/
	dst_img = dst_img.rowRange(dst_img.rows * 3 / 4, dst_img.rows);
	/*��ֵ��*/
	threshold(dst_img, show, 230, 255, THRESH_BINARY);
	
	/*��̬ѧ������ʹ��Ļ������ͨ*/
	ele = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
	morphologyEx(show, dst_img, MORPH_OPEN, ele);
	morphologyEx(dst_img, dst_img, MORPH_CLOSE, ele);

	ele = getStructuringElement(MORPH_ELLIPSE, Size(1, 150));
	morphologyEx(dst_img, dst_img, MORPH_CLOSE, ele);
	ele = getStructuringElement(MORPH_ELLIPSE, Size(30, 5));
	morphologyEx(dst_img, dst_img, MORPH_CLOSE, ele);

	/*��ֵ��*/
	threshold(dst_img, dst_img, 200, 255, THRESH_BINARY);
	/*��̬ѧ�ղ�����ȥ��һЩ���*/
	ele = getStructuringElement(MORPH_ELLIPSE, Size(4, 4));
	morphologyEx(dst_img, dst_img, MORPH_OPEN, ele);

	bool has_caption = false;
	vector<vector<Point>> contours;
	/*��ȡ����ͨ����*/
	findContours(dst_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	int size = contours.size();
	int width, height, x, y;
	vector<Rect> boundRect(size);

	/*����������ͨ����*/
	for (int i = 0; i < size; ++i) {
		/*��ͨ�������Ӿ���*/
		boundRect[i] = boundingRect(Mat(contours[i]));
		width = boundRect[i].width + 20;
		height = boundRect[i].height + 10;
		x = boundRect[i].x - 10;
		y = boundRect[i].y - 10;

		/*��ֹԽ��*/
		if (y < 0)
			y = 0;
		if (y + height > dst_img.rows)
			height -= 10;
		if (x < 0)
			x = 0;
		if (x + width > dst_img.cols)
			width -= 20;

		/*��Ļ��������*/
		if (abs(x + width / 2 - dst_img.cols / 2) < 50 && abs(dst_img.rows - y - height) < 30 && width*height >= 4000) {
			has_caption = true;
			cout << width*height << endl;
			show(Rect(x, y, width, height)).copyTo(dst_img(Rect(x, y, width, height)));
		}
		else {
			/*��������Ļ������Ϳ�ɺ�ɫ*/
			dst_img(Rect(x, y, width, height)) = 0;
		}
	}

	/*��������ͼƬ����outputImg*/
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