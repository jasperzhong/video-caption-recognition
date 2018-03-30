#include <iostream>
#include <cv.h>
#include <opencv2\opencv.hpp>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

using namespace std;
using namespace cv;

/*ȫ��*/
queue<Mat> mat_queue;
mutex mtx;
condition_variable cond;
int frame_num;
volatile int ok = 1;
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
	frame_num = capture.get(CV_CAP_PROP_FRAME_COUNT);
	Mat frame;
	int cur_frame = 0;
	bool on = true;
	while (on) {
		on = capture.read(frame);
		if (cur_frame % 20 == 0) {
			std::lock_guard<std::mutex> lck(mtx);
			mat_queue.push(frame);
			cond.notify_one();
		}
		++cur_frame;
	}
	ok = 0;
	cond.notify_one();
}

/*
 * ����ͼƬ�����������Ļ������true��ͬʱ�Ѵ�����ͼƬ����outputImg
 * �����������Ļ������false
 */
bool SubtitleLocating(Mat& input_img, Mat& output_img) {
	Mat temp;
	/*�ҶȻ�*/ 
	cvtColor(input_img, temp, CV_RGB2GRAY);     
	Imadjust(temp, 50, 200, 0, 255, 1);

	/*��̬ѧ������*/
	Mat ele = getStructuringElement(MORPH_ELLIPSE, Size(12, 12));
	morphologyEx(temp, output_img, MORPH_OPEN, ele);
	/*���*/
	output_img = temp - output_img;
	/*��ֵ��*/
	threshold(output_img, output_img, 230, 255, THRESH_BINARY);
	temp = output_img.clone();	
	
	/*��̬ѧ������ʹ��Ļ������ͨ*/
	ele = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
	morphologyEx(output_img, output_img, MORPH_OPEN, ele);
	morphologyEx(output_img, output_img, MORPH_CLOSE, ele);

	ele = getStructuringElement(MORPH_ELLIPSE, Size(1, 150));
	morphologyEx(output_img, output_img, MORPH_CLOSE, ele);
	ele = getStructuringElement(MORPH_ELLIPSE, Size(30, 5));
	morphologyEx(output_img, output_img, MORPH_CLOSE, ele);

	/*��ֵ��*/
	threshold(output_img, output_img, 200, 255, THRESH_BINARY);
	/*��̬ѧ�ղ�����ȥ��һЩ���*/
	ele = getStructuringElement(MORPH_ELLIPSE, Size(4, 4));
	morphologyEx(output_img, output_img, MORPH_OPEN, ele);

	bool has_caption = false;
	vector<vector<Point>> contours;
	/*��ȡ����ͨ����*/
	findContours(output_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
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
		if (y + height > output_img.rows)
			height -= 10;
		if (x < 0)
			x = 0;
		if (x + width > output_img.cols)
			width -= 20;

		/*��Ļ��������*/
		if (abs(x + width / 2 - output_img.cols / 2) < 50 && abs(output_img.rows - y - height) < 30 && width*height >= 4000) {
			has_caption = true;
			temp(Rect(x, y, width, height)).copyTo(output_img(Rect(x, y, width, height)));
		}
		else {
			/*��������Ļ������Ϳ�ɺ�ɫ*/
			output_img(Rect(x, y, width, height)) = 0;
		}
	}

	return has_caption;
}


/*
 * ����������ĻͼƬ���������ƶȣ�����������0.5����ô��Ϊ����Ļ��ͬ
 * ��֮��Ϊ��Ļ��ͬ
 */
double SubtitleSimilarity(const Mat& a, const Mat& b) {
	double dot = a.dot(b);
	double norm1 = norm(a, CV_L2);
	double norm2 = norm(b, CV_L2);
	return dot / norm1 / norm2;
}

/*
 * ��Ļ��ȡ,����ÿ��ͼƬ�ĳߴ���ͬ
 */
void SubtitleExtraction(const string save_path) {
	Mat input_img;
	bool has_caption;
	int row, col;
	double cosine; 

	unique_lock<mutex> lck(mtx);
	cond.wait(lck, [] {return mat_queue.size() != 0 || !ok; });
	if (!ok) {
		return;
	}
	input_img = mat_queue.front();
	mat_queue.pop();
	row = input_img.rows;
	col = input_img.cols;
	/*��ʼ��*/
	Mat output_img(Size(col,row/4),CV_8U), pre_img(Size(col, row / 4), CV_8U);

	/*����ÿ��ͼƬ*/
	int i = 0;
	cond.wait(lck, [] {return mat_queue.size() != 0 || !ok; });
	while(ok) {
		input_img = mat_queue.front();
		mat_queue.pop();
		input_img = input_img.rowRange(row * 3 / 4, row);
		has_caption = SubtitleLocating(input_img, output_img);
		
		if (has_caption) {
			cosine = SubtitleSimilarity(pre_img, output_img);
			if (cosine < 0.5) {
				cout << "save" << i << endl;
				imwrite(save_path + to_string(i++) + ".png", output_img);
				pre_img = output_img.clone();
			}
		}
		cond.wait(lck, [] {return mat_queue.size() != 0 || !ok; });
	}
}

/* argv����Ƶ��·���ͱ���λ��·��*/
int main(int argc, char** argv)
{
	
	if (argc != 3) {
		cerr << "Wrong Command." << endl;
		cout << "the format is: $ ./run.exe  your_vedio_path  subtitle_save_path" << endl;
		exit(EXIT_FAILURE);
	}

	VideoCapture capture(argv[1]);
	if (!capture.isOpened()) {
		cerr << "Cannot open the vedio." << endl;
		exit(EXIT_FAILURE);
	}
	thread produce(VedioFrameExtraction, std::ref(capture));
	thread comsume(SubtitleExtraction, argv[2]);

	produce.join();
	comsume.join();

	cout << "Done." << endl;
	system("pause");
	return 0;
}