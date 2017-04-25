#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\objdetect.hpp>
#include <opencv2\face.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;

#define IMAGE_SIZE 200

void detectAndDisplay(Mat &frame, Ptr<FaceRecognizer> &model);
Ptr<FaceRecognizer> trainer();

string face_cascade_name = "C:\\OpenCV32\\build\\install\\etc\\haarcascades\\haarcascade_frontalface_default.xml";

CascadeClassifier face_cascade;

int main(int argc, char** argv)
{

	if (!face_cascade.load(face_cascade_name))
	{
		cout << "Error face_cascade" << endl;
		system("pause");
		return -1;
	}


	// Read the video stream
	VideoCapture capture(0);
	if (!capture.isOpened())
	{
		cout << "Video stream not read!" << endl;
		system("pause");
		return -1;
	}

	// Train
	Ptr<FaceRecognizer> model = trainer();

	// Starting reading frame
	Mat frame;
	while (capture.read(frame))
	{
		if (frame.empty())
		{
			cout << "Empty frame!" << endl;
			break;
		}

		detectAndDisplay(frame, model);

		int c = waitKey(30);
		if ((char)c == 27)
		{
			break;
		}
	}

	return 0;
}

void detectAndDisplay(Mat &frame, Ptr<FaceRecognizer> &model)
{
	vector<Rect> faces;
	Mat frame_gray;

	string username;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	//equalizeHist(frame_gray, frame_gray);

	// Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.5);

	for (size_t i = 0; i < faces.size(); ++i)
	{
		Rect face = faces[i];

		rectangle(frame, Point(face.x, face.y), Point(face.x + face.width, face.y + face.height), Scalar(255, 0, 255), 2);
		
		// Who I am
		Mat testImage = frame_gray;
		resize(frame_gray, testImage, Size(IMAGE_SIZE, IMAGE_SIZE));
		int predictedLabel = model->predict(testImage);
		switch (predictedLabel)
		{
		case 0:
			username = "Ozan";
			break;
		case 1:
			username = "Alkan";
			break;
		case 2:
			username = "Obama";
			break;
		default:
			username = "Taniyamadim ki!";
			break;
		}
		putText(frame, username, cvPoint(faces[i].x, faces[i].y + faces[i].height + 30), CV_FONT_HERSHEY_PLAIN, 2, Scalar::all(255), 2, CV_AA);
	}

	// Showing
	imshow("Pencere", frame);
}

Ptr<FaceRecognizer> trainer()
{
	/*in this two vector we put the images and labes for training*/
	vector<Mat> images;
	vector<int> labels;

	for (size_t i = 1; i < 30; i++)
	{
		string s = to_string(i);
		Mat image = imread("dataset\\ozan\\" + s + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		resize(image, image, Size(IMAGE_SIZE, IMAGE_SIZE));
		images.push_back(image); labels.push_back(0);
	}

	for (size_t i = 1; i < 12; i++)
	{
		string s = to_string(i);
		Mat image = imread("dataset\\alkan\\" + s + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		resize(image, image, Size(IMAGE_SIZE, IMAGE_SIZE));
		images.push_back(image); labels.push_back(1);
	}

	for (size_t i = 1; i < 8; i++)
	{
		string s = to_string(i);
		Mat image = imread("dataset\\obama\\" + s + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		resize(image, image, Size(IMAGE_SIZE, IMAGE_SIZE));
		images.push_back(image); labels.push_back(2);
	}


	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
	model->train(images, labels);

	return model;
}