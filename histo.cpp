#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <string>
using namespace cv;
using namespace std;

// Functions for facial feature detection
static void detectFaces(Mat&, vector<Rect_<int> >&, string);
static void detectLeftEye(Mat&, vector<Rect_<int> >&, string);
static void detectRightEye(Mat&, vector<Rect_<int> >&, string);
static void detectMouth(Mat&, vector<Rect_<int> >&, string);
static void detectFacialFeaures(Mat& img, const vector<Rect_<int> > faces,
		string left_eye_cascade, string right_eye_cascade,
		string mouth_cascade);
static void createSample(Mat& img, Rect ROI, String type, int filenumber);
string detectEmotion(Mat& img, Rect rect, std::string part);

int MAX_DISTANCE_MOUTH = 250;
int MAX_DISTANCE_EYES = 210;
int MIN_DISTANCE = 0;
int NUMBER_OF_SAMPLES_PER_EMOTION = 71;
int NUMBER_OF_EMOTIONS = 7;
vector<string> emotions;
vector<string> parts;

string input_image_path;
string face_cascade_path, left_eye_cascade_path, right_eye_cascade_path,
		mouth_cascade_path;
static int filenumber = 0;

cv::SiftFeatureDetector detector;
FlannBasedMatcher matcher;

int main(int argc, char** argv) {
	emotions.push_back("Joy");
	emotions.push_back("Sadness");
	emotions.push_back("Anger");
	emotions.push_back("Fear");
	emotions.push_back("Neutral");
	emotions.push_back("Disgust");
	emotions.push_back("Surprise");
	parts.push_back("lefteye");
	parts.push_back("righteye");
	parts.push_back("mouth");

	int x = 1;
	face_cascade_path =
			"/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
	left_eye_cascade_path =
			"/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_lefteye.xml";
	right_eye_cascade_path =
			"/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_righteye.xml";
	mouth_cascade_path =
			"/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml";

	//for (x = 1; x <= NUMBER_OF_SAMPLES_PER_EMOTION; x++) {

	filenumber = x;
	//cout << x << endl;
	input_image_path = "";
	stringstream ssfn;
	ssfn << "face" << ".png";
	//ssfn << "Surprise/" << x << ".jpg";
	input_image_path = ssfn.str();
	// Chargement de l'image
	Mat image;
	image = imread(input_image_path);

	// Detecter les visages
	vector<Rect_<int> > faces;
	detectFaces(image, faces, face_cascade_path);

	//Detecter les yeux et la bouche
	detectFacialFeaures(image, faces, left_eye_cascade_path,
			right_eye_cascade_path, mouth_cascade_path);

	imshow(input_image_path, image);

	waitKey(0);
	//}

	return 0;
}

static void detectFaces(Mat& img, vector<Rect_<int> >& faces,
		string cascade_path) {
	CascadeClassifier face_cascade;
	face_cascade.load(cascade_path);

	face_cascade.detectMultiScale(img, faces, 1.05, 3, 0 | CASCADE_SCALE_IMAGE,
			Size(30, 30));
	return;
}

static void detectFacialFeaures(Mat& img, const vector<Rect_<int> > faces,
		string left_eye_cascade, string right_eye_cascade,
		string mouth_cascade) {
	for (unsigned int i = 0; i < faces.size(); ++i) {
		// Créer rectangle contenant le visage
		Rect face = faces[i];
		//Créer échantillon contenant le visage
		createSample(img, face, "testface", filenumber);

		//Dessiner rectangle autour du visage
		rectangle(img, Point(face.x, face.y),
				Point(face.x + face.width, face.y + face.height),
				Scalar(255, 0, 0), 1, 4);

		// Les yeux et la bouche doivent etre detectés a l'intérieur du visage
		Mat ROI = img(Rect(face.x, face.y, face.width, face.height));

		// Detection des yeux dans la moitié supérieure du visage
		ROI = img(Rect(face.x, face.y, face.width / 2, face.height / 2));

		vector<Rect_<int> > lefteye;
		detectLeftEye(ROI, lefteye, left_eye_cascade);

		Rect le;
		Rect lemax;
		lemax.width = 0;
		for (unsigned int j = 0; j < lefteye.size(); ++j) {
			le = lefteye[j];
			if (le.width > lemax.width) {
				lemax = le;
			}
		}
		//Créer échantillon contenant l'oeil gauche
		createSample(ROI, lemax, "testlefteye", filenumber);

		//Detecter l'emotion associée a l'oeil droit

		string em = detectEmotion(ROI, lemax, "lefteye");
		cout << em << endl;
		putText(img, em, Point(face.x, face.y), FONT_HERSHEY_SIMPLEX, 0.7,
				Scalar(0, 255, 255), 2, 8, false);


		//Dessin (cercle et rectangle)
		circle(ROI,
				Point(lemax.x + lemax.width / 2, lemax.y + lemax.height / 2), 3,
				Scalar(0, 255, 0), -1, 8);
		rectangle(ROI, Point(lemax.x, lemax.y),
				Point(lemax.x + lemax.width, lemax.y + lemax.height),
				Scalar(0, 255, 255), 1, 4);

		ROI = img(
				Rect(face.x + (face.width / 2), face.y, face.width / 2,
						face.height / 2));

		vector<Rect_<int> > righteye;
		detectRightEye(ROI, righteye, left_eye_cascade);

		Rect re;
		Rect remax;
		lemax.width = 0;
		for (unsigned int j = 0; j < righteye.size(); ++j) {
			re = righteye[j];
			if (re.width > remax.width) {
				remax = re;
			}
		}
		//Créer échantillon contenant l'oeil gauche
		createSample(ROI, remax, "testrighteye", filenumber);

		string em2 = detectEmotion(ROI, remax, "righteye");
		cout << em2 << endl;
		putText(img, em2, Point(face.x, face.y + 20), FONT_HERSHEY_SIMPLEX, 0.7,
				Scalar(255, 255, 0), 2, 8, false);

		//Dessin (cercle et rectangle)
		circle(ROI,
				Point(remax.x + remax.width / 2, remax.y + remax.height / 2), 3,
				Scalar(0, 255, 0), -1, 8);
		rectangle(ROI, Point(remax.x, remax.y),
				Point(remax.x + remax.width, remax.y + remax.height),
				Scalar(255, 255, 0), 1, 4);

		// Detection de la bouche depuis la moitié inférieure du visage

		ROI = img(
				Rect(face.x, face.y + (face.height / 2), face.width,
						face.height / 2));

		vector<Rect_<int> > mouth;
		detectMouth(ROI, mouth, mouth_cascade);

		Rect m;
		Rect max;
		max.width = 0;
		max.y = 0;
		for (unsigned int j = 0; j < mouth.size(); ++j) {
			m = mouth[j];
			if (m.width > max.width && m.y > max.y) {
				max = m;
			}
		}
		createSample(ROI, max, "testmouth", filenumber);

		string em3 = detectEmotion(ROI, max, "mouth");
		cout << em3 << endl;
		putText(img, em3, Point(face.x, face.y + 40), FONT_HERSHEY_SIMPLEX, 0.7,
				Scalar(0, 0, 255), 2, 8, false);

		rectangle(ROI, Point(max.x, max.y),
				Point(max.x + max.width, max.y + max.height), Scalar(0, 0, 255),
				1, 4);

	}

	return;
}

static void detectLeftEye(Mat& img, vector<Rect_<int> >& eyes,
		string cascade_path) {
	CascadeClassifier left_eye_cascade;
	left_eye_cascade.load(cascade_path);

	left_eye_cascade.detectMultiScale(img, eyes, 1.05, 6,
			0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	return;
}

static void detectRightEye(Mat& img, vector<Rect_<int> >& eyes,
		string cascade_path) {
	CascadeClassifier right_eye_cascade;
	right_eye_cascade.load(cascade_path);

	right_eye_cascade.detectMultiScale(img, eyes, 1.05, 6,
			0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	return;
}

static void detectMouth(Mat& img, vector<Rect_<int> >& mouth,
		string cascade_path) {
	CascadeClassifier mouth_cascade;
	mouth_cascade.load(cascade_path);

	mouth_cascade.detectMultiScale(img, mouth, 1.05, 6, 0 | CASCADE_SCALE_IMAGE,
			Size(30, 30));
	return;
}
static void createSample(Mat& img, Rect ROI, String folder, int filenumberr) {
	Mat crop = img(ROI);
	//resize(crop, crop, Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images
	cvtColor(crop, crop, CV_BGR2GRAY); // Convert cropped image to Grayscale

	// Form a filename
	string folderName = folder;
	string folderCreateCommand = "mkdir " + folderName;
	system(folderCreateCommand.c_str());

	String filename = "";
	stringstream ssfn;
	ssfn << folderName << "/" << filenumberr << ".jpg";
	filename = ssfn.str();

	imwrite(filename, crop);

	return;
}

string detectEmotion(Mat& ROI, Rect rect, std::string part) {

	int goodMatchesTab[emotions.size()];
	Mat img = ROI(rect);
	cvtColor(img, img, CV_BGR2GRAY);

	vector<cv::KeyPoint> inputKeypoints;
	vector<cv::KeyPoint> sampleKeypoints;

	detector.detect(img, inputKeypoints);

	Mat inputDescriptors;
	if (inputDescriptors.type() != CV_32F) {
		inputDescriptors.convertTo(inputDescriptors, CV_32F);
	}
	detector.compute(img, inputKeypoints, inputDescriptors);

	std::vector<DMatch> matches;

	for (unsigned e = 0; e < emotions.size(); e++) {
		int goodMatches = 0;
		for (int i = 1; i < NUMBER_OF_SAMPLES_PER_EMOTION; i++) {
			String samplePath = "";
			stringstream ssfn;
			ssfn << emotions[e] << part << "/" << i << ".jpg";
			samplePath = ssfn.str();
			Mat sample = imread(samplePath, 1);
			detector.detect(sample, sampleKeypoints);

			Mat sampleDescriptors;
			if (sampleDescriptors.type() != CV_32F) {
				sampleDescriptors.convertTo(sampleDescriptors, CV_32F);
			}
			detector.compute(sample, sampleKeypoints, sampleDescriptors);

			try {
				matcher.match(inputDescriptors, sampleDescriptors, matches);
				for (unsigned j = 0; j < matches.size(); j++) {
					double dist = matches[j].distance;
					//cout << "Distance entre " << i << " et la source est :" << dist << endl;
					if ((dist < MAX_DISTANCE_MOUTH && part=="mouth")||(dist < MAX_DISTANCE_EYES && ((part=="lefteye")||(part=="righteye")))) {
						goodMatches++;    //Count the number of "good matches"
					}
				}
			} catch (exception e) {
			}
		}
		goodMatchesTab[e] = goodMatches;
	}
	int maxGoodMatch = 0;
	int aa = 0;
	for (unsigned a = 0; a < emotions.size(); a++) {
		if (goodMatchesTab[a] > maxGoodMatch) {
			maxGoodMatch = goodMatchesTab[a];
			aa = a;
		}
	}
	if (goodMatchesTab[aa] == 0) {
		return "unidentified";
	} else {
		String res = "";
		stringstream ssf;
		ssf << (goodMatchesTab[aa] * 100) / 70 << "% " << emotions[aa];
		res = ssf.str();
		return res;
	}
}

