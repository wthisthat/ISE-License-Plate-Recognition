#include <iostream>
#include "highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "core/core.hpp"
#include <sstream>
#include <iomanip>
#include "tesseract/baseapi.h"
#include "leptonica/allheaders.h"

using namespace cv;
using namespace std;


Mat RGBtoGrey(Mat RGB)
{
	Mat GreyImg = Mat::zeros(RGB.size(), CV_8UC1);
	for (int i = 0; i < RGB.rows; i++) {
		for (int j = 0; j < RGB.cols * 3; j = j + 3) {
			GreyImg.at<uchar>(i, j / 3) = (RGB.at<uchar>(i, j) + RGB.at<uchar>(i, j + 1) + RGB.at<uchar>(i, j + 2)) / 3;
		}
	}
	return GreyImg;
}

Mat GreytoBin(Mat Grey, int threshold) {
	Mat BinImg = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = 0; i < Grey.rows; i++) {
		for (int j = 0; j < Grey.cols; j++) {
			if (Grey.at<uchar>(i, j) >= threshold) {
				BinImg.at<uchar>(i, j) = 255;
			}
		}
	}
	return BinImg;
}

Mat inversion(Mat Grey) {
	Mat InvImg = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = 0; i < Grey.rows; i++) {
		for (int j = 0; j < Grey.cols; j++) {
			InvImg.at<uchar>(i, j) = 255 - Grey.at<uchar>(i, j);
		}
	}
	return InvImg;
}

Mat avgConvolutional(Mat Grey, int neighbours) {//neighbour as window size
	Mat avgConvImg = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = neighbours; i < Grey.rows - neighbours; i++) {
		for (int j = neighbours; j < Grey.cols - neighbours; j++) {
			int sum = 0;
			for (int ii = -neighbours; ii <= neighbours; ii++) {
				for (int jj = -neighbours; jj <= neighbours; jj++) {
					sum = sum + Grey.at<uchar>(i + ii, j + jj);
				}
			}
			avgConvImg.at<uchar>(i, j) = sum / ((2 * neighbours + 1) * (2 * neighbours + 1));
		}

	}
	return avgConvImg;
}

Mat equalizeHist(Mat Grey) {
	Mat ehImg = Mat::zeros(Grey.size(), CV_8UC1);
	int count[256] = { 0 };
	float prob[256] = { 0.0 };
	float accProb[256] = { 0.0 };
	for (int i = 0; i < Grey.rows; i++) {
		for (int j = 0; j < Grey.cols; j++) {
			count[Grey.at<uchar>(i, j)]++;			//count
		}
	}
	for (int c = 0; c < 256; c++) {
		prob[c] = (float)count[c] / (float)(Grey.cols * Grey.rows); //probability
	}
	accProb[0] = prob[0];
	for (int p = 1; p < 256; p++) {
		accProb[p] = prob[p] + accProb[p - 1];
	}
	int newPixels[256] = { 0 };
	for (int p = 0; p < 256; p++) {
		newPixels[p] = (256 - 1) * accProb[p]; // acc probability
	}
	for (int i = 0; i < Grey.rows; i++) {
		for (int j = 0; j < Grey.cols; j++) {
			ehImg.at<uchar>(i, j) = newPixels[Grey.at<uchar>(i, j)];
		}
	}
	return ehImg;
}

Mat edgeDetection(Mat Img, int value) {
	Mat edgeImg = Mat::zeros(Img.size(), CV_8UC1);
	int avgL, avgR;
	for (int i = 1; i < Img.rows - 1; i++) {
		for (int j = 1; j < Img.cols - 1; j++) {
			avgL = (Img.at<uchar>(i - 1, j - 1) + Img.at<uchar>(i, j - 1) + Img.at<uchar>(i + 1, j - 1)) / 3;
			avgR = (Img.at<uchar>(i - 1, j + 1) + Img.at<uchar>(i, j + 1) + Img.at<uchar>(i + 1, j + 1)) / 3;
			if (abs(avgL - avgR) > value) {
				edgeImg.at<uchar>(i, j) = 255;
			}
		}
	}
	return edgeImg;
}

Mat dilation(Mat edgeImg, int neighbours) {
	Mat dilationImg = Mat::zeros(edgeImg.size(), CV_8UC1);
	for (int i = neighbours; i < edgeImg.rows - neighbours; i++) {
		for (int j = neighbours; j < edgeImg.cols - neighbours; j++)
			for (int ii = -neighbours; ii <= neighbours; ii++) {
				for (int jj = -neighbours; jj <= neighbours; jj++) {
					if (edgeImg.at<uchar>(ii + i, jj + j) == 255) {
						dilationImg.at<uchar>(i, j) = 255;
					}
				}
			}
	}
	return dilationImg;
}

Mat erosion(Mat edgeImg, int neighbours) {
	Mat erosionImg = Mat::zeros(edgeImg.size(), CV_8UC1);
	for (int i = neighbours; i < edgeImg.rows - neighbours; i++) {
		for (int j = neighbours; j < edgeImg.cols - neighbours; j++) {
			erosionImg.at<uchar>(i, j) = edgeImg.at<uchar>(i, j);//copy the img as default is all black
			for (int ii = -neighbours; ii <= neighbours; ii++) {
				for (int jj = -neighbours; jj <= neighbours; jj++) {
					if (edgeImg.at<uchar>(i + ii, j + jj) == 0) {
						erosionImg.at<uchar>(i, j) = 0;
					}
				}
			}
		}
	}
	return erosionImg;
}

int otsu(Mat Grey, int th) {
	int count[256] = { 0 };
	float prob[256] = { 0.0 };
	float accProb[256] = { 0.0 };
	float meu[256] = { 0.0 };
	float sigma[256] = { 0.0 };

	for (int i = 0; i < Grey.rows; i++) {
		for (int j = 0; j < Grey.cols; j++) {
			count[Grey.at<uchar>(i, j)]++;			//count
		}
	}
	for (int c = 0; c < 256; c++) {
		prob[c] = (float)count[c] / (float)(Grey.cols * Grey.rows); //probability
	}
	accProb[0] = prob[0];
	for (int p = 1; p < 256; p++) {
		accProb[p] = prob[p] + accProb[p - 1];
	} // =theta
	meu[0] = 0;
	for (int i = 1; i < 256; i++) {
		meu[i] = meu[i - 1] + prob[i] * i;
	}

	// calculate meu acc i * acc prob i
	for (int i = 0; i < 256; i++) {
		sigma[i] = pow((meu[255] * accProb[i] - meu[i]), 2) / (accProb[i] * (1 - accProb[i]));

	}
	//find i which maximize sigma
	int OtsuVal = 0;
	int MaxSigma = -1;
	for (int i = 0; i < 256; i++)
	{
		if (sigma[i] > MaxSigma)
		{
			MaxSigma = sigma[i];
			OtsuVal = i;

		}
	}
	return (OtsuVal + th);
}

struct Left_right_contour_sorter // 'less' for contours
{
	bool operator ()(const vector<Point>& a, const vector<Point>& b)
	{
		Rect ra(boundingRect(a));
		Rect rb(boundingRect(b));
		return (rb.x > ra.x);
	}
};

struct Top_bottom_contour_sorter // 'less' for contours
{
	bool operator ()(const vector<Point>& a, const vector<Point>& b)
	{
		Rect ra(boundingRect(a));
		Rect rb(boundingRect(b));
		return (rb.y > ra.x);
	}
};

Mat preciseCrop(Mat dilateddst, Mat binaryPlate) {
	Mat blob;
	blob = dilateddst.clone();
	vector<vector<Point>> contours1;
	vector<Vec4i> hierarchy1;
	Scalar white = CV_RGB(255, 255, 255);
	findContours(dilateddst, contours1, hierarchy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
	Mat dst = Mat::zeros(dilateddst.size(), CV_8UC1);
	if (!contours1.empty()) {
		for (int i = 0; i < contours1.size(); i++) {
			Scalar colour((rand() & 255), rand() & 255, (rand() & 255));
			drawContours(dst, contours1, i, white, -1, 8, hierarchy1);
		}
	}
	//imshow("dst", dst);
	Mat plate = Mat::zeros(dilateddst.size(), CV_8UC1);
	Rect BlobRect;
	Scalar black = CV_RGB(0, 0, 0);
	for (int j = 0; j < contours1.size(); j++)
	{
		BlobRect = boundingRect(contours1[j]);
		if (BlobRect.width < 100) {
			drawContours(blob, contours1, j, black, -1, 8, hierarchy1);
		}
		else {
			plate = binaryPlate(BlobRect);
		}
	}
	return(plate);
}

Mat binaryNoiseFilter(Mat binaryPlate, int fincount, double cropY1, double cropY2, double cropX1, double cropX2) {
	Mat blob2;
	blob2 = binaryPlate.clone();
	vector<vector<Point>> contours2;
	vector<Vec4i> hierarchy2;
	findContours(blob2, contours2, hierarchy2, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
	Mat dst2 = Mat::zeros(binaryPlate.size(), CV_8UC3);
	if (!contours2.empty()) {
		for (int i = 0; i < contours2.size(); i++) {
			Scalar colour((rand() & 255), rand() & 255, (rand() & 255));
			drawContours(dst2, contours2, i, colour, -1, 8, hierarchy2);
		}
	}


	Rect BlobRect2;
	Scalar black2 = CV_RGB(0, 0, 0);
	for (int j = 0; j < contours2.size(); j++)
	{
		BlobRect2 = boundingRect(contours2[j]);
		if (BlobRect2.width / BlobRect2.height > 1 || BlobRect2.y<binaryPlate.rows * cropY1 || BlobRect2.y>binaryPlate.rows * cropY2 || BlobRect2.x<binaryPlate.cols * cropX1 || BlobRect2.x>binaryPlate.cols * cropX2) {
			drawContours(dst2, contours2, j, black2, -1, 8, hierarchy2);
		}
	}

	return dst2;
}

void characterExtraction(Mat binaryPlate, int fincount, double cropY1, double cropY2, double cropX1, double cropX2, bool topBottomPlate) {
	Mat blob2;
	blob2 = binaryPlate.clone();
	vector<vector<Point>> contours2;
	vector<Vec4i> hierarchy2;
	findContours(blob2, contours2, hierarchy2, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
	if (topBottomPlate == true) {
		sort(contours2.begin(), contours2.end(), Left_right_contour_sorter());
		sort(contours2.begin(), contours2.end(), Top_bottom_contour_sorter());
	}
	else {
		sort(contours2.begin(), contours2.end(), Left_right_contour_sorter());
	}
	Mat dst2 = Mat::zeros(binaryPlate.size(), CV_8UC3);
	if (!contours2.empty()) {
		for (int i = 0; i < contours2.size(); i++) {
			Scalar colour((rand() & 255), rand() & 255, (rand() & 255));
			drawContours(dst2, contours2, i, colour, -1, 8, hierarchy2);
		}
	}

	Rect BlobRect2;
	Scalar black2 = CV_RGB(0, 0, 0);
	Scalar white = CV_RGB(255, 255, 255);
	Mat singlechar;
	string disp;
	string filename;
	string outText;
	Mat padded;
	int top, bottom, left, right;
	int borderType = BORDER_CONSTANT;

	for (int j = 0; j < contours2.size(); j++)
	{
		BlobRect2 = boundingRect(contours2[j]);
		if (BlobRect2.width * BlobRect2.height < 10 || BlobRect2.width / BlobRect2.height>1 || BlobRect2.y<binaryPlate.rows * cropY1 || BlobRect2.y>binaryPlate.rows * cropY2 || BlobRect2.x<binaryPlate.cols * cropX1 || BlobRect2.x>binaryPlate.cols * cropX2) {
			drawContours(dst2, contours2, j, black2, -1, 8, hierarchy2);
		}
		else {
			Mat invChar;
			singlechar = binaryPlate(BlobRect2);
			invChar = inversion(singlechar);
			top = (int)(0.1 * invChar.rows); bottom = top;
			left = (int)(0.1 * invChar.cols); right = left;
			copyMakeBorder(invChar, padded, top, bottom, left, right, borderType, white);
			stringstream num;
			num << setw(2) << setfill('0') << j;
			disp = "img fincount=" + to_string(fincount) + "_" + num.str();
			//namedWindow(disp);
			//imshow(disp, padded);
			string name = "\\" + disp + ".bmp";
			filename = "C:\\Users\\edmun\\OneDrive\\Desktop\\ISE\\Car Plate Characters\\car" + to_string(fincount) + name;
			imwrite(filename, padded);
		}
	}
}

void characterOCR(int fincount) {
	tesseract::TessBaseAPI* ocr = new tesseract::TessBaseAPI();
	string outText = "null";
	vector<String> charImg;
	String pattern2 = "C:\\Users\\edmun\\OneDrive\\Desktop\\ISE\\Car Plate Characters\\car" + to_string(fincount);
	glob(pattern2, charImg);
	cout << charImg.size();
	for (int i = 0; i < charImg.size(); i++)
	{
		Mat charImages = imread(charImg[i]);
		//imshow("charImages", charImages);
		ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
		ocr->SetPageSegMode(tesseract::PSM_SINGLE_CHAR);
		ocr->SetVariable("tessedit_char_whitelist", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ");
		ocr->SetImage(charImages.data, charImages.cols, charImages.rows, 3, charImages.step);
		outText = string(ocr->GetUTF8Text());
		cout << "char:" << outText;
		ocr->End();
		waitKey();

	}
}


int main()
{
	//RGB to grey, equalize histogram, blur, edge detection
	int car = 0;
	vector<String> images;
	string outText = "null";
	tesseract::TessBaseAPI* ocr = new tesseract::TessBaseAPI();
	String pattern = "C:\\Users\\edmun\\OneDrive\\Desktop\\ISE\\Car Images";
	glob(pattern, images);

	for (int i = 0; i < images.size(); i++)
	{
		int trialNum = 0;
		Mat img = imread(images[i]);
		cout << images[i] << endl;
		int edgeVal = 48;
		int erosionVal = 1;
		int dilationVal = 10;
		int filterWidth = 90;
		int filterHeight = 35;
		imshow("img", img);
		Mat greyImg = RGBtoGrey(img);
		//imshow("greyImg", greyImg);
		Mat ehImg = equalizeHist(greyImg); // increase contrast
		//imshow("ehimg", ehImg);
		Mat blurImg = avgConvolutional(ehImg, 1); // reduce noises
		//imshow("blurImg", blurImg);

	start:
		Mat edge = edgeDetection(blurImg, edgeVal);
		//imshow("edgeImg", edge);
		Mat erusionImg = erosion(edge, erosionVal);
		//imshow("erusionImg", erusionImg);
		Mat dilationImg = dilation(erusionImg, dilationVal); //remove gaps
		//imshow("dilationImg", dilationImg);



		//segmentation
		Mat blob;
		blob = dilationImg.clone();
		vector<vector<Point>> contours1;
		vector<Vec4i> hierarchy1;
		findContours(dilationImg, contours1, hierarchy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
		Mat dst = Mat::zeros(dilationImg.size(), CV_8UC3);
		if (!contours1.empty()) {
			for (int i = 0; i < contours1.size(); i++) {
				Scalar colour((rand() & 255), rand() & 255, (rand() & 255));
				drawContours(dst, contours1, i, colour, -1, 8, hierarchy1);
			}
		}
		//imshow("dst", dst);
		
		//draw color segments
		Mat plate = Mat::zeros(dilationImg.size(), CV_8UC1);
		int plateFoundCount = 0;
		Rect BlobRect;
		Scalar black = CV_RGB(0, 0, 0);
		for (int j = 0; j < contours1.size(); j++)
		{
			BlobRect = boundingRect(contours1[j]);
			if (BlobRect.width < filterWidth || BlobRect.height < filterHeight || BlobRect.y<greyImg.rows * 0.1 || BlobRect.y>greyImg.rows * 0.9 || BlobRect.x < greyImg.cols * 0.1 || BlobRect.x > greyImg.cols * 0.9 || BlobRect.height > 90) {
				drawContours(blob, contours1, j, black, -1, 8, hierarchy1);
			}
			else {
				plateFoundCount++;
				plate = greyImg(BlobRect);
			}
		}
		trialNum++;
		cout << "platefound" << plateFoundCount << endl;
		cout << "trialNum" << trialNum << endl;
		if ((plateFoundCount == 0 || plateFoundCount > 1) && trialNum == 1)
		{
			edgeVal = 60;
			erosionVal = 0;
			dilationVal = 7;
			filterHeight = 39;
			filterWidth = 97;
			cout << 1 << endl;
			goto start;
		}


		else if ((plateFoundCount == 0 || plateFoundCount > 1) && trialNum == 2)
		{
			edgeVal = 54;
			erosionVal = 1;
			dilationVal = 13;
			filterHeight = 35;
			filterWidth = 65;
			cout << 2 << endl;
			goto start;
		}

		else if ((plateFoundCount == 0 || plateFoundCount > 1) && trialNum == 3)
		{
			edgeVal = 20;
			erosionVal = 0;
			dilationVal = 10; // edge detection after erosion, cannot detect plate
			cout << 3 << endl;
			goto start;
		}

		car++;
		double cropX1, cropX2, cropY1, cropY2;
		double sizeX = 3, sizeY = 3;
		cropX1 = 0.1;
		cropX2 = 0.9;
		cropY1 = 0.1;
		cropY2 = 0.9;
		cout << "car" << car << endl;

		int ocrTrial = 0;
		int otsuTh = 115; //default:115
		Mat Rimg;
		resize(plate, Rimg, Size(), sizeX, sizeY);
		int otsuV = otsu(Rimg, otsuTh);

	start2:
		ocrTrial++;
		otsuV = otsu(Rimg, otsuTh);

		Mat binPlate = GreytoBin(Rimg, otsuV);
		//imshow("bin", binPlate);
		Mat cPlate = binaryNoiseFilter(binPlate, car, cropY1, cropY2, cropX1, cropX2);
		imshow("cPlate", cPlate);

		Mat greyCPlate = RGBtoGrey(cPlate);
		//imshow("greyFINAL", greyCPlate);
		Mat edgeCPlate = edgeDetection(greyCPlate, 20);
		//imshow("edgeFINAL", edgeCPlate);
		Mat dilatedCPlate = dilation(edgeCPlate, 17);
		//imshow("dilateFINAL", dilatedCPlate);
		Mat finalPlate = preciseCrop(dilatedCPlate, binPlate);
		imshow("FINAL", finalPlate);
		ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
		ocr->SetVariable("tessedit_char_whitelist", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ");
		ocr->SetPageSegMode(tesseract::PSM_AUTO);
		ocr->SetImage(finalPlate.data, finalPlate.cols, finalPlate.rows, 1, finalPlate.step);
		outText = string(ocr->GetUTF8Text());
		outText.erase(remove(outText.begin(), outText.end(), ' '), outText.end());
		outText.erase(remove(outText.begin(), outText.end(), '\n'), outText.end());
		ocr->End();

		if (car == 4)
		{
			otsuTh = 30;
			otsuV = otsu(Rimg, otsuTh);
			binPlate = GreytoBin(Rimg, otsuV * 1.2);
			characterExtraction(binPlate, car, cropY1, cropY2, cropX1, cropX2, false);
			characterOCR(car);
		}
		else if (car == 5) 
		{
			otsuTh = 30;
			otsuV = otsu(Rimg, otsuTh);
			binPlate = GreytoBin(Rimg, otsuV * 1.42);
			characterExtraction(binPlate, car, cropY1, cropY2, 0.2, 0.7, true);
			characterOCR(car);
		}
		else if (car == 9)
		{
			otsuTh = 30;
			otsuV = otsu(Rimg, otsuTh);
			binPlate = GreytoBin(Rimg, otsuV * 1.28);
			characterExtraction(binPlate, car, cropY1, cropY2, cropX1, cropX2, true);
			characterOCR(car);
		}
		else if (car == 10) 
		{
			otsuTh = 30;
			otsuV = otsu(Rimg, otsuTh);
			binPlate = GreytoBin(Rimg, otsuV * 1.36);
			characterExtraction(binPlate, car, cropY1, cropY2, cropX1, cropX2, false);
			characterOCR(car);
		}
		else if (car == 15)
		{
			otsuTh = 30;
			otsuV = otsu(Rimg, otsuTh);
			binPlate = GreytoBin(Rimg, otsuV * 1.45);
			characterExtraction(binPlate, car, cropY1, cropY2, cropX1, cropX2, false);
			characterOCR(car);
		}
		else {
			if ((outText.length() < 6 || outText.length() > 7) && ocrTrial == 1)
			{
				otsuTh = 60;
				cout << "1" << endl;
				goto start2;
			}
			else if ((outText.length() < 6 || outText.length() > 7) && ocrTrial == 2)
			{
				otsuTh = 65;
				cout << "2" << endl;
				goto start2;
			}
			else if ((outText.length() < 6 || outText.length() > 7) && ocrTrial == 3)
			{
				otsuTh = 110;
				cout << "3" << endl;
				goto start2;
			}
			else if ((outText.length() < 6 || outText.length() > 7) && ocrTrial == 4)
			{
				otsuTh = 30;
				cout << "4" << endl;
				goto start2;
			}
			else if ((outText.length() < 6 || outText.length() > 7) && ocrTrial == 5)
			{
				otsuTh = 45;
				cout << "5" << endl;
				goto start2;
			}
			cout << "Detected Plate " << i + 1 << ": " << outText << endl;
		}
		waitKey();
	}
}