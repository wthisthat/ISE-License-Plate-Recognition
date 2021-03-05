#include <iostream>
#include "highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "core/core.hpp"
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
Mat stepFunction(Mat Grey, int th1, int th2) {
	Mat stepImg = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = 0; i < Grey.rows; i++) {
		for (int j = 0; j < Grey.cols; j++) {
			if (Grey.at<uchar>(i, j) >= th1 && Grey.at<uchar>(i, j) <= th2) {
				stepImg.at<uchar>(i, j) = 255;
			}
		}
	}
	return stepImg;
}
Mat darkerFunction(Mat Grey, int th) {
	Mat darkerImg = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = 0; i < Grey.rows; i++) {
		for (int j = 0; j < Grey.cols; j++) {
			if (Grey.at<uchar>(i, j) >= th) {
				darkerImg.at<uchar>(i, j) = th;
			}
			else {
				darkerImg.at<uchar>(i, j) = Grey.at<uchar>(i, j);
			}
		}
	}
	return darkerImg;
}
Mat maxConvolutional(Mat Grey, int neighbours) {//neighbour as window size
	Mat maxConvImg = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = neighbours; i < Grey.rows - neighbours; i++) {
		for (int j = neighbours; j < Grey.cols - neighbours; j++) {
			int maxVal = 0;
			for (int ii = -neighbours; ii <= neighbours; ii++) {
				for (int jj = -neighbours; jj <= neighbours; jj++) {
					if (Grey.at<uchar>(i + ii, j + jj) > maxVal) {
						maxVal = Grey.at<uchar>(i + ii, j + jj);
					}
				}
			}
			maxConvImg.at<uchar>(i, j) = maxVal;
		}
	}
	return maxConvImg;
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
Mat sobelDetectionVertical(Mat Img) {
	Mat sobelImg = Mat::zeros(Img.size(), CV_8UC1);
	int avgL, avgR;
	for (int i = 1; i < Img.rows - 1; i++) {
		for (int j = 1; j < Img.cols - 1; j++) {
			avgL = (Img.at<uchar>(i - 1, j - 1) * -1 + Img.at<uchar>(i, j - 1) * -2 + Img.at<uchar>(i + 1, j - 1) * -1);
			avgR = (Img.at<uchar>(i - 1, j + 1) + Img.at<uchar>(i, j + 1) * 2 + Img.at<uchar>(i + 1, j + 1));
			if ((avgL + avgR) < 0) {
				sobelImg.at<uchar>(i, j) = 0;
			}
			else if ((avgL + avgR) > 255) {
				sobelImg.at<uchar>(i, j) = 255;
			}
			else {
				sobelImg.at<uchar>(i, j) = (avgL + avgR);
			}
		}
	}
	return sobelImg;
}
Mat sobelDetectionHorizontal(Mat Img) {
	Mat sobelImg = Mat::zeros(Img.size(), CV_8UC1);
	int avgL, avgR;
	for (int j = 1; j < Img.cols - 1; j++) {
		for (int i = 1; i < Img.rows - 1; i++) {
			avgL = (-1 * Img.at<uchar>(i - 1, j - 1) + -2 * Img.at<uchar>(i - 1, j) + -1 * Img.at<uchar>(i - 1, j + 1));
			avgR = (Img.at<uchar>(i + 1, j - 1) + 2 * Img.at<uchar>(i + 1, j) + Img.at<uchar>(i + 1, j + 1));
			if (avgL + avgR < 0) {
				sobelImg.at<uchar>(i, j) = 0;
			}
			else if (avgL + avgR > 255) {
				sobelImg.at<uchar>(i, j) = 255;
			}
			else {
				sobelImg.at<uchar>(i, j) = avgL + avgR;
			}
		}
	}
	return sobelImg;
}
Mat sobelMask(Mat Img) {
	Mat sobelImg = Mat::zeros(Img.size(), CV_8UC1);
	int sumL, sumR, sumUp, sumDown;
	for (int i = 1; i < Img.rows - 1; i++) {
		for (int j = 1; j < Img.cols - 1; j++) {
			sumL = (Img.at<uchar>(i - 1, j - 1) * -1 + Img.at<uchar>(i, j - 1) * -2 + Img.at<uchar>(i + 1, j - 1) * -1);
			sumR = (Img.at<uchar>(i - 1, j + 1) + Img.at<uchar>(i, j + 1) * 2 + Img.at<uchar>(i + 1, j + 1));
		}
	}
	return sobelImg;
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
int otsu(Mat Grey) {
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
	return OtsuVal + 30;
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

Mat plateRecognition(Mat dilationImg, Mat greyImg, int w, int h) {
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
	imshow("dst", dst);
	Mat plate = Mat::zeros(dilationImg.size(), CV_8UC1);
	Rect BlobRect;
	Scalar black = CV_RGB(0, 0, 0);
	for (int j = 0; j < contours1.size(); j++)
	{
		BlobRect = boundingRect(contours1[j]);
		if (BlobRect.width < w || BlobRect.height>h || BlobRect.y<greyImg.rows * 0.2 || BlobRect.y>greyImg.rows * 0.8 || BlobRect.x > greyImg.cols * 0.75) {
			drawContours(blob, contours1, j, black, -1, 8, hierarchy1);
		}
		else {
			plate = greyImg(BlobRect);
		}
	}
	imshow("blob", blob);
	return(plate);

}



Mat characterSegmentation(Mat binaryPlate, Mat OgPlate, int fincount, double cropY1, double cropY2, double cropX1, double cropX2) {
	Mat blob2;
	blob2 = binaryPlate.clone();
	vector<vector<Point>> contours2;
	vector<Vec4i> hierarchy2;
	findContours(blob2, contours2, hierarchy2, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
	sort(contours2.begin(), contours2.end(), Left_right_contour_sorter());
	Mat dst2 = Mat::zeros(binaryPlate.size(), CV_8UC3);
	if (!contours2.empty()) {
		for (int i = 0; i < contours2.size(); i++) {
			Scalar colour((rand() & 255), rand() & 255, (rand() & 255));
			drawContours(dst2, contours2, i, colour, -1, 8, hierarchy2);
		}
	}

	int x = -1;
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
	tesseract::TessBaseAPI* ocr = new tesseract::TessBaseAPI();

	for (int j = 0; j < contours2.size(); j++)
	{
		BlobRect2 = boundingRect(contours2[j]);
		//float ratio = (float)BlobRect2.x / (float)BlobRect2.y;
		//
		if (BlobRect2.width / BlobRect2.height > 1 || BlobRect2.y<binaryPlate.rows * cropY1 || BlobRect2.y>binaryPlate.rows * cropY2 || BlobRect2.x<binaryPlate.cols * cropX1 || BlobRect2.x>binaryPlate.cols * cropX2) {
			drawContours(dst2, contours2, j, black2, -1, 8, hierarchy2);
		}
		else {

			Mat invChar;
			singlechar = binaryPlate(BlobRect2);
			invChar = inversion(singlechar);
			top = (int)(0.1 * invChar.rows); bottom = top;
			left = (int)(0.1 * invChar.cols); right = left;
			copyMakeBorder(invChar, padded, top, bottom, left, right, borderType, white);
			disp = "img fincount=" + to_string(fincount) + "_" + to_string(j);
			namedWindow(disp);
			imshow(disp, padded);
			//ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
			//ocr->SetPageSegMode(tesseract::PSM_SINGLE_CHAR);
			//ocr->SetImage(padded.data, padded.cols, padded.rows,3, padded.step);
			////ocr->SetImage(dst2.data, dst2.cols, dst2.rows, 3, dst2.step);
			//outText = string(ocr->GetUTF8Text());
			//cout << "char:" << outText;
			//filename = "C:\\Users\\edmun\\OneDrive\\Desktop\\ISE\\" + disp + ".jpg";
			//imwrite(filename, padded);
			//ocr->End();
			
		}
	}

	imshow("dst2", dst2);
	return dst2;
}

int main()
{ //RGB to grey, equalize histogram, blur, edge detection
	int fincount = 0;
	vector<String> images;
	string outText = "null";
	tesseract::TessBaseAPI* ocr = new tesseract::TessBaseAPI();
	String pattern = "C:\\Users\\edmun\\OneDrive\\Desktop\\ISE\\Car Images";
	glob(pattern, images);

	//for (int i = 0; i < images.size(); i++)
	//{
	//	int trialNum = 0;
	//	Mat img = imread(images[i]);
	//	cout << images[i] << endl;
	//	int edgeVal = 48;
	//	int erosionVal = 1;
	//	int dilationVal = 10;
	//	int filterWidth = 90;
	//	int filterHeight = 35;
	//	//Mat img = imread("D:\\Year 2 Sem 2\\ISE\\car images\\car11.jpg");
	//	imshow("img", img);
	//	Mat greyImg = RGBtoGrey(img);
	//	//imshow("greyImg", greyImg);
	//	Mat ehImg = equalizeHist(greyImg); // increase contrast
	//	//imshow("ehimg", ehImg);
	//	Mat blurImg = avgConvolutional(ehImg, 1); // reduce noises
	//	//imshow("blurImg", blurImg);

	//start:
	//	Mat edge = edgeDetection(blurImg, edgeVal);
	//	//imshow("edgeImg", edge);
	//	Mat erusionImg = erosion(edge, erosionVal);
	//	//imshow("erusionImg", erusionImg);
	//	Mat dilationImg = dilation(erusionImg, dilationVal); //remove gaps
	//	//imshow("dilationImg", dilationImg);



	//	//segmentation
	//	Mat blob;
	//	blob = dilationImg.clone();
	//	vector<vector<Point>> contours1;
	//	vector<Vec4i> hierarchy1;
	//	findContours(dilationImg, contours1, hierarchy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
	//	Mat dst = Mat::zeros(dilationImg.size(), CV_8UC3);
	//	if (!contours1.empty()) {
	//		for (int i = 0; i < contours1.size(); i++) {
	//			Scalar colour((rand() & 255), rand() & 255, (rand() & 255));
	//			drawContours(dst, contours1, i, colour, -1, 8, hierarchy1);
	//		}
	//	}
	//	imshow("dst", dst);
	//	//draw color segments




	//	Mat plate = Mat::zeros(dilationImg.size(), CV_8UC1);
	//	int plateFoundCount = 0;
	//	Rect BlobRect;
	//	Scalar black = CV_RGB(0, 0, 0);
	//	for (int j = 0; j < contours1.size(); j++)
	//	{
	//		BlobRect = boundingRect(contours1[j]);
	//		//float plateRatio = (float)BlobRect.height / (float)BlobRect.width;
	//		if (BlobRect.width < filterWidth || BlobRect.height < filterHeight || BlobRect.y<greyImg.rows * 0.1 || BlobRect.y>greyImg.rows * 0.9 || BlobRect.x < greyImg.cols * 0.1 || BlobRect.x > greyImg.cols * 0.9 || BlobRect.height > 90) {
	//			drawContours(blob, contours1, j, black, -1, 8, hierarchy1);
	//		}
	//		else {
	//			plateFoundCount++;
	//			plate = greyImg(BlobRect);
	//		}
	//	}
	//	trialNum++;
	//	cout << "platefound" << plateFoundCount << endl;
	//	cout << "trialNum" << trialNum << endl;
	//	if ((plateFoundCount == 0 || plateFoundCount > 1) && trialNum == 1)
	//	{
	//		edgeVal = 60;
	//		erosionVal = 0;
	//		dilationVal = 7;
	//		filterHeight = 39;
	//		filterWidth = 97;
	//		cout << 1 << endl;
	//		goto start;
	//	}


	//	else if ((plateFoundCount == 0 || plateFoundCount > 1) && trialNum == 2)
	//	{
	//		edgeVal = 54;
	//		erosionVal = 1;
	//		dilationVal = 13;
	//		filterHeight = 35;
	//		filterWidth = 65;
	//		cout << 2 << endl;
	//		goto start;
	//	}

	//	else if ((plateFoundCount == 0 || plateFoundCount > 1) && trialNum == 3)
	//	{
	//		edgeVal = 20;
	//		erosionVal = 0;
	//		dilationVal = 10; // edge detection after erosion, cannot detect plate
	//		cout << 3 << endl;
	//		goto start;
	//	}
	//	fincount++;
	//	double times = 1.3;
	//	double cropX1, cropX2, cropY1, cropY2;
	//	double sizeX = 3, sizeY = 3;
	//	cropX1 = 0.1;
	//	cropX2 = 0.9;
	//	cropY1 = 0.1;
	//	cropY2 = 0.9;
	//	cout << "fincount" << fincount << endl;
	//	if (fincount == 3) {
	//		cropX1 = 0.4;
	//	}
	//	else if (fincount == 4 || fincount == 19) {
	//		times = 1.1;
	//	}
	//	else if (fincount == 5) {
	//		times = 1.54;
	//	}
	//	else if (fincount == 8) {
	//		cropX1 = 0.6;
	//	}
	//	else if (fincount == 12) {
	//		times = 0.9;
	//	}
	//	else if (fincount == 13) {
	//		times = 1.42;
	//		cropY1 = 0.3;
	//		cropY2 = 0.7;
	//	}
	//	else if (fincount == 1 || fincount == 14 || fincount == 16 || fincount == 17 || fincount == 18)
	//	{
	//		times = 1.42;
	//	}
	//	/*if (fincount == 1) {
	//		times = 1;
	//	}
	//	else if (fincount == 2||fincount==3) {
	//		times = 1.3;
	//	}*/
	//	//else if (fincount == 11) {
	//	//	times=0.95;
	//	//}
	//	//else if (fincount == 12){
	//	//	times = 0.96;
	//	//	cropX1 = 0.1;
	//	//	cropX2 = 0.9;
	//	//	cropY1 = 0.1;
	//	//	cropY2 = 0.85;
	//	//}
	//	//else if (fincount == 6) {
	//	//	times = 1.44;
	//	//}
	//	//else if (fincount == 3||fincount == 4) {
	//	//	times = 1.12;
	//	//}
	//	//else if (fincount == 19) {
	//	//	times = 1.13;
	//	//}
	//	//else if (fincount == 5 ) {
	//	//	times = 1.36;
	//	//	sizeX = 1.2;
	//	//	sizeY = 1.2;
	//	//}
	//	//else if (fincount == 10|| fincount == 7|| fincount == 9) {
	//	//	times = 1.36;
	//	//	sizeX = 1.5;
	//	//	sizeY = 1.5;
	//	//}
	//	////3 and 8 need crop
	//	//else if (fincount == 8 ) {
	//	//	times = 1.16; //0.4 crop
	//	//	cropX1 = 0.4;
	//	//	cropX2 = 0.9;
	//	//	cropY1 = 0.1;
	//	//	cropY2 = 0.9;
	//	//	sizeX = 1.3;
	//	//	sizeY = 1.3;
	//	//}
	//	//else if (fincount == 14 ) {
	//	//	times = 1.2;
	//	//	sizeX = 1.2;
	//	//	sizeY = 1.2;
	//	//}
	//	//else if (fincount == 15) {
	//	//	times = 1.45;
	//	//}
	//	//else if (fincount == 16) {
	//	//	times = 0.97;
	//	//	sizeX = 1.2;
	//	//	sizeY = 1.2;
	//	//}
	//	//else if (fincount == 18) {
	//	//	times = 1.27;
	//	//	sizeX = 1.1;
	//	//	sizeY = 1.1;
	//	//}


	//	Mat Rimg;
	//	resize(plate, Rimg, Size(), sizeX, sizeY);
	//	int th = otsu(Rimg);
	//	Mat binaryP = GreytoBin(Rimg, th);
	//	Mat binaryPlate = GreytoBin(Rimg, th * times);
	//	imshow("binary plate", binaryPlate);

	//	/*String disp = "img " + to_string(fincount);
	//	namedWindow(disp);
	//	String filename = "D:\\Year 2 Sem 2\\ISE\\car" + disp + ".jpg";
	//	imwrite(filename, plate);*/
	//	Mat dst2 = characterSegmentation(binaryPlate, binaryP, fincount, cropX1, cropX2, cropY1, cropY2);
	//	imshow("dst2", dst2);


	//	waitKey();

	//}
	vector<String> charImg;
	String pattern2 = "C:\\Users\\edmun\\OneDrive\\Desktop\\ISE\\no process data\\car16";
	glob(pattern2, charImg);
	for (int i = 0; i < charImg.size(); i++)
	{
		Mat charImages = imread(charImg[i]);
		//int th = otsu(charImages);
		//Mat binaryPlate =equalizeHist(charImages);
		imshow("charImages", charImages);
		ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
		//ocr->SetVariable("tessedit_char_whitelist", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ");
		ocr->SetPageSegMode(tesseract::PSM_SINGLE_CHAR);
		//Mat plateChar=imread()
		ocr->SetImage(charImages.data, charImages.cols, charImages.rows, 3, charImages.step);
		//ocr->SetImage(dst2.data, dst2.cols, dst2.rows, 3, dst2.step);
		outText = string(ocr->GetUTF8Text());
		cout << "char:" << outText;
		ocr->End();
		waitKey();
	}
}