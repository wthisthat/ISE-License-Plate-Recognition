// Lab1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

//#include <iostream>
//#include "highgui/highgui.hpp"
//#include "opencv2/opencv.hpp"
//#include "core/core.hpp"
//#include "tesseract/baseapi.h"
//#include "leptonica/allheaders.h"
//
//using namespace cv;
//using namespace std;
//
//
////function RGB to Grey , *return Grey Image
//Mat RGBtoGrey(Mat RGB) {
//    Mat GreyImg = Mat::zeros(RGB.size(), CV_8UC1);
//    for (int i = 0; i < RGB.rows; i++) {
//        for (int j = 0; j < RGB.cols * 3; j = j + 3) {
//            GreyImg.at<uchar>(i, j/3) = (RGB.at<uchar>(i, j) + RGB.at<uchar>(i, j + 1)+ RGB.at<uchar>(i, j + 2)) / 3;
//        }
//    }
//    return GreyImg;
//}
//
////function Grey to binary , *return binary Image
//Mat GreytoBin(Mat Grey, int threshold) {
//    Mat BinImg = Mat::zeros(Grey.size(), CV_8UC1);
//    for (int i = 0; i < Grey.rows;i++) {
//        for (int j = 0; j < Grey.cols;j++) {
//            if (Grey.at<uchar>(i, j) > threshold)
//                BinImg.at<uchar>(i, j) = 255;
//        }
//    }
//    return BinImg;
//}
//
//Mat InvertedImg(Mat Grey) {
//    Mat InvImg = Mat::zeros(Grey.size(), CV_8UC1);
//    for (int i = 0; i < Grey.rows;i++) {
//        for (int j = 0; j < Grey.cols;j++) {
//            InvImg.at<uchar>(i, j) = 255 - Grey.at<uchar>(i, j);
//        }
//    }
//    return InvImg;
//}
//
//Mat StepFunc(Mat Grey, int th1, int th2) {
//    Mat StepImg = Mat::zeros(Grey.size(), CV_8UC1);
//    for (int i = 0; i < Grey.rows; i++){
//        for (int j = 0; j < Grey.cols; j++)
//        {
//            if (Grey.at<uchar>(i,j) >= th1 && Grey.at<uchar>(i, j) <= th2)
//            {
//                StepImg.at<uchar>(i, j) = 255;
//            }
//        }
//    }
//    return StepImg;
//}
//
//Mat DarkenFunc(Mat Grey, int th) {
//    Mat DarkenImg = Mat::zeros(Grey.size(), CV_8UC1);
//    for (int i = 0; i < Grey.rows; i++){
//        for (int j = 0; j < Grey.cols; j++){
//            if (Grey.at<uchar>(i, j) >= th)
//                DarkenImg.at<uchar>(i, j) = th;
//            else
//                DarkenImg.at<uchar>(i, j) = Grey.at<uchar>(i, j);
//        }
//    }
//    return DarkenImg;
//}
//
////Exclude border
//Mat MaxConvolution(Mat Grey) {
//    Mat MaxImg = Mat::zeros(Grey.size(), CV_8UC1);
//    for (int i = 1; i < Grey.rows-1; i++)
//    {
//        for (int j = 1; j < Grey.cols-1; j++)
//        {
//            int max = 0;
//            for (int ii = -1; ii <=1 ; ii++)
//            {
//                for (int jj = -1; jj <=1; jj++)
//                {
//                    if (Grey.at<uchar>(i + ii, j + jj) > max)
//                        max = Grey.at<uchar>(i + ii, j + jj);                   
//                }
//            }
//            MaxImg.at<uchar>(i, j) = max;
//        }
//    }
//    return MaxImg;
//}
//
//Mat MaxFunction(Mat Grey, int neighbour) { //neighbour = window size
//    Mat MaxFuncImg = Mat::zeros(Grey.size(), CV_8UC1);
//    for (int i = neighbour; i < Grey.rows - neighbour; i++)
//    {
//        for (int j = neighbour; j < Grey.cols - neighbour; j++)
//        {
//            int max = 0;
//            for (int ii = -neighbour; ii <= neighbour; ii++)
//            {
//                for (int jj = -neighbour; jj <= neighbour; jj++)
//                {
//                    if (Grey.at<uchar>(i + ii, j + jj) > max)
//                        max = Grey.at<uchar>(i + ii, j + jj);
//                }
//            }
//            MaxFuncImg.at<uchar>(i, j) = max;
//        }
//    }
//    return MaxFuncImg;
//}
//
//Mat AvgBlur(Mat Grey, int neighbour) { //neighbour = window size
//    Mat BlurImg = Mat::zeros(Grey.size(), CV_8UC1);
//    for (int i = neighbour; i < Grey.rows-neighbour; i++)
//    {
//        for (int j = neighbour; j < Grey.cols- neighbour; j++)
//        {
//            int sum = 0;
//            for (int ii = -neighbour; ii <= neighbour; ii++)
//            {
//                for (int jj = -neighbour; jj <= neighbour; jj++)
//                {
//                    sum = sum + Grey.at<uchar>(i + ii, j + jj);
//                }
//            }
//            BlurImg.at<uchar>(i, j) = sum / ((neighbour * 2 + 1)*(neighbour * 2 + 1)); // sum / mask
//        }
//    }
//    return BlurImg;
//}
//
//Mat BinBlur(Mat Binary, int neighbour) { // neighbour = window size
//    Mat BinBlurImg = Mat::zeros(Binary.size(), CV_8UC1);
//    for (int i = neighbour; i < Binary.rows-neighbour; i++)
//    {
//        for (int j = neighbour; j < Binary.cols-neighbour; j++)
//        {
//            int sum = 0;
//            for (int ii = -neighbour; ii <= neighbour; ii++)
//            {
//                for (int jj = -neighbour; jj <= neighbour; jj++)
//                {
//                    sum += Binary.at<uchar>(i + ii, j + jj);
//                }
//            }
//            BinBlurImg.at<uchar>(i, j) = sum / ((neighbour * 2 + 1) * (neighbour * 2 + 1));
//        }
//    }
//    return BinBlurImg;
//}
//
//Mat EqualizeHistogram(Mat Grey) {
//    Mat EHImg = Mat::zeros(Grey.size(), CV_8UC1);
//    int count[256] = { 0 };    //create count array
//    for (int i = 0; i < Grey.rows; i++)
//    {
//        for (int j = 0; j < Grey.cols; j++)
//        {
//            count[Grey.at<uchar>(i, j)]++;      //scan whole img, count element ++
//        }
//    }
//
//    float prob[256] = { 0 };    //create probability array
//    for (int i = 0; i < 256; i++)
//    {
//        prob[i] = (float)count[i] / (float)(Grey.rows*Grey.cols);   //prob= count / pixels
//    }
//
//    float accprob[256] = { 0.0 };   //create accumulative probability array
//    accprob[0] = prob[0];   //declare index 0 before, else array out of bound in for loop
//    for (int i = 1; i < 256; i++)   //loop start from int i = 1
//    {
//        accprob[i] = prob[i] + accprob[i - 1];  //accprob = current prob + (accprob of one index before)
//    }
//
//    int newPixel[256] = { 0 }; //create G-1*accprob array
//    for (int i = 0; i < 256; i++)
//    {
//        newPixel[i] = (256 - 1) * accprob[i];  
//    }
//
//    for (int i = 0; i < Grey.rows; i++)
//    {
//        for (int j = 0; j < Grey.cols; j++)
//        {
//            EHImg.at<uchar>(i, j) = newPixel[Grey.at<uchar>(i, j)]; //new pixel for Equalized img
//        }
//    }
//    
//    return EHImg;
//}
//
//Mat EdgeDetect(Mat Blur, int th) {
//    Mat EdgeImg = Mat::zeros(Blur.size(), CV_8UC1);
//    for (int i = 1; i < Blur.rows-1; i++)
//    {
//        for (int j = 1; j < Blur.cols-1; j++)
//        { 
//            int avgL = (Blur.at<uchar>(i - 1, j - 1) + Blur.at<uchar>(i, j - 1) + Blur.at<uchar>(i + 1, j - 1)) / 3;
//            int avgR = (Blur.at<uchar>(i - 1, j + 1) + Blur.at<uchar>(i, j + 1) + Blur.at<uchar>(i + 1, j + 1)) / 3;
//            if (abs(avgL - avgR) > th) {
//                EdgeImg.at<uchar>(i, j) = 255;
//            }
//        }
//    }
//    return EdgeImg;
//}
//
//Mat SobelVertical(Mat Blur) {
//    Mat SobelImg = Mat::zeros(Blur.size(), CV_8UC1);
//    for (int i = 1; i < Blur.rows - 1; i++)
//    {
//        for (int j = 1; j < Blur.cols - 1; j++)
//        {
//            int sumL = Blur.at<uchar>(i - 1, j - 1)*-1 +Blur.at<uchar>(i, j - 1)*-2 +Blur.at<uchar>(i + 1, j - 1)*-1;
//            int sumR = Blur.at<uchar>(i - 1, j + 1) + Blur.at<uchar>(i, j + 1)*2 + Blur.at<uchar>(i + 1, j + 1);
//            if ((sumL + sumR) < 0)
//                SobelImg.at<uchar>(i, j) = 0;
//            else if ((sumL + sumR) > 255)
//                SobelImg.at<uchar>(i, j) = 255;
//            else
//                SobelImg.at<uchar>(i, j) = (sumL + sumR);
//
//        }
//    }
//    return SobelImg;
//}
//
//Mat SobelHorizontal(Mat Blur) {
//    Mat SobelImg = Mat::zeros(Blur.size(), CV_8UC1);
//    for (int i = 1; i < Blur.rows - 1; i++)
//    {
//        for (int j = 1; j < Blur.cols - 1; j++)
//        {
//            int sumU = Blur.at<uchar>(i - 1, j - 1) * -1 + Blur.at<uchar>(i - 1, j) * -2 + Blur.at<uchar>(i - 1 + 1, j + 1) * -1;
//            int sumD = Blur.at<uchar>(i + 1, j - 1) + Blur.at<uchar>(i + 1, j) * 2 + Blur.at<uchar>(i + 1, j + 1);
//            if ((sumU + sumD) < 0)
//                SobelImg.at<uchar>(i, j) = 0;
//            else if ((sumU + sumD) > 255)
//                SobelImg.at<uchar>(i, j) = 255;
//            else
//                SobelImg.at<uchar>(i, j) = (sumU + sumD);
//
//        }
//    }
//    return SobelImg;
//}
//
//Mat Sobel(Mat Blur) {
//    Mat SobelImg = Mat::zeros(Blur.size(), CV_8UC1);
//    for (int i = 1; i < Blur.rows - 1; i++)
//    {
//        for (int j = 1; j < Blur.cols - 1; j++)
//        {
//            int sumL = Blur.at<uchar>(i - 1, j - 1) * -1 + Blur.at<uchar>(i, j - 1) * -2 + Blur.at<uchar>(i + 1, j - 1) * -1;
//            int sumR = Blur.at<uchar>(i - 1, j + 1) + Blur.at<uchar>(i, j + 1) * 2 + Blur.at<uchar>(i + 1, j + 1);
//            int Gx = sumL + sumR;
//
//            int sumU = Blur.at<uchar>(i - 1, j - 1) * -1 + Blur.at<uchar>(i - 1, j) * -2 + Blur.at<uchar>(i - 1 + 1, j + 1) * -1;
//            int sumD = Blur.at<uchar>(i + 1, j - 1) + Blur.at<uchar>(i + 1, j) * 2 + Blur.at<uchar>(i + 1, j + 1);
//            int Gy = sumU + sumD;
//            int G = abs(Gx) + abs(Gy);
//            if (G > 255)
//                SobelImg.at<uchar>(i, j) = 255;
//            else
//                SobelImg.at<uchar>(i, j) = (G);
//
//        }
//    }
//    return SobelImg;
//}
//
//Mat Dilation(Mat Edge,int neighbour) {
//    Mat DilationImg = Mat::zeros(Edge.size(), CV_8UC1);
//    for (int i = neighbour; i < Edge.rows-neighbour; i++)
//    {
//        for (int j = neighbour; j < Edge.cols-neighbour; j++)
//        {
//            for (int ii = -neighbour; ii <= neighbour; ii++)
//            {
//                for (int jj = -neighbour; jj <= neighbour; jj++)
//                {
//                    if (Edge.at<uchar>(i + ii, j + jj) == 255) {
//                        DilationImg.at<uchar>(i, j) = 255;
//                        break;
//                    }
//                }
//            }
//        }
//    }
//    return DilationImg;
//
//}
//
//Mat Erosion(Mat Edge, int neighbour) {
//    Mat ErosionImg = Mat::zeros(Edge.size(), CV_8UC1);
//    for (int i = neighbour; i < Edge.rows-neighbour; i++)
//    {
//        for (int j = neighbour; j < Edge.cols-neighbour; j++)
//        {
//            ErosionImg.at<uchar>(i, j) = Edge.at<uchar>(i, j);
//            for (int ii = -neighbour; ii <= neighbour; ii++)
//            {
//                for (int jj = -neighbour; jj <= neighbour; jj++)
//                {
//                    if (Edge.at<uchar>(i + ii, j + jj) == 0) {
//                        ErosionImg.at<uchar>(i, j) = 0;
//                        break;
//                    }
//                }
//            }
//        }
//    }
//    return ErosionImg;
//}
//
//int otsu(Mat Grey) {
//    int count[256] = { 0 };    //create count array
//    for (int i = 0; i < Grey.rows; i++)
//    {
//        for (int j = 0; j < Grey.cols; j++)
//        {
//            count[Grey.at<uchar>(i, j)]++;      //scan whole img, count element ++
//        }
//    }
//
//    float prob[256] = { 0 };    //create probability array
//    for (int i = 0; i < 256; i++)
//    {
//        prob[i] = (float)count[i] / (float)(Grey.rows * Grey.cols);   //prob= count / pixels
//    }
//
//    float accprob[256] = { 0.0 };   //create accumulative probability array
//    accprob[0] = prob[0];   //declare index 0 before, else array out of bound in for loop
//    for (int i = 1; i < 256; i++)   //loop start from int i = 1
//    {
//        accprob[i] = prob[i] + accprob[i - 1];  //accprob = current prob + (accprob of one index before)
//    }//theta
//
//    float meu[256] = { 0.0 };   //calculate meu (acc i *prob(i))
//    meu[0] = 0;
//    for (int i = 1; i < 256; i++)
//    {
//        meu[i] = i * prob[i] + meu[i - 1];
//    }
//
//    float sigma[256] = { 0.0 }; //calculate sigma, find i that maximize sigma
//    for (int i = 0; i < 256; i++)
//    {
//        sigma[i] = pow(meu[255] * accprob[i] - meu[i], 2) / (accprob[i] * (1 - accprob[i]));
//    }
//
//    int otsuVal = 0;
//    int MaxSigma = -1;
//    for (int i = 0; i < 256; i++)
//    {
//        if (sigma[i] > MaxSigma) {
//            MaxSigma = sigma[i];
//            otsuVal = i;
//        }
//    }
//    return otsuVal + 30;
//}
//
//Mat PlateRecog(Mat DilatedErosion,Mat Grey, int w, int h) {
//    Mat blob;
//    blob = DilatedErosion.clone();
//    vector<vector<Point>> contours1;
//    vector<Vec4i> hierarchy1;
//    findContours(DilatedErosion, contours1, hierarchy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
//    Mat dst = Mat::zeros(DilatedErosion.size(), CV_8UC3);
//    if (!contours1.empty()) {
//        for (int i = 0;i < contours1.size();i++) {
//            Scalar colour((rand() & 255), rand() & 255, (rand() & 255));
//            drawContours(dst, contours1, i, colour, -1, 8, hierarchy1);
//        }
//    }
//    imshow("dst", dst);
//
//    Mat plate;
//    Rect BlobRect;
//    Scalar black = CV_RGB(0, 0, 0);
//    for (int j = 0; j < contours1.size(); j++)
//    {
//        BlobRect = boundingRect(contours1[j]);
//        if (BlobRect.width < w || BlobRect.height > h || BlobRect.y<Grey.rows * 0.05 || BlobRect.y>Grey.rows * 0.95) {
//            drawContours(blob, contours1, j, black, -1, 8, hierarchy1);
//        }
//        else {
//            plate = Grey(BlobRect);
//        }
//    }
//
//    return plate;
//}
//
//struct Left_right_contour_sorter // 'less' for contours
//{
//    bool operator ()(const vector<Point>& a, const vector<Point>& b)
//    {
//        Rect ra(boundingRect(a));
//        Rect rb(boundingRect(b));
//        return (rb.x > ra.x);
//    }
//};
//
//Mat characterSegmentation(Mat binaryPlate,float cropX1, float cropX2 ,float cropY1,float cropY2) {
//    Mat blob2;
//    blob2 = binaryPlate.clone();
//    vector<vector<Point>> contours2;
//    vector<Vec4i> hierarchy2;
//    findContours(blob2, contours2, hierarchy2, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
//    sort(contours2.begin(), contours2.end(), Left_right_contour_sorter());
//    Mat dst2 = Mat::zeros(binaryPlate.size(), CV_8UC3);
//    if (!contours2.empty()) {
//        for (int i = 0; i < contours2.size(); i++) {
//            Scalar colour((rand() & 255), rand() & 255, (rand() & 255));
//            drawContours(dst2, contours2, i, colour, -1, 8, hierarchy2);
//        }
//    }
//    //string carNum[7] = { "num1","num2","num3","num4","num5","num6","num7" };
//    //Mat plateNum[7] = { Mat::zeros(binaryPlate.size(), CV_8UC1) };
//    //int x = -1;
//    Rect BlobRect2;
//    Scalar black2 = CV_RGB(0, 0, 0);
//
//    for (int j = 0; j < contours2.size(); j++)
//    {
//        BlobRect2 = boundingRect(contours2[j]);
//        double ratio = BlobRect2.width / BlobRect2.height;
//        if (ratio >2 ||BlobRect2.y<binaryPlate.rows*cropY1 || BlobRect2.y>binaryPlate.rows* cropY2 || BlobRect2.x<binaryPlate.cols * cropX1 || BlobRect2.x>binaryPlate.cols * cropX2 ) {
//            drawContours(dst2, contours2, j, black2, -1, 8, hierarchy2);
//        }
//        else {
//            
//        }
//        
//        
//        //else {
//        //    if (x < 7) {
//        //        x += 1;
//        //        plateNum[x] = binaryPlate(BlobRect2);
//        //    }
//        //}
//        ////imshow("blob2", blob2);
//        //for (int i = 0; i < 7; i++) {
//        //    if (!plateNum[i].empty()) {
//        //        imshow(carNum[i], plateNum[i]);
//        //    }
//    }
//    
//    /*Mat Inverteddst2 = Mat::zeros(binaryPlate.size(), CV_8UC1);
//    Inverteddst2 = InvertedImg(dst2);*/
//    imshow("dst2222", dst2);
//    return dst2;
//
//}
//
//
//
//int main()
//{
//    int car = 0;
//    vector<String> images;
//    string outText = "null";
//    tesseract::TessBaseAPI* ocr = new tesseract::TessBaseAPI();
//    String pattern = "C:\\Users\\edmun\\OneDrive\\Desktop\\ISE\\Car Images";
//    glob(pattern, images);
//
//    for (int i = 0; i < images.size(); i++)
//    {
//        int trialNum = 0;
//        Mat img = imread(images[i]);
//        cout << images[i] << endl;
//        int edgeTh = 48;
//        int erosionTh = 1;
//        int dilationTh = 10;
//        int filterWidth = 90;
//        int filterHeight = 35;
//        int blurTh = 1;
//
//        //RGB to Grey
//        Mat GreyImg = RGBtoGrey(img);
//        //imshow("Grey Image", GreyImg);
//    
//        //equalize histogram img
//        Mat EHImg = EqualizeHistogram(GreyImg);
//        //imshow("Equalized Image", EHImg);
//        //blurred equalize histogram img
//     
//    start:
//        Mat BEHImg = AvgBlur(EHImg, blurTh);
//        //imshow("Blurred Equalized Image", BEHImg);
//
//        //edge detection img
//        Mat EdgeImg = EdgeDetect(BEHImg,edgeTh);
//        //imshow("Edge Detection Image", EdgeImg);
//    
//        //erosion img
//        Mat ErosionImg = Erosion(EdgeImg, erosionTh);
//        //imshow("Erosion Image", ErosionImg);
//        //dilated erosion img
//        Mat DilatedErosionImg = Dilation(ErosionImg, dilationTh);
//        //imshow("Dilated Erosion Image", DilatedErosionImg);
//
//        //ddraw contours
//        Mat blob;
//        blob = DilatedErosionImg.clone();
//        vector<vector<Point>> contours1;
//        vector<Vec4i> hierarchy1;
//        findContours(DilatedErosionImg, contours1, hierarchy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
//        Mat dst = Mat::zeros(DilatedErosionImg.size(), CV_8UC3);
//        if (!contours1.empty()) {
//            for (int i = 0; i < contours1.size(); i++) {
//                Scalar colour((rand() & 255), rand() & 255, (rand() & 255));
//                drawContours(dst, contours1, i, colour, -1, 8, hierarchy1);
//            }
//        }
//        imshow("dst", dst);
//
//        //color black
//        Mat plate; //= Mat::zeros(DilatedErosionImg.size(), CV_8UC1);
//        int plateFoundCount = 0;
//        Rect BlobRect;
//        Scalar black = CV_RGB(0, 0, 0);
//        for (int j = 0; j < contours1.size(); j++)
//        {
//            BlobRect = boundingRect(contours1[j]);
//            //float plateRatio = (float)BlobRect.height / (float)BlobRect.width;
//            if (BlobRect.width < filterWidth || BlobRect.height < filterHeight || BlobRect.y<GreyImg.rows * 0.1 || BlobRect.y>GreyImg.rows * 0.9 || BlobRect.x < GreyImg.cols * 0.1 || BlobRect.x > GreyImg.cols * 0.9 || BlobRect.height > 90) {
//                drawContours(blob, contours1, j, black, -1, 8, hierarchy1);
//            }
//            else {
//                plateFoundCount++;
//                plate = GreyImg(BlobRect);
//            }
//        }
//
//        trialNum++;
//        cout << "platefound" << plateFoundCount << endl;
//        cout << "trialNum" << trialNum << endl;
//
//        if ((plateFoundCount == 0 || plateFoundCount > 1) && trialNum == 1)
//        {
//            //near
//            edgeTh = 60;
//            erosionTh = 0;
//            dilationTh = 7;
//            filterHeight = 39;
//            filterWidth = 97;
//            cout << "check1" << endl;
//            goto start;
//        }
//
//        else if ((plateFoundCount == 0 || plateFoundCount > 1) && trialNum == 2)
//        {
//            //square
//            edgeTh = 54;
//            erosionTh = 1;
//            dilationTh = 13;
//            filterHeight = 35;
//            filterWidth = 65;
//            blurTh = 1;
//            cout << "check2" << endl;
//            goto start;
//        }
//
//        
//        else if ((plateFoundCount == 0 || plateFoundCount > 1) && trialNum == 3)
//        {
//            //dark
//            edgeTh = 20;
//            erosionTh = 0;
//            dilationTh = 10; // edge detection after erosion, cannot detect plate
//            blurTh = 1;
//            cout << "check3" << endl;
//            goto start;
//        }
//
//
//        car++;
//        double cropX1 = 0.1, cropY1 = 0.1;
//        double cropX2 = 0.9, cropY2 = 0.9;
//        double sizeX = 1, sizeY = 1;
//        
//        double percentage = 1;
//        cout << "car" << car << endl;
//        
//        if (car == 1 || car == 11) {
//            percentage = 0.95;
//        }
//        else if (car == 7 || car == 18)
//            percentage = 1.4;
//        else if (car == 14)
//            percentage = 1.2;
//        else if (car == 19)
//            percentage = 1.13;
//        imshow("Final Plate", plate);
//        int th = otsu(plate);
//        Mat binaryPlate = GreytoBin(plate, th*percentage); 
//        imshow("binary plate", binaryPlate);
//        Mat dst2 = characterSegmentation(binaryPlate,cropX1,cropX2,cropY1,cropY2);
//       /* ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
//        ocr->SetVariable("tessedit_char_whitelist", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ");
//        ocr->SetPageSegMode(tesseract::PSM_AUTO);
//        Mat Rimg;
//        resize(dst2, Rimg, Size(), 3, 3);
//        imshow("Rimg", Rimg);
//        ocr->SetImage(Rimg.data, Rimg.cols, Rimg.rows, 3, Rimg.step);
//        outText = string(ocr->GetUTF8Text());
//        cout << "Detected car plate:" << outText;
//        ocr->End();*/
//        waitKey();
//    }
//}

//#include <iostream>
//#include "highgui/highgui.hpp"
//#include "opencv2/opencv.hpp"
//#include "core/core.hpp"
//#include <sstream>
//#include <iomanip>
//#include "tesseract/baseapi.h"
//#include "leptonica/allheaders.h"
//
//using namespace cv;
//using namespace std;
//
//
//Mat RGBtoGrey(Mat RGB)
//{
//	Mat GreyImg = Mat::zeros(RGB.size(), CV_8UC1);
//	for (int i = 0; i < RGB.rows; i++) {
//		for (int j = 0; j < RGB.cols * 3; j = j + 3) {
//			GreyImg.at<uchar>(i, j / 3) = (RGB.at<uchar>(i, j) + RGB.at<uchar>(i, j + 1) + RGB.at<uchar>(i, j + 2)) / 3;
//		}
//	}
//	return GreyImg;
//}
//Mat GreytoBin(Mat Grey, int threshold) {
//	Mat BinImg = Mat::zeros(Grey.size(), CV_8UC1);
//	for (int i = 0; i < Grey.rows; i++) {
//		for (int j = 0; j < Grey.cols; j++) {
//			if (Grey.at<uchar>(i, j) >= threshold) {
//				BinImg.at<uchar>(i, j) = 255;
//			}
//		}
//	}
//	return BinImg;
//}
//Mat inversion(Mat Grey) {
//	Mat InvImg = Mat::zeros(Grey.size(), CV_8UC1);
//	for (int i = 0; i < Grey.rows; i++) {
//		for (int j = 0; j < Grey.cols; j++) {
//			InvImg.at<uchar>(i, j) = 255 - Grey.at<uchar>(i, j);
//		}
//	}
//	return InvImg;
//}
//Mat stepFunction(Mat Grey, int th1, int th2) {
//	Mat stepImg = Mat::zeros(Grey.size(), CV_8UC1);
//	for (int i = 0; i < Grey.rows; i++) {
//		for (int j = 0; j < Grey.cols; j++) {
//			if (Grey.at<uchar>(i, j) >= th1 && Grey.at<uchar>(i, j) <= th2) {
//				stepImg.at<uchar>(i, j) = 255;
//			}
//		}
//	}
//	return stepImg;
//}
//Mat darkerFunction(Mat Grey, int th) {
//	Mat darkerImg = Mat::zeros(Grey.size(), CV_8UC1);
//	for (int i = 0; i < Grey.rows; i++) {
//		for (int j = 0; j < Grey.cols; j++) {
//			if (Grey.at<uchar>(i, j) >= th) {
//				darkerImg.at<uchar>(i, j) = th;
//			}
//			else {
//				darkerImg.at<uchar>(i, j) = Grey.at<uchar>(i, j);
//			}
//		}
//	}
//	return darkerImg;
//}
//Mat maxConvolutional(Mat Grey, int neighbours) {//neighbour as window size
//	Mat maxConvImg = Mat::zeros(Grey.size(), CV_8UC1);
//	for (int i = neighbours; i < Grey.rows - neighbours; i++) {
//		for (int j = neighbours; j < Grey.cols - neighbours; j++) {
//			int maxVal = 0;
//			for (int ii = -neighbours; ii <= neighbours; ii++) {
//				for (int jj = -neighbours; jj <= neighbours; jj++) {
//					if (Grey.at<uchar>(i + ii, j + jj) > maxVal) {
//						maxVal = Grey.at<uchar>(i + ii, j + jj);
//					}
//				}
//			}
//			maxConvImg.at<uchar>(i, j) = maxVal;
//		}
//	}
//	return maxConvImg;
//}
//Mat avgConvolutional(Mat Grey, int neighbours) {//neighbour as window size
//	Mat avgConvImg = Mat::zeros(Grey.size(), CV_8UC1);
//	for (int i = neighbours; i < Grey.rows - neighbours; i++) {
//		for (int j = neighbours; j < Grey.cols - neighbours; j++) {
//			int sum = 0;
//			for (int ii = -neighbours; ii <= neighbours; ii++) {
//				for (int jj = -neighbours; jj <= neighbours; jj++) {
//					sum = sum + Grey.at<uchar>(i + ii, j + jj);
//				}
//			}
//			avgConvImg.at<uchar>(i, j) = sum / ((2 * neighbours + 1) * (2 * neighbours + 1));
//		}
//
//	}
//	return avgConvImg;
//}
//Mat equalizeHist(Mat Grey) {
//	Mat ehImg = Mat::zeros(Grey.size(), CV_8UC1);
//	int count[256] = { 0 };
//	float prob[256] = { 0.0 };
//	float accProb[256] = { 0.0 };
//	for (int i = 0; i < Grey.rows; i++) {
//		for (int j = 0; j < Grey.cols; j++) {
//			count[Grey.at<uchar>(i, j)]++;			//count
//		}
//	}
//	for (int c = 0; c < 256; c++) {
//		prob[c] = (float)count[c] / (float)(Grey.cols * Grey.rows); //probability
//	}
//	accProb[0] = prob[0];
//	for (int p = 1; p < 256; p++) {
//		accProb[p] = prob[p] + accProb[p - 1];
//	}
//	int newPixels[256] = { 0 };
//	for (int p = 0; p < 256; p++) {
//		newPixels[p] = (256 - 1) * accProb[p]; // acc probability
//	}
//	for (int i = 0; i < Grey.rows; i++) {
//		for (int j = 0; j < Grey.cols; j++) {
//			ehImg.at<uchar>(i, j) = newPixels[Grey.at<uchar>(i, j)];
//		}
//	}
//	return ehImg;
//}
//Mat edgeDetection(Mat Img, int value) {
//	Mat edgeImg = Mat::zeros(Img.size(), CV_8UC1);
//	int avgL, avgR;
//	for (int i = 1; i < Img.rows - 1; i++) {
//		for (int j = 1; j < Img.cols - 1; j++) {
//			avgL = (Img.at<uchar>(i - 1, j - 1) + Img.at<uchar>(i, j - 1) + Img.at<uchar>(i + 1, j - 1)) / 3;
//			avgR = (Img.at<uchar>(i - 1, j + 1) + Img.at<uchar>(i, j + 1) + Img.at<uchar>(i + 1, j + 1)) / 3;
//			if (abs(avgL - avgR) > value) {
//				edgeImg.at<uchar>(i, j) = 255;
//			}
//		}
//	}
//	return edgeImg;
//}
//Mat sobelDetectionVertical(Mat Img) {
//	Mat sobelImg = Mat::zeros(Img.size(), CV_8UC1);
//	int avgL, avgR;
//	for (int i = 1; i < Img.rows - 1; i++) {
//		for (int j = 1; j < Img.cols - 1; j++) {
//			avgL = (Img.at<uchar>(i - 1, j - 1) * -1 + Img.at<uchar>(i, j - 1) * -2 + Img.at<uchar>(i + 1, j - 1) * -1);
//			avgR = (Img.at<uchar>(i - 1, j + 1) + Img.at<uchar>(i, j + 1) * 2 + Img.at<uchar>(i + 1, j + 1));
//			if ((avgL + avgR) < 0) {
//				sobelImg.at<uchar>(i, j) = 0;
//			}
//			else if ((avgL + avgR) > 255) {
//				sobelImg.at<uchar>(i, j) = 255;
//			}
//			else {
//				sobelImg.at<uchar>(i, j) = (avgL + avgR);
//			}
//		}
//	}
//	return sobelImg;
//}
//Mat sobelDetectionHorizontal(Mat Img) {
//	Mat sobelImg = Mat::zeros(Img.size(), CV_8UC1);
//	int avgL, avgR;
//	for (int j = 1; j < Img.cols - 1; j++) {
//		for (int i = 1; i < Img.rows - 1; i++) {
//			avgL = (-1 * Img.at<uchar>(i - 1, j - 1) + -2 * Img.at<uchar>(i - 1, j) + -1 * Img.at<uchar>(i - 1, j + 1));
//			avgR = (Img.at<uchar>(i + 1, j - 1) + 2 * Img.at<uchar>(i + 1, j) + Img.at<uchar>(i + 1, j + 1));
//			if (avgL + avgR < 0) {
//				sobelImg.at<uchar>(i, j) = 0;
//			}
//			else if (avgL + avgR > 255) {
//				sobelImg.at<uchar>(i, j) = 255;
//			}
//			else {
//				sobelImg.at<uchar>(i, j) = avgL + avgR;
//			}
//		}
//	}
//	return sobelImg;
//}
//Mat sobelMask(Mat Img) {
//	Mat sobelImg = Mat::zeros(Img.size(), CV_8UC1);
//	int sumL, sumR, sumUp, sumDown;
//	for (int i = 1; i < Img.rows - 1; i++) {
//		for (int j = 1; j < Img.cols - 1; j++) {
//			sumL = (Img.at<uchar>(i - 1, j - 1) * -1 + Img.at<uchar>(i, j - 1) * -2 + Img.at<uchar>(i + 1, j - 1) * -1);
//			sumR = (Img.at<uchar>(i - 1, j + 1) + Img.at<uchar>(i, j + 1) * 2 + Img.at<uchar>(i + 1, j + 1));
//		}
//	}
//	return sobelImg;
//}
//Mat dilation(Mat edgeImg, int neighbours) {
//	Mat dilationImg = Mat::zeros(edgeImg.size(), CV_8UC1);
//	for (int i = neighbours; i < edgeImg.rows - neighbours; i++) {
//		for (int j = neighbours; j < edgeImg.cols - neighbours; j++)
//			for (int ii = -neighbours; ii <= neighbours; ii++) {
//				for (int jj = -neighbours; jj <= neighbours; jj++) {
//					if (edgeImg.at<uchar>(ii + i, jj + j) == 255) {
//						dilationImg.at<uchar>(i, j) = 255;
//					}
//				}
//			}
//	}
//	return dilationImg;
//}
//Mat erosion(Mat edgeImg, int neighbours) {
//	Mat erosionImg = Mat::zeros(edgeImg.size(), CV_8UC1);
//	for (int i = neighbours; i < edgeImg.rows - neighbours; i++) {
//		for (int j = neighbours; j < edgeImg.cols - neighbours; j++) {
//			erosionImg.at<uchar>(i, j) = edgeImg.at<uchar>(i, j);//copy the img as default is all black
//			for (int ii = -neighbours; ii <= neighbours; ii++) {
//				for (int jj = -neighbours; jj <= neighbours; jj++) {
//					if (edgeImg.at<uchar>(i + ii, j + jj) == 0) {
//						erosionImg.at<uchar>(i, j) = 0;
//					}
//				}
//			}
//		}
//	}
//	return erosionImg;
//}
//int otsu(Mat Grey, int th) {
//	int count[256] = { 0 };
//	float prob[256] = { 0.0 };
//	float accProb[256] = { 0.0 };
//	float meu[256] = { 0.0 };
//	float sigma[256] = { 0.0 };
//
//	for (int i = 0; i < Grey.rows; i++) {
//		for (int j = 0; j < Grey.cols; j++) {
//			count[Grey.at<uchar>(i, j)]++;			//count
//		}
//	}
//	for (int c = 0; c < 256; c++) {
//		prob[c] = (float)count[c] / (float)(Grey.cols * Grey.rows); //probability
//	}
//	accProb[0] = prob[0];
//	for (int p = 1; p < 256; p++) {
//		accProb[p] = prob[p] + accProb[p - 1];
//	} // =theta
//	meu[0] = 0;
//	for (int i = 1; i < 256; i++) {
//		meu[i] = meu[i - 1] + prob[i] * i;
//	}
//
//	// calculate meu acc i * acc prob i
//	for (int i = 0; i < 256; i++) {
//		sigma[i] = pow((meu[255] * accProb[i] - meu[i]), 2) / (accProb[i] * (1 - accProb[i]));
//
//	}
//	//find i which maximize sigma
//	int OtsuVal = 0;
//	int MaxSigma = -1;
//	for (int i = 0; i < 256; i++)
//	{
//		if (sigma[i] > MaxSigma)
//		{
//			MaxSigma = sigma[i];
//			OtsuVal = i;
//
//		}
//	}
//	return (OtsuVal + th);
//}
//struct Left_right_contour_sorter // 'less' for contours
//{
//	bool operator ()(const vector<Point>& a, const vector<Point>& b)
//	{
//		Rect ra(boundingRect(a));
//		Rect rb(boundingRect(b));
//		return (rb.x > ra.x);
//	}
//};
//struct Top_bottom_contour_sorter // 'less' for contours
//{
//	bool operator ()(const vector<Point>& a, const vector<Point>& b)
//	{
//		Rect ra(boundingRect(a));
//		Rect rb(boundingRect(b));
//		return (rb.y > ra.x);
//	}
//};
//
//Mat plateRecognition(Mat dilationImg, Mat greyImg, int w, int h) {
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
//	Mat plate = Mat::zeros(dilationImg.size(), CV_8UC1);
//	Rect BlobRect;
//	Scalar black = CV_RGB(0, 0, 0);
//	for (int j = 0; j < contours1.size(); j++)
//	{
//		BlobRect = boundingRect(contours1[j]);
//		if (BlobRect.width < w || BlobRect.height>h || BlobRect.y<greyImg.rows * 0.2 || BlobRect.y>greyImg.rows * 0.8 || BlobRect.x > greyImg.cols * 0.75) {
//			drawContours(blob, contours1, j, black, -1, 8, hierarchy1);
//		}
//		else {
//			plate = greyImg(BlobRect);
//		}
//	}
//	imshow("blob", blob);
//	return(plate);
//
//}
//
//Mat preciseCrop(Mat dilateddst,Mat binaryPlate) {
//	Mat blob;
//	blob = dilateddst.clone();
//	vector<vector<Point>> contours1;
//	vector<Vec4i> hierarchy1;
//	Scalar white = CV_RGB(255, 255, 255);
//	findContours(dilateddst, contours1, hierarchy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
//	Mat dst = Mat::zeros(dilateddst.size(), CV_8UC1);
//	if (!contours1.empty()) {
//		for (int i = 0; i < contours1.size(); i++) {
//			Scalar colour((rand() & 255), rand() & 255, (rand() & 255));
//			drawContours(dst, contours1, i, white, -1, 8, hierarchy1);
//		}
//	}
//	imshow("dst", dst);
//	Mat plate = Mat::zeros(dilateddst.size(), CV_8UC1);
//	Rect BlobRect;
//	Scalar black = CV_RGB(0, 0, 0);
//	for (int j = 0; j < contours1.size(); j++)
//	{
//		BlobRect = boundingRect(contours1[j]);
//		if (BlobRect.width <100 ) {
//			drawContours(blob, contours1, j, black, -1, 8, hierarchy1);
//		}
//		else {
//			plate = binaryPlate(BlobRect);
//		}
//	}
//	return(plate);
//}
//
//Mat binaryNoiseFilter(Mat binaryPlate, int fincount, double cropY1, double cropY2, double cropX1, double cropX2) {
//	Mat blob2;
//	blob2 = binaryPlate.clone();
//	vector<vector<Point>> contours2;
//	vector<Vec4i> hierarchy2;
//	findContours(blob2, contours2, hierarchy2, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
//	Mat dst2 = Mat::zeros(binaryPlate.size(), CV_8UC3);
//	if (!contours2.empty()) {
//		for (int i = 0; i < contours2.size(); i++) {
//			Scalar colour((rand() & 255), rand() & 255, (rand() & 255));
//			drawContours(dst2, contours2, i, colour, -1, 8, hierarchy2);
//		}
//	}
//
//	
//	Rect BlobRect2;
//	Scalar black2 = CV_RGB(0, 0, 0);
//	for (int j = 0; j < contours2.size(); j++)
//	{
//		BlobRect2 = boundingRect(contours2[j]);
//		if (BlobRect2.width / BlobRect2.height > 1 || BlobRect2.y<binaryPlate.rows * cropY1 || BlobRect2.y>binaryPlate.rows * cropY2 || BlobRect2.x<binaryPlate.cols * cropX1 || BlobRect2.x>binaryPlate.cols * cropX2) {
//			drawContours(dst2, contours2, j, black2, -1, 8, hierarchy2);
//		}
//	}
//	
//	return dst2;
//}
//
//void characterExtraction(Mat binaryPlate, int fincount, double cropY1, double cropY2, double cropX1, double cropX2, bool topBottomPlate) {
//	Mat blob2;
//	blob2 = binaryPlate.clone();
//	vector<vector<Point>> contours2;
//	vector<Vec4i> hierarchy2;
//	findContours(blob2, contours2, hierarchy2, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
//	if (topBottomPlate == true) {
//		sort(contours2.begin(), contours2.end(), Left_right_contour_sorter());
//		sort(contours2.begin(), contours2.end(), Top_bottom_contour_sorter());
//	}
//	else {
//		sort(contours2.begin(), contours2.end(), Left_right_contour_sorter());
//	}
//	Mat dst2 = Mat::zeros(binaryPlate.size(), CV_8UC3);
//	if (!contours2.empty()) {
//		for (int i = 0; i < contours2.size(); i++) {
//			Scalar colour((rand() & 255), rand() & 255, (rand() & 255));
//			drawContours(dst2, contours2, i, colour, -1, 8, hierarchy2);
//		}
//	}
//
//	Rect BlobRect2;
//	Scalar black2 = CV_RGB(0, 0, 0);
//	Scalar white = CV_RGB(255, 255, 255);
//	Mat singlechar;
//	string disp;
//	string filename;
//	string outText;
//	Mat padded;
//	int top, bottom, left, right;
//	int borderType = BORDER_CONSTANT;
//
//	for (int j = 0; j < contours2.size(); j++)
//	{
//		BlobRect2 = boundingRect(contours2[j]);
//		if (BlobRect2.width * BlobRect2.height < 10 || BlobRect2.width / BlobRect2.height>1 || BlobRect2.y<binaryPlate.rows * cropY1 || BlobRect2.y>binaryPlate.rows * cropY2 || BlobRect2.x<binaryPlate.cols * cropX1 || BlobRect2.x>binaryPlate.cols * cropX2) {
//			drawContours(dst2, contours2, j, black2, -1, 8, hierarchy2);
//		}
//		else {
//			Mat invChar;
//			singlechar = binaryPlate(BlobRect2);
//			invChar = inversion(singlechar);
//			top = (int)(0.1 * invChar.rows); bottom = top;
//			left = (int)(0.1 * invChar.cols); right = left;
//			copyMakeBorder(invChar, padded, top, bottom, left, right, borderType, white);
//			stringstream num;
//			num << setw(2) << setfill('0') << j;
//			disp = "img fincount=" + to_string(fincount) + "_" + num.str();
//			namedWindow(disp);
//			imshow(disp, padded);
//			string name = "\\" + disp + ".bmp";
//			filename = "C:\\Users\\edmun\\OneDrive\\Desktop\\ISE\\Car Plate Characters\\car" + to_string(fincount) + name;
//			imwrite(filename, padded);
//		}
//	}
//}
//
//void characterOCR(int fincount) {
//	tesseract::TessBaseAPI* ocr = new tesseract::TessBaseAPI();
//	string outText = "null";
//	vector<String> charImg;
//	String pattern2 = "C:\\Users\\edmun\\OneDrive\\Desktop\\ISE\\Car Plate Characters\\car" + to_string(fincount);
//	glob(pattern2, charImg);
//	cout << charImg.size();
//	for (int i = 0; i < charImg.size(); i++)
//	{
//		Mat charImages = imread(charImg[i]);
//		//int th = otsu(charImages);
//		//Mat binaryPlate =equalizeHist(charImages);
//		imshow("charImages", charImages);
//		ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
//		ocr->SetPageSegMode(tesseract::PSM_SINGLE_CHAR);
//		ocr->SetVariable("tessedit_char_whitelist", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ");
//		//Mat plateChar=imread()
//		ocr->SetImage(charImages.data, charImages.cols, charImages.rows, 3, charImages.step);
//		//ocr->SetImage(dst2.data, dst2.cols, dst2.rows, 3, dst2.step);
//		outText = string(ocr->GetUTF8Text());
//		cout << "char:" << outText;
//		ocr->End();
//		waitKey();
//
//	}
//}
//
//
//int main()
//{
//	//RGB to grey, equalize histogram, blur, edge detection
//	int fincount = 0;
//	vector<String> images;
//	string outText = "null";
//	tesseract::TessBaseAPI* ocr = new tesseract::TessBaseAPI();
//	String pattern = "C:\\Users\\edmun\\OneDrive\\Desktop\\ISE\\Car Images";
//	glob(pattern, images);
//
//	for (int i = 0; i < images.size(); i++)
//	{
//		int trialNum = 0;
//		Mat img = imread(images[i]);
//		cout << images[i] << endl;
//		int edgeVal = 48;
//		int erosionVal = 1;
//		int dilationVal = 10;
//		int filterWidth = 90;
//		int filterHeight = 35;
//		//Mat img = imread("D:\\Year 2 Sem 2\\ISE\\car images\\car11.jpg");
//		imshow("img", img);
//		Mat greyImg = RGBtoGrey(img);
//		//imshow("greyImg", greyImg);
//		Mat ehImg = equalizeHist(greyImg); // increase contrast
//		//imshow("ehimg", ehImg);
//		Mat blurImg = avgConvolutional(ehImg, 1); // reduce noises
//		//imshow("blurImg", blurImg);
//
//	start:
//		Mat edge = edgeDetection(blurImg, edgeVal);
//		//imshow("edgeImg", edge);
//		Mat erusionImg = erosion(edge, erosionVal);
//		//imshow("erusionImg", erusionImg);
//		Mat dilationImg = dilation(erusionImg, dilationVal); //remove gaps
//		//imshow("dilationImg", dilationImg);
//
//
//
//		//segmentation
//		Mat blob;
//		blob = dilationImg.clone();
//		vector<vector<Point>> contours1;
//		vector<Vec4i> hierarchy1;
//		findContours(dilationImg, contours1, hierarchy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
//		Mat dst = Mat::zeros(dilationImg.size(), CV_8UC3);
//		if (!contours1.empty()) {
//			for (int i = 0; i < contours1.size(); i++) {
//				Scalar colour((rand() & 255), rand() & 255, (rand() & 255));
//				drawContours(dst, contours1, i, colour, -1, 8, hierarchy1);
//			}
//		}
//		imshow("dst", dst);
//		//draw color segments
//
//
//
//
//		Mat plate = Mat::zeros(dilationImg.size(), CV_8UC1);
//		int plateFoundCount = 0;
//		Rect BlobRect;
//		Scalar black = CV_RGB(0, 0, 0);
//		for (int j = 0; j < contours1.size(); j++)
//		{
//			BlobRect = boundingRect(contours1[j]);
//			//float plateRatio = (float)BlobRect.height / (float)BlobRect.width;
//			if (BlobRect.width < filterWidth || BlobRect.height < filterHeight || BlobRect.y<greyImg.rows * 0.1 || BlobRect.y>greyImg.rows * 0.9 || BlobRect.x < greyImg.cols * 0.1 || BlobRect.x > greyImg.cols * 0.9 || BlobRect.height > 90) {
//				drawContours(blob, contours1, j, black, -1, 8, hierarchy1);
//			}
//			else {
//				plateFoundCount++;
//				plate = greyImg(BlobRect);
//			}
//		}
//		trialNum++;
//		cout << "platefound" << plateFoundCount << endl;
//		cout << "trialNum" << trialNum << endl;
//		if ((plateFoundCount == 0 || plateFoundCount > 1) && trialNum == 1)
//		{
//			edgeVal = 60;
//			erosionVal = 0;
//			dilationVal = 7;
//			filterHeight = 39;
//			filterWidth = 97;
//			cout << 1 << endl;
//			goto start;
//		}
//
//
//		else if ((plateFoundCount == 0 || plateFoundCount > 1) && trialNum == 2)
//		{
//			edgeVal = 54;
//			erosionVal = 1;
//			dilationVal = 13;
//			filterHeight = 35;
//			filterWidth = 65;
//			cout << 2 << endl;
//			goto start;
//		}
//
//		else if ((plateFoundCount == 0 || plateFoundCount > 1) && trialNum == 3)
//		{
//			edgeVal = 20;
//			erosionVal = 0;
//			dilationVal = 10; // edge detection after erosion, cannot detect plate
//			cout << 3 << endl;
//			goto start;
//		}
//
//		fincount++;
//		double times = 1.3;
//		double cropX1, cropX2, cropY1, cropY2;
//		double sizeX = 3, sizeY = 3;
//		cropX1 = 0.1;
//		cropX2 = 0.9;
//		cropY1 = 0.1;
//		cropY2 = 0.9;
//		cout << "fincount" << fincount << endl;
//
//		int ocrTrial = 0;
//		int otsuTh = 115; //default:115
//		Mat Rimg;
//		resize(plate, Rimg, Size(), sizeX, sizeY);
//		int otsuV = otsu(Rimg, otsuTh);
//
//	start2:
//		ocrTrial++;
//		otsuV = otsu(Rimg, otsuTh);
//
//		Mat binPlate = GreytoBin(Rimg, otsuV);
//		imshow("bin", binPlate);
//		//int th = otsu(charImages);
//		//Mat binaryPlate =equalizeHist(charImages);
//		//imshow("Final Plate", plateImg);
//		Mat cPlate = binaryNoiseFilter(binPlate, fincount, cropY1, cropY2, cropX1, cropX2);
//		imshow("cPlate", cPlate);
//
//		Mat greyCPlate = RGBtoGrey(cPlate);
//		imshow("greyFINAL", greyCPlate);
//		Mat edgeCPlate = edgeDetection(greyCPlate, 20);
//		imshow("edgeFINAL", edgeCPlate);
//		Mat dilatedCPlate = dilation(edgeCPlate, 17);
//		imshow("dilateFINAL", dilatedCPlate);
//		Mat finalPlate = preciseCrop(dilatedCPlate, binPlate);
//		imshow("FINAL", finalPlate);
//		ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
//		ocr->SetVariable("tessedit_char_whitelist", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ");
//		ocr->SetPageSegMode(tesseract::PSM_AUTO);
//		ocr->SetImage(finalPlate.data, finalPlate.cols, finalPlate.rows, 1, finalPlate.step);
//		//ocr->SetImage(dst2.data, dst2.cols, dst2.rows, 3, dst2.step);
//		outText = string(ocr->GetUTF8Text());
//		outText.erase(remove(outText.begin(), outText.end(), ' '), outText.end());
//		outText.erase(remove(outText.begin(), outText.end(), '\n'), outText.end());
//		//cout << "Detected Plate " << i + 1 << ": " << outText << endl;
//		cout << "text length: " << outText.length() << endl;
//		ocr->End();
//
//		if (fincount == 4)
//		{
//			//238.8 //199 //39.8 //40
//			otsuTh = 30;
//			otsuV = otsu(Rimg, otsuTh);
//			binPlate = GreytoBin(Rimg, otsuV * 1.2);
//			characterExtraction(binPlate, fincount, cropY1, cropY2, cropX1, cropX2, false);
//			characterOCR(fincount);
//		}
//		else if (fincount == 5) //this
//		{
//			//115 //164.45 //49.45
//			otsuTh = 30;
//			otsuV = otsu(Rimg, otsuTh);
//			binPlate = GreytoBin(Rimg, otsuV * 1.42); 
//			imshow("binnn", binPlate);
//			characterExtraction(binPlate, fincount, cropY1, cropY2, 0.2, 0.7, true);
//			characterOCR(fincount);
//		}
//		else if (fincount == 9) //this
//		{
//			//127 //163.83 //36.83
//			otsuTh = 30;
//			otsuV = otsu(Rimg, otsuTh);
//			binPlate = GreytoBin(Rimg, otsuV * 1.28);	
//			characterExtraction(binPlate, fincount, cropY1, cropY2, cropX1, cropX2, true);
//			characterOCR(fincount);
//		}
//		else if (fincount == 10) //this
//		{
//			//145 //197.2 //52.2
//			otsuTh = 30;
//			otsuV = otsu(Rimg, otsuTh);
//			binPlate = GreytoBin(Rimg, otsuV * 1.36);	
//			characterExtraction(binPlate, fincount, cropY1, cropY2, cropX1, cropX2, false);
//			characterOCR(fincount);
//		}
//		else if (fincount == 15)
//		{
//			//153 //221.85 //68.85
//			otsuTh = 30;
//			otsuV = otsu(Rimg, otsuTh);
//			cout << "otsuV: " << otsuV * 1.45 << endl;
//			binPlate = GreytoBin(Rimg, otsuV * 1.45);
//			characterExtraction(binPlate, fincount, cropY1, cropY2, cropX1, cropX2, false);
//			characterOCR(fincount);
//		}
//		else {
//			if ((outText.length() < 6 || outText.length() > 7) && ocrTrial == 1)
//			{
//				otsuTh = 60;
//				cout << "1" << endl;
//				goto start2;
//			}
//			else if ((outText.length() < 6 || outText.length() > 7) && ocrTrial == 2)
//			{
//				otsuTh = 65;
//				cout << "2" << endl;
//				goto start2;
//			}
//			else if ((outText.length() < 6 || outText.length() > 7) && ocrTrial == 3)
//			{
//				otsuTh = 110;
//				cout << "3" << endl;
//				goto start2;
//			}
//			else if ((outText.length() < 6 || outText.length() > 7) && ocrTrial == 4)
//			{
//				otsuTh = 30;
//				cout << "4" << endl;
//				goto start2;
//			}
//			else if ((outText.length() < 6 || outText.length() > 7) && ocrTrial == 5)
//			{
//				otsuTh = 45;
//				cout << "5" << endl;
//				goto start2;
//			}
//			cout << "Detected Plate " << i + 1 << ": " << outText << endl;
//		}
//
//		waitKey();
//	}
//}