// main.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>
#include<conio.h>           // it may be necessary to change or remove this line if not using Windows

#include "Blob.h"

#define SHOW_STEPS            // un-comment or comment this line to show steps or not

using namespace std;
using namespace cv;

// global variables ///////////////////////////////////////////////////////////////////////////////
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

// function prototypes ////////////////////////////////////////////////////////////////////////////
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob>& existingBlobs, std::vector<Blob>& currentFrameBlobs);
void addBlobToExistingBlobs(Blob& currentFrameBlob, std::vector<Blob>& existingBlobs, int& intIndex);
void addNewBlob(Blob& currentFrameBlob, std::vector<Blob>& existingBlobs);
double distanceBetweenPoints(cv::Point point1, cv::Point point2);
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName);
void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName);
bool checkIfBlobsCrossedTheLine(std::vector<Blob>& blobs, int& intHorizontalLinePosition, int& carCount);
void drawBlobInfoOnImage(std::vector<Blob>& blobs, cv::Mat& imgFrame2Copy);
void drawCarCountOnImage(int& carCount, cv::Mat& imgFrame2Copy);

///////////////////////////////////////////////////////////////////////////////////////////////////
int main(void) {

    cv::VideoCapture capVideo;

    cv::Mat imgFrame1;
    cv::Mat imgFrame2;

    std::vector<Blob> blobs;

    cv::Point crossingLine[2];

    int carCount = 0;

    capVideo.open("C://208//CarsDrivingUnderBridge.mp4");

    if (!capVideo.isOpened()) {                                                 // if unable to open video file
        std::cout << "error reading video file" << std::endl << std::endl;      // show error message
        _getch();                   // it may be necessary to change or remove this line if not using Windows
        return(0);                                                              // and exit program
    }

    if (capVideo.get(CAP_PROP_FRAME_COUNT) < 2) {
        std::cout << "error: video file must have at least two frames";
        _getch();                   // it may be necessary to change or remove this line if not using Windows
        return(0);
    }

    capVideo.read(imgFrame1);
    capVideo.read(imgFrame2);

    int intHorizontalLinePosition = (int)std::round((double)imgFrame1.rows * 0.35); 
    
    //РАСЧЕТ ГОРИЗОНТАЛЬНОЙ ЛИНИИ ПЕРЕСЕЧЕНИЯ
    //ОПРЕДЕЛЯЕМ ЭТУ ЛИНИЮ КАК ДВЕ ТОЧКИ

    crossingLine[0].x = 0;
    crossingLine[0].y = intHorizontalLinePosition;

    crossingLine[1].x = imgFrame1.cols - 1;
    crossingLine[1].y = intHorizontalLinePosition;

    char chCheckForEscKey = 0;

    bool blnFirstFrame = true;

    int frameCount = 2;

    while (capVideo.isOpened() && chCheckForEscKey != 27) {

        std::vector<Blob> currentFrameBlobs;

        cv::Mat imgFrame1Copy = imgFrame1.clone();
        cv::Mat imgFrame2Copy = imgFrame2.clone();

        cv::Mat imgDifference;
        cv::Mat imgThresh;

        cv::cvtColor(imgFrame1Copy, imgFrame1Copy, COLOR_BGR2GRAY);
        cv::cvtColor(imgFrame2Copy, imgFrame2Copy, COLOR_BGR2GRAY);

        cv::GaussianBlur(imgFrame1Copy, imgFrame1Copy, cv::Size(5, 5), 0);
        cv::GaussianBlur(imgFrame2Copy, imgFrame2Copy, cv::Size(5, 5), 0);

        cv::absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);

        cv::threshold(imgDifference, imgThresh, 30, 255.0, THRESH_BINARY);

        cv::imshow("imgThresh", imgThresh);

        cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        cv::Mat structuringElement15x15 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));

        for (unsigned int i = 0; i < 2; i++) {
            cv::dilate(imgThresh, imgThresh, structuringElement5x5);
            cv::dilate(imgThresh, imgThresh, structuringElement5x5);
            cv::erode(imgThresh, imgThresh, structuringElement5x5);
        }

        cv::Mat imgThreshCopy = imgThresh.clone();

        std::vector<std::vector<cv::Point> > contours;

        cv::findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        drawAndShowContours(imgThresh.size(), contours, "imgContours");

        std::vector<std::vector<cv::Point> > convexHulls(contours.size());

        for (unsigned int i = 0; i < contours.size(); i++) {
            cv::convexHull(contours[i], convexHulls[i]);
        }

        drawAndShowContours(imgThresh.size(), convexHulls, "imgConvexHulls");

        for (auto& convexHull : convexHulls) {    //ВЫБРАСЫВАЕМ ВСЕ ВЫПУКЛЫЕ КАПЛИ, КОТОРЫЕ НАМ НЕ ПОДХОДЯТ
            Blob possibleBlob(convexHull);

            if (//possibleBlob.currentBoundingRect.area() > 400 && //ПЛОЩАЛЬ
                possibleBlob.dblCurrentAspectRatio > 0.2 &&
                possibleBlob.dblCurrentAspectRatio < 4.0 &&      //СООТНОШЕНИЕ СТОРОН
                possibleBlob.currentBoundingRect.width > 30 &&  //ШИРИНА
                possibleBlob.currentBoundingRect.height > 30 && //ВЫСОТА
                possibleBlob.dblCurrentDiagonalSize > 60.0 &&   //ДИАГОНАЛЬ
                (cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.50) {
                currentFrameBlobs.push_back(possibleBlob);
                //cout << possibleBlob.currentBoundingRect.area() << "    "<< possibleBlob.currentBoundingRect.width << "    " << possibleBlob.currentBoundingRect.height<<endl;
            }
        }

        drawAndShowContours(imgThresh.size(), currentFrameBlobs, "imgCurrentFrameBlobs");//РИСУЕМ КАПЛИ КОТОРЫЕ ОСТАЛИСЬ

        if (blnFirstFrame == true) {        //ЕСЛИ ЭТО НАШ ПЕРВЫЙ КАДР ПРОСТО ДОБАВЛЯЕМ ВСЕ КАПЛИ В НАШ ТЕКУЩИЙ СПИСОК КАПЕЛЬ
            for (auto& currentFrameBlob : currentFrameBlobs) {
                blobs.push_back(currentFrameBlob);
            }
        }
        else {
            matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs);    //ЕСЛИ У НАС НЕСКОЛЛЬКО КАПЕЛЬ ИЗ ПРЕДЫДУЩИХ КАДРОВ
        }

        //currentFrameBlobs - ТЕКУЩИЙ НАБОР КАДРОВ
        //blobs - ПРЕДЫДУЩИЙ НАБОР КАДРОВ
        //  matchCurrentFrameBlobsToExistingBlobs - СОПОСТАВЛЕНИЕ КАПЛИ ТЕКУЩЕГО КАДРА С СУЩЕСТВУЮЩИМИ КАПЛЯМИ

        drawAndShowContours(imgThresh.size(), blobs, "imgBlobs"); //РИСУЕМ КПЛИ ПОСЛЕ СОПОСТАВЛЕНИЯ 


        imgFrame2Copy = imgFrame2.clone();          // get another copy of frame 2 since we changed the previous frame 2 copy in the processing above

        drawBlobInfoOnImage(blobs, imgFrame2Copy);

        bool blnAtLeastOneBlobCrossedTheLine = checkIfBlobsCrossedTheLine(blobs, intHorizontalLinePosition, carCount);

        if (blnAtLeastOneBlobCrossedTheLine == true) {
            cv::line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_GREEN, 2);
        }
        else {
            cv::line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_RED, 2);
        }

        drawCarCountOnImage(carCount, imgFrame2Copy);

        cv::imshow("imgFrame2Copy", imgFrame2Copy);

        //cv::waitKey(0);                 // uncomment this line to go frame by frame for debugging

        // now we prepare for the next iteration

        currentFrameBlobs.clear();

        imgFrame1 = imgFrame2.clone();           // move frame 1 up to where frame 2 is

        if ((capVideo.get(CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CAP_PROP_FRAME_COUNT)) {
            capVideo.read(imgFrame2);
        }
        else {
            std::cout << "end of video\n";
            break;
        }

        blnFirstFrame = false;
        frameCount++;
        chCheckForEscKey = cv::waitKey(1);
    }

    if (chCheckForEscKey != 27) {               // if the user did not press esc (i.e. we reached the end of the video)
        cv::waitKey(0);                         // hold the windows open to allow the "end of video" message to show
    }
    // note that if the user did press esc, we don't need to hold the windows open, we can simply let the program end which will close the windows

    return(0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob>& existingBlobs, std::vector<Blob>& currentFrameBlobs) {

    for (auto& existingBlob : existingBlobs) {

        existingBlob.blnCurrentMatchFoundOrNewBlob = false;

        existingBlob.predictNextPosition();
    }

    for (auto& currentFrameBlob : currentFrameBlobs) {

        int intIndexOfLeastDistance = 0;
        double dblLeastDistance = 100000.0; //ПОРОГОВОЕ ЗНАЧЕНИЕ РАСТОЯНИЯ ДО ПРОГНОЗИРОВАННОГО ОБЪЕКТА

        for (unsigned int i = 0; i < existingBlobs.size(); i++) {

            if (existingBlobs[i].blnStillBeingTracked == true) {

                double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);

                if (dblDistance < dblLeastDistance) {
                    dblLeastDistance = dblDistance;
                    intIndexOfLeastDistance = i;
                }
            }
        }

        if (dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize * 0.5) {  //ЕСЛИ ЭТО РАСТОЯНИЕ МЕНЬШЕ ЧЕМ ДИАГОНАЛЬ КАПЛИ
            addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfLeastDistance);//ТО У НАС ЕСТЬ СОВПАДЕНИЕ С ТАКОЙ ТОЧКОЙ
        }
        else {
            addNewBlob(currentFrameBlob, existingBlobs);
        }

    }

    for (auto& existingBlob : existingBlobs) { //ЕСЛИ МЫ НЕ НАШЛИ ТЕКУЩЕГО СОВПАДЕНИЯ ИЛИ ЭТО НОВАЯ КАПЛЯ

        if (existingBlob.blnCurrentMatchFoundOrNewBlob == false) { //ТО УСТАНОВИМ FOLSE
            existingBlob.intNumOfConsecutiveFramesWithoutAMatch++; //ЗАТЕМ УВЕЛИЧИВАЕМ КОЛИЧЕСТВО КАДРОВ БЕЗ СОВПАДНЕИЯ
        }
        //ЕСЛИ КОЛИЧЕСТВО ПОСЛЕДОВАТЕЛЬНЫХ КАДРОВ БЕЗ СОВПАДЕНИЯ ПРЕВЫШАЕТ 5, ТО ЕГО БОЛЬШЕ НЕ ОТСЛЕЖИВАЕМ
        if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 5) {
            existingBlob.blnStillBeingTracked = false;
        }

    }

}


//ФУНКЦИЯ ДОБАВЛЕНИЯ БОЛЬШОГО ОБЪЕКТА К СУЩЕСТВУЮЩЕЙ КАПЛИ
///////////////////////////////////////////////////////////////////////////////////////////////////
void addBlobToExistingBlobs(Blob& currentFrameBlob, std::vector<Blob>& existingBlobs, int& intIndex) {

    existingBlobs[intIndex].currentContour = currentFrameBlob.currentContour;
    existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;

    existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());

    existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
    existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;

    existingBlobs[intIndex].blnStillBeingTracked = true;
    existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
}
//СОЗДАНИЕ НОВОГО ОБЪКТА
///////////////////////////////////////////////////////////////////////////////////////////////////
void addNewBlob(Blob& currentFrameBlob, std::vector<Blob>& existingBlobs) {

    currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;

    existingBlobs.push_back(currentFrameBlob);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
double distanceBetweenPoints(cv::Point point1, cv::Point point2) { //ДИСТАНИЦЯ МЕЖДУ ТОЧКАМИ (ТЕОРЕМА ПИФАГОРА)

    int intX = abs(point1.x - point2.x);
    int intY = abs(point1.y - point2.y);

    return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

//РИСОВАНИЕ И ОТОБРАЖЕНИЕ КОНТУРОВ
// ЗАВИСИТ ЧТО МЫ ПЕРЕЛАЕМ
// ПЕРЕДАЕМ ФАКТИЧЕСКИЕ КОНТУРЫ
///////////////////////////////////////////////////////////////////////////////////////////////////
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName) {
    cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

    cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

    cv::imshow(strImageName, image);
}
// ПЕРЕДАЕМ ВЕКТОР КАПЕЛЬ
///////////////////////////////////////////////////////////////////////////////////////////////////
void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName) {

    cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

    std::vector<std::vector<cv::Point> > contours;

    for (auto& blob : blobs) {
        if (blob.blnStillBeingTracked == true) {
            contours.push_back(blob.currentContour);
        }
    }

    cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

    cv::imshow(strImageName, image);
}
//ПЕРЕСЕКЛА ЛИ КАПЛЯ ЛИНИЮ
///////////////////////////////////////////////////////////////////////////////////////////////////
bool checkIfBlobsCrossedTheLine(std::vector<Blob>& blobs, int& intHorizontalLinePosition, int& carCount) {
    bool blnAtLeastOneBlobCrossedTheLine = false;

    for (auto blob : blobs) {
        //ЕСЛИ В ПРЕДЫДУЩЕМ КАДРЕ КАПЛЯ БЫЛА НИЖЕ 
        if (blob.blnStillBeingTracked == true && blob.centerPositions.size() >= 2) {
            int prevFrameIndex = (int)blob.centerPositions.size() - 2;
            int currFrameIndex = (int)blob.centerPositions.size() - 1;
            //ЕСЛИ В ПРЕДЫДУЩЕМ КАДРЕ КАПЛЯ БЫЛА НИЖЕ ЛИНИИ                              //ЕСЛИ В СЛЕДУЮЩЕМ КАДРЕ КАПЛЯ НАД ЛИНИЕЙ
            if (blob.centerPositions[prevFrameIndex].y > intHorizontalLinePosition && blob.centerPositions[currFrameIndex].y <= intHorizontalLinePosition) {
                carCount++;
                blnAtLeastOneBlobCrossedTheLine = true;
            }
        }

    }

    return blnAtLeastOneBlobCrossedTheLine;//ЕСЛИ КАПЛЯ ПЕРЕСЕКЛА ЛИНИЮ ВОЗВРАЩАЕМ TRUE ДЛЯ РРИСОВОВАНИЯ ЗЕЛЕНОЙ ЛИНИИ
}

//РИСОВАНИЕ ИНФОРМАЦИИ НА ИЗОБРАЖЕНИЕ
///////////////////////////////////////////////////////////////////////////////////////////////////
void drawBlobInfoOnImage(std::vector<Blob>& blobs, cv::Mat& imgFrame2Copy) {

    for (unsigned int i = 0; i < blobs.size(); i++) {

        if (blobs[i].blnStillBeingTracked == true) {
            cv::rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, SCALAR_RED, 2);

            int intFontFace = FONT_HERSHEY_SIMPLEX;
            double dblFontScale = blobs[i].dblCurrentDiagonalSize / 60.0;
            int intFontThickness = (int)std::round(dblFontScale * 1.0);

            cv::putText(imgFrame2Copy, std::to_string(i), blobs[i].centerPositions.back(), intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);
        }
    }
}
//РИСУЕМ КОЛИЧЕСТВО АВТОМОБИЛЕЙ ПЕРЕСЕЧЕННЫХ ЛИНИЮ
///////////////////////////////////////////////////////////////////////////////////////////////////
void drawCarCountOnImage(int& carCount, cv::Mat& imgFrame2Copy) {

    int intFontFace = FONT_HERSHEY_SIMPLEX;
    double dblFontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 300000.0;
    int intFontThickness = (int)std::round(dblFontScale * 1.5);

    cv::Size textSize = cv::getTextSize(std::to_string(carCount), intFontFace, dblFontScale, intFontThickness, 0);

    cv::Point ptTextBottomLeftPosition;

    ptTextBottomLeftPosition.x = imgFrame2Copy.cols - 1 - (int)((double)textSize.width * 1.25);
    ptTextBottomLeftPosition.y = (int)((double)textSize.height * 1.25);

    cv::putText(imgFrame2Copy, std::to_string(carCount), ptTextBottomLeftPosition, intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);

}
