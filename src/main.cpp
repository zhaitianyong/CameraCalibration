#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/QR>
#include "homography.h"
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <opencv2/core/eigen.hpp>
void decomposePMatrix() {
    using namespace Eigen;

    MatrixXd pMatrix(3, 4);
    pMatrix(0, 0) = 3.53553e+2;
    pMatrix(0, 1) = 3.39645e+2;
    pMatrix(0, 2) = 2.77744e+2;
    pMatrix(0, 3) = -1.44946e+6;
    pMatrix(1, 0) = -1.03528e+2;
    pMatrix(1, 1) = 2.33212e+1;
    pMatrix(1, 2) = 4.59607e+2;
    pMatrix(1, 3) = -6.32525e+5;
    pMatrix(2, 0) = 7.07107e-1;
    pMatrix(2, 1) = -3.53553e-1;
    pMatrix(2, 2) = 6.12372e-1;
    pMatrix(2, 3) = -9.18559e+2;

    Matrix3d M = pMatrix.block(0, 0, 3, 3);
    HouseholderQR<Matrix3d> householderQR;;

    householderQR.compute(M.inverse());

    MatrixXd Q = householderQR.householderQ();
    MatrixXd R =  householderQR.matrixQR().triangularView<Upper>();


    MatrixXd K = R.inverse().cwiseAbs(); // 绝对值

    MatrixXd Rotate =  K.inverse() * M;
    MatrixXd T = K.inverse() * pMatrix.col(3);
    MatrixXd C = -Rotate.transpose()*T;

    std::cout <<"C " << C << std::endl;
    std::cout << "Hello, world!" << std::endl;
    std::cout << "K " << K << std::endl;
    std::cout << "Rotate " << Rotate << std::endl;
    std::cout << "R " << Q.transpose() << std::endl;
    std::cout << "T " << T << std::endl;
}

void testHomography(){
    
     std::vector<std::string> files= {
        "/home/atway/code/opencv_data/left01.jpg",
        "/home/atway/code/opencv_data/left04.jpg"
    };

    std::vector<std::vector<cv::Point2f>> allCorners;
    cv::Size boardSize(6, 9);
    for(int i=0; i<files.size(); ++i) {

        cv::Mat img =  cv::imread(files[i]);
        std::vector<cv::Point2f> corners;

        bool ok = cv::findChessboardCorners(img, boardSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
        if(ok) {
            allCorners.push_back(corners);
           // cv::drawChessboardCorners(img, boardSize, cv::Mat(corners), ok);
           // cv::imshow("corners", img);
           // cv::waitKey();
        }

    }


    std::vector<Eigen::Vector2d> srcPoints, dstPoints;

    for(cv::Point2f& pt: allCorners[0])
    {
        srcPoints.push_back(Eigen::Vector2d(pt.x, pt.y));
    }
    for(cv::Point2f& pt: allCorners[1])
    {
        
        dstPoints.push_back(Eigen::Vector2d(pt.x, pt.y));

    }
    std::cout << srcPoints.size() << std::endl;
    Eigen::Matrix3d H;
    bool ok = findHomography(srcPoints, dstPoints, H, false);

    if(ok){
        std::cout << "H = " << H << std::endl;
        cv::Mat srcImage = cv::imread(files[0]);
        cv::Mat dstImage = cv::imread(files[1]);
        cv::Mat result;
        cv::Mat Hmat;
        cv::eigen2cv(H, Hmat);
        cv::warpPerspective(srcImage, result, Hmat, dstImage.size());
        cv::imshow("result", result);
        cv::waitKey();
        cv::imwrite("result.jpg", result);
    }
    /*
     * 优化前
     * H = 1.07368    -0.209543     -74.3206
        -0.0632842      1.02258      41.3408
        -0.00012842 -0.000406687            1
        
        // 优化后
        H = 1.07926    -0.155123      -74.276
        -0.0604588      1.02924      40.2497
        -0.000114997 -0.000406556            1
     * 
    H =  1.59953   -0.0338682     -1059.74
        -0.033448       1.6032     -889.907
        -3.29594e-07 -1.55556e-06       1.0004
     
     */
    
}

int main(int argc, char **argv) {

    std::vector<std::string> files= {
        "/home/atway/code/opencv_data/left01.jpg",
        "/home/atway/code/opencv_data/left02.jpg",
        "/home/atway/code/opencv_data/left03.jpg",
        "/home/atway/code/opencv_data/left04.jpg",
        "/home/atway/code/opencv_data/left05.jpg",
        "/home/atway/code/opencv_data/left06.jpg",
        "/home/atway/code/opencv_data/left07.jpg",
        "/home/atway/code/opencv_data/left08.jpg",
        "/home/atway/code/opencv_data/left09.jpg",
        "/home/atway/code/opencv_data/left11.jpg",
        "/home/atway/code/opencv_data/left12.jpg",
        "/home/atway/code/opencv_data/left13.jpg",
        "/home/atway/code/opencv_data/left14.jpg",
    };

    std::vector<std::vector<Eigen::Vector2d>> imagePoints;
    std::vector<std::vector<Eigen::Vector3d>> objectPoints;
    cv::Size boardSize(9, 6);
    cv::Size2f squareSize(25., 25.);
    for(int i=0; i<files.size(); ++i) {

        cv::Mat img =  cv::imread(files[i]);
        std::vector<cv::Point2f> corners;

        bool ok = cv::findChessboardCorners(img, boardSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
        if(ok) {
            cv::Mat gray;
            cv::cvtColor(img, gray, CV_BGR2GRAY);
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
            
            
           cv::drawChessboardCorners(img, boardSize, cv::Mat(corners), ok);
           cv::imshow("corners", img);
           cv::waitKey(100);
            std::vector<Eigen::Vector2d> _corners;
            for(auto& pt: corners){
                _corners.push_back(Eigen::Vector2d(pt.x, pt.y));
            }
            imagePoints.push_back(_corners);
        }

    }
    
    for(int i=0; i<imagePoints.size(); ++i){
        std::vector<Eigen::Vector3d> corners;
        getObjecPoints(boardSize, squareSize, corners);
        objectPoints.push_back(corners);
    }
    
    
    computeCameraCalibration(imagePoints, objectPoints);
    


  
    return 0;
}
