//##########################################################################
// DO NOT MODIFY
//
//This project was created within an academic research setting, and thus should
//be considered as EXPERIMENTAL code. There may be bugs and deficiencies in the
//code, so please adjust expectations accordingly. With that said, we are
//intrinsically motivated to ensure its correctness (and often its performance).
//Please use the corresponding web repository tool (e.g. github, bitbucket, etc)
//to file bugs, suggestions, pull requests; we will do our best to address them
//in a timely manner.
//
// SOFTWARE LICENSE AGREEMENT (BSD LICENSE):
//
//
//Copyright (c) 2015, Philippe Ludivig
//All rights reserved.
//
//Redistribution and use in source and binary forms, with or without
//modification, are permitted provided that the following conditions are met:
//
//* Redistributions of source code must retain the above copyright notice, this
//  list of conditions and the following disclaimer.
//
//* Redistributions in binary form must reproduce the above copyright notice,
//  this list of conditions and the following disclaimer in the documentation
//  and/or other materials provided with the distribution.
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//##########################################################################


#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/video/tracking.hpp>
#include <dynamic_reconfigure/server.h>
#include <face_detection/face_trackConfig.h>

#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include "std_msgs/Int32MultiArray.h"

#include <iostream>
#include <stdio.h>
#include <sstream>
#include <string>
#include <time.h>


using namespace std;
using namespace cv;

// OpenCV publishing windows
static const std::string OPENCV_WINDOW = "Image window";



//####################################################################
//#                                                                  #
//####################################################################
//###################### Face Detector Class #########################
//####################################################################
//#                                                                  #
//####################################################################
class FaceDetector
{
  ros::NodeHandle nh_;
  ros::NodeHandle n;

  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  ros::Publisher faceCoord_pub;

  // required for the dynamic reconfigure server
  dynamic_reconfigure::Server<face_detection::face_trackConfig> srv;
  dynamic_reconfigure::Server<face_detection::face_trackConfig>::CallbackType f;

  int counter;
  int neighborsValue;
  float scaleValue;
  int minSize;
  int cascadeValue;
  float imgScale;
  int histOnOff;
  int blurFactor;
  int brightnessFactor;
  float contrastFactor;
  int debug;
  int button1;
  int flag;
  int colourValue;
  int inputSkipp;
  int maxSize;
  float totalTime;
  int windowOnOff;
  int maxTrackingNum;
  int initialDetectionNum;
  int pixelSwitch;
  int maxNumFeatures;
  int trackSearchWinSize;
  int IDcounter;
  string imageInput = "/camera/image_raw";
  string imageOutput = "/face_det/image_raw";

  char myflag;

  cv::CascadeClassifier face_cascade;
  cv::CascadeClassifier face_cascade_0;
  cv::CascadeClassifier face_cascade_1;
  cv::CascadeClassifier face_cascade_2;
  cv::CascadeClassifier face_cascade_3;
  cv::CascadeClassifier face_cascade_profile;

  std::vector<cv::Rect> faces;
  std::vector<int> lastSeen;
  std::vector<int> faceID;
  std::vector<int> detectionLength;
  int gFps;
  int gCounter;
  Mat gray;
  std::vector<cv::Rect>::const_iterator i;


  // Mat used for tracking

  Mat previousFrame;
  Mat inputImage;
  // start and end times
  time_t start, end, timeZero, currentTime;
  ros::Time begin;

  // fps calculated using number of frames / seconds
  double fps;
  // frame counter
  int frameCounter;
  int totalFrameCounter;
  // floating point seconds elapsed since start
  double sec;
  int totalDetections;

private:
    //####################################################################
    //############# called every time theres a new image #################
    //####################################################################
    void newImageCallBack(const sensor_msgs::ImageConstPtr& msg)
    {

      // starts time calculations, one of the counters is being reset every once in a while
      if (frameCounter == 0){
          time(&start);
          if (totalFrameCounter == 0) {
              time(&timeZero);
              begin = ros::Time::now();



          }
      }


      // retrieves the image from the camera driver
      cv_bridge::CvImagePtr cv_ptr;
      try
      {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      }
      catch (cv_bridge::Exception& e)
      {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }

      if (totalFrameCounter == 0) {

          previousFrame = cv_ptr->image;
          cvtColor( previousFrame, previousFrame, CV_BGR2GRAY );

      }



      //####################################################################
      //######################## image preprocessing #######################
      //####################################################################

      gCounter += 1;

      // change contrast: 0.5 = half  ; 2.0 = double
      cv_ptr->image.convertTo(gray, -1, contrastFactor, 0);

      // create B&W image
      cvtColor( gray, gray, CV_BGR2GRAY );

      //equalize the histogram
      if(histOnOff == 1){
        equalizeHist( gray, gray );
      }

      //blur image by blurfactor
      if(blurFactor > 0){
        blur( gray, gray, Size( blurFactor, blurFactor) );
      }


      inputImage = gray.clone();
      //printf("width = %d\n", inputImage.cols);

      //scale image
      resize(gray, gray, Size(), imgScale , imgScale);



      //####################################################################
      //####################### detection part #############################
      //####################################################################
      // depending on the gFps setting, this part is only executed every couple of frames

      std::vector<cv::Rect> newFaces;
      if(gCounter > gFps -1){
        //gCounter = 0;

        //if(totalFrameCounter % 2){
        face_cascade.detectMultiScale(
          gray,                       // input image (grayscale)
          newFaces,                      // output variable containing face rectangle
          scaleValue,                 // scale factor
          neighborsValue,             // minimum neighbors
          0|myflag,                   // flags
          cv::Size(minSize, minSize), // minimum size
          cv::Size(maxSize, maxSize)  // minimum size
        );

        /* printf("this\n" );
        } else {
        face_cascade_profile.detectMultiScale(
          gray,                       // input image (grayscale)
          newFaces,                      // output variable containing face rectangle
          scaleValue,                 // scale factor
          neighborsValue,             // minimum neighbors
          0|myflag,                   // flags
          cv::Size(minSize, minSize), // minimum size
          cv::Size(maxSize, maxSize)  // minimum size
        );
        printf("that\n" );
        }*/
          if(imgScale != 1){
              for(unsigned i = 0; i < newFaces.size(); ++i) {
                  newFaces[i].x = newFaces[i].x/imgScale;
                  newFaces[i].y = newFaces[i].y/imgScale;
                  newFaces[i].width = newFaces[i].width/imgScale;
                  newFaces[i].height = newFaces[i].height/imgScale;
              }
          }
        }



      //####################################################################
      //####################### tracking part ##############################
      //####################################################################
      //
      Mat croppedImage;

      int mx;
      int my;
      int mh;
      int mw;

      for (unsigned i = 0; i < faces.size(); i++) {
        //printf("next faces analysis \n");
        mx = faces[i].x;
        my = faces[i].y;
        mh = faces[i].height;
        mw = faces[i].width;
        //printf("adding values \n");
        double mvRateX = 0.0;
        double mvRateY = 0.0;

        std::vector<cv::Point2f> features_prev, features_next;
        std::vector<uchar> status;
        std::vector<float> err;

        //printf("cropping \n");
        //printf("mx %d \n", mx);
        croppedImage = inputImage(Rect(mx,my, mw,mh));
        cv::goodFeaturesToTrack(croppedImage, // the image
          features_prev,   // the output detected features
          maxNumFeatures,  // the maximum number of features
          0.4,     // quality level
          2     // min distance between two features
        );


        //printf("test %lu \n",features_prev.size());
        for (unsigned j = 0; j < features_prev.size(); j++) {
            features_prev[j].x = features_prev[j].x + mx;
            features_prev[j].y = features_prev[j].y + my;
            if(pixelSwitch == 0){
                cv::circle(cv_ptr->image, cv::Point(features_prev[j].x , features_prev[j].y), 1, CV_RGB(255,0,0),CV_FILLED);
            }


        }

        //printf("Optical Flow \n");
        // ####################################
        // Optical Flow
        cv::Size winSize(trackSearchWinSize,trackSearchWinSize);
        if(features_prev.size() != 0){
            cv::calcOpticalFlowPyrLK(
              previousFrame, inputImage, // 2 consecutive images
              features_prev, // input point positions in first im
              features_next, // output point positions in the 2nd
              status,    // tracking success
              err,      // tracking error
              winSize
            );
        }
        //printf("print sizes of arrays %lu %lu %lu\n", features_prev.size(), features_next.size(), status.size() );
        //for (unsigned j = 0; j < features_next.size(); j++) {
        //    printf("%u ", status[j]);
        //}

        // ####################################
        // add mx my to the values of the cropped window
        // then calc error rate
        for (unsigned j = 0; j < features_next.size(); j++) {
            if(pixelSwitch == 0){
                cv::circle(cv_ptr->image, cv::Point(features_next[j].x , features_next[j].y), 1, CV_RGB(255,255,255),CV_FILLED);
            }
            mvRateX += features_next[j].x - features_prev[j].x;
            mvRateY += features_next[j].y - features_prev[j].y;

        }
        mvRateX = mvRateX /features_next.size();
        mvRateY = mvRateY /features_next.size();

        if(pixelSwitch == 0){
            cv::circle(cv_ptr->image, cv::Point(mx + mw/2, my + mh/2), 10, CV_RGB(0,255,0));
            cv::circle(cv_ptr->image, cv::Point(mx + mw/2 + mvRateX, my + mh/2 +mvRateY), 10, CV_RGB(255,0,0));
        }


        // update error rate
        faces[i].x = faces[i].x + mvRateX;
        faces[i].y = faces[i].y + mvRateY;
        //printf("update error rate \n");
        //if(faces[i].x < 0){ faces[i].x = 0;}
        //if(faces[i].y < 0){ faces[i].y = 0;}

        if(faces[i].x < 0 || faces[i].y < 0 || (faces[i].x + faces[i].width) > (cv_ptr->image.cols) || (faces[i].y+ faces[i].height)  > (cv_ptr->image.rows)){
            faces.erase (faces.begin()+i);
            lastSeen.erase (lastSeen.begin()+i);
            faceID.erase (faceID.begin()+i);
            detectionLength.erase (detectionLength.begin()+i);
            i--;
        }

      }



      //####################################################################
      //################### find intersection  #############################
      //####################################################################
      // here we compare the tracked faces against the newly detected faces:
      // if we have an intersection, the tracked faces is udated with a new detection
      // if we have no intersection, the new detection is added to the current faces
      if(gCounter > gFps -1){
          gCounter = 0;
          int duplicatedFaceDetection = 0;
          for (unsigned i = 0; i < newFaces.size(); i++) {
              for ( unsigned j = 0; j < faces.size(); j++) {
                  Rect interSection = faces[j] & newFaces[i];

                  // all values == 0 means no intersection
                  if (interSection.width != 0){
                        // we have intersection
                        duplicatedFaceDetection = 1;
                        //printf("instersection %d %d %d %d\n", interSection.x,interSection.y,interSection.width,interSection.height);
                        faces[j] = newFaces[i];
                        lastSeen[j] = maxTrackingNum;
                  } //else {
                        // we have no intersection
                        //printf("no instersection %d %d %d %d\n", interSection.x,interSection.y,interSection.width,interSection.height);
                        //faces[j] = newFaces[i];
                  //}

              }

              if(duplicatedFaceDetection == 0){
                  faces.push_back(newFaces[i]);
                  lastSeen.push_back(initialDetectionNum);
                  faceID.push_back(IDcounter);
                  IDcounter++;
                  detectionLength.push_back(1);
              }
          }


      //####################################################################
      //################### count lastSeen #################################
      //####################################################################
          for (unsigned i = 0; i < faces.size(); i++) {
              if(lastSeen[i] == 1){
                  faces.erase (faces.begin()+i);
                  lastSeen.erase (lastSeen.begin()+i);
                  faceID.erase (faceID.begin()+i);
                  detectionLength.erase (detectionLength.begin()+i);
                  i--;
              } else {
                  lastSeen[i] = lastSeen[i] -1;
                  detectionLength[i] = detectionLength[i] + 1;
              }
          }
      }






      // ##############################
      // rest program
      previousFrame = inputImage.clone();

      //keep number of total detections
      totalDetections += faces.size();

      //print faces on top of image
      if(pixelSwitch == 0){
          cv_ptr->image = drawFaces(cv_ptr->image);
      } else {

          //blur faces
          Mat blurMyFace;
          for (i = faces.begin(); i != faces.end(); ++i) {
              Rect cropROI((i->x),(i->y),(i->width), (i->height));
              blurMyFace = cv_ptr->image(cropROI);
              blurMyFace = pixelate(blurMyFace,16);
              blurMyFace.copyTo(cv_ptr->image(cropROI));
          }
      }


      //Display Section, images will only displayed if option is selected
      if(debug != 0){
        if(debug == 1 || debug == 3){
          cv::imshow(OPENCV_WINDOW, cv_ptr->image);
        }
        else if(debug == 2){
          for (i = faces.begin(); i != faces.end(); ++i) {
              cv::rectangle(
                gray,
                cv::Point((i->x)*imgScale, (i->y)*imgScale),
                cv::Point((i->x)*imgScale + (i->width)*imgScale, (i->y)*imgScale + (i->height)*imgScale),
                CV_RGB(0, 255, 0),
                2);
          }
          cv::imshow(OPENCV_WINDOW, gray);
        }
      }



      if(debug != 3){
          // Output modified video stream
          image_pub_.publish(cv_ptr->toImageMsg());
          //image_pub_.publish(cv_ptr->toImageMsg());
          //ros::NodeHandle n;
          //

          // ### publishing coordinates ###
          std_msgs::Int32MultiArray myMsg;
          myMsg.data.clear();
          // publish current fps rate
          myMsg.data.push_back(fps);
          // publish number of detected faces
          myMsg.data.push_back(faces.size());
          // width of the image
          myMsg.data.push_back(cv_ptr->image.cols);
          // height of the image
          myMsg.data.push_back(cv_ptr->image.rows);
          //for (i = faces.begin(); i != faces.end(); ++i) {
          for ( unsigned i = 0; i < faces.size(); i++) {
              myMsg.data.push_back(faceID[i]);
              myMsg.data.push_back(detectionLength[i]);
              myMsg.data.push_back(faces[i].x);
              myMsg.data.push_back(faces[i].y);
              myMsg.data.push_back(faces[i].width);
              myMsg.data.push_back(faces[i].height);
          }
          faceCoord_pub.publish(myMsg);
      }


      // fps counter begin
      time(&end);
      frameCounter++;
      if (frameCounter > 100){
          sec = difftime(end, start);
          fps = frameCounter/sec;
          //printf("%.2f fps\n", fps);
          //printf("%.2f fps   ; sec = %.2f ; counter = %i \n", fps, sec,frameCounter);
          frameCounter = 0;
        }
      // fps counter end

      // print out averages
      totalTime = ( ros::Time::now() - begin).toSec();
      sec = difftime(end, timeZero);
      printf("time passed: %.2f  #  frames: %i  #  fps: %f  \n#  total number of detections: %i # FPS in last 100 frames : %.2f \n", totalTime, totalFrameCounter, totalFrameCounter/totalTime, totalDetections, fps);

      totalFrameCounter += 1;
      cv::waitKey(3);



    }

    // #########################################
    // #### used for pixelising images #########
    // #########################################
    Mat pixelate(Mat myImage, int pixelizationRate){

      cv::Mat result = cv::Mat::zeros(myImage.size(), CV_8UC3);
      for (int i = 0; i < myImage.rows; i += pixelizationRate)
         {
             for (int j = 0; j < myImage.cols; j += pixelizationRate)
             {
                 cv::Rect rect = cv::Rect(j, i, pixelizationRate, pixelizationRate) &
                                 cv::Rect(0, 0, myImage.cols, myImage.rows);

                 cv::Mat sub_dst(result, rect);
                 sub_dst.setTo(cv::mean(myImage(rect)));

             }
         }

      return result;
    }



    //####################################################################
    //##################### drawFaces ####################################
    //####################################################################
    //takes the detected faces and daws them on top of an image
    //cv_bridge::CvImagePtr drawFaces(cv_bridge::CvImagePtr myImagePtr) {
    cv::Mat drawFaces(cv::Mat myImage) {

        for (unsigned i = 0; i < faces.size(); ++i) {
            //printf("last seen %d \n", lastSeen[i]);
            if(lastSeen[i] == (maxTrackingNum -1) ){
                // green = newly detected
                //printf("seen \n");
                cv::rectangle(
                  myImage,
                  cv::Point((faces[i].x), (faces[i].y)),
                  cv::Point((faces[i].x) + (faces[i].width), (faces[i].y) + (faces[i].height)),
                  CV_RGB(50, 255 , 50),
                  2);
            } else if(lastSeen[i] != 1) {
                // blue = old face
                cv::rectangle(
                  myImage,
                  cv::Point((faces[i].x), (faces[i].y)),
                  cv::Point((faces[i].x) + (faces[i].width), (faces[i].y) + (faces[i].height)),
                  CV_RGB(50, 50 , 255),
                  2);
            } else {
                // red = about to disappear
                cv::rectangle(
                  myImage,
                  cv::Point((faces[i].x), (faces[i].y)),
                  cv::Point((faces[i].x) + (faces[i].width), (faces[i].y) + (faces[i].height)),
                  CV_RGB(255, 50 , 50),
                  2);
            }

            cv::putText(myImage, std::to_string(faceID[i]), cv::Point(faces[i].x,faces[i].y+faces[i].height+20), CV_FONT_NORMAL, 0.5, Scalar(255,255,255),1,1);

        }
        return myImage;
    }


public:

  //######################################################################
  //##################### constructor ####################################
  //######################################################################
  FaceDetector(String casc0, String casc1, String casc2, String casc3)
    : it_(nh_)
  {
    inputSkipp = 1;


    // Subscrive to input video feed and publish output video feed "/camera/image_raw",
    //string imageInput = "/camera/image_raw";
    //inputSkipp = 1
    image_sub_ = it_.subscribe(imageInput, inputSkipp, &FaceDetector::newImageCallBack, this);
    //string imageOutput = "/face_det/image_raw";
    image_pub_ = it_.advertise(imageOutput, inputSkipp);



    //loads in the different cascade detection files
    printf("################\n" );
    if (face_cascade_0.load(casc0) == false) {
      printf("cascade.load_0() failed...\n");
      printf("The missing cascade file is /include/face_detection/HaarCascades/haarcascade_frontalface_alt.xml\n");
      exit(0);
      }
    if (face_cascade_1.load(casc1) == false) {
      printf("cascade.load_1() failed...\n");
      printf("The missing cascade file is /include/face_detection/HaarCascades/haarcascade_frontalface_alt2.xml\n");
      exit(0);
      }
    if (face_cascade_2.load(casc2) == false) {
      printf("cascade.load_2() failed...\n");
      printf("The missing cascade file is /include/face_detection/HaarCascades/haarcascade_frontalface_alt_tree.xml\n");
      exit(0);
      }
    if (face_cascade_3.load(casc3) == false) {
      printf("cascade.load_3() failed...\n");
      printf("The missing cascade file is /include/face_detection/HaarCascades/haarcascade_frontalface_default.xml\n");
      exit(0);
      }

    if (face_cascade_profile.load("/home/phil/catkin_ws/src/face_detection/include/face_detection//HaarCascades/haarcascade_frontalface_default.xml") == false) {
      printf("cascade.load_3() failed...\n");
      printf("The missing cascade file is /include/face_detection/HaarCascades/haarcascade_frontalface_default.xml\n");
      exit(0);
      }
    printf("OpenCV: %s \n", cv::getBuildInformation().c_str());

    counter = 0;
    gFps = 2;
    gCounter = gFps -1;
    neighborsValue = 2;
    scaleValue = 1.2;
    minSize = 13;
    maxSize = 250;
    cascadeValue = 2;
    imgScale = 1.0;
    histOnOff = 0;
    blurFactor = 0;
    brightnessFactor = 0;
    button1 = 0;
    frameCounter = 0;
    contrastFactor = 1.5;
    flag = 2;
    fps = -1;
    debug = 0;
    totalFrameCounter = 0;
    totalTime = 0;
    myflag = CV_HAAR_DO_CANNY_PRUNING;
    windowOnOff = 0;
    totalDetections = 0;
    maxTrackingNum = 60;
    initialDetectionNum = 4;
    maxNumFeatures = 15;
    pixelSwitch = 1;
    trackSearchWinSize = 100;
    IDcounter = 1;

    //setNumThreads(0);

    //setting up the dynamic reconfigure server
    f = boost::bind(&FaceDetector::callback, this, _1, _2);
    srv.setCallback(f);


    faceCoord_pub = n.advertise<std_msgs::Int32MultiArray>("faceCoord", 1000);

    //generate windows
    if(debug != 0){
      cv::namedWindow(OPENCV_WINDOW);
    }

  }

  //####################################################################
  //##################### destroyer ####################################
  //####################################################################
  ~FaceDetector()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }


  //########################################################
  //########## reconfigure callback function ###############
  //########################################################
  void callback(face_detection::face_trackConfig &config, uint32_t level)
  {

    ROS_INFO("Reconfigure request");


    if (config.debug == 0 && debug > 0) {
      cv::destroyWindow(OPENCV_WINDOW);
    }

    gFps = config.skipFrames;
    neighborsValue = config.neighborsValue;
    scaleValue = config.scaleValue;

    minSize = config.minSize/scaleValue;
    maxSize = config.maxSize/scaleValue;
    cascadeValue = config.cascadeValue;
    imgScale = config.imgScale;
    histOnOff = config.histOnOff;
    blurFactor = config.blurFactor;
    brightnessFactor = config.brightnessFactor;
    contrastFactor = config.contrastFactor;
    debug = config.debug;
    inputSkipp = config.inputSkipp;
    pixelSwitch = config.pixelSwitch;
    maxNumFeatures = config.maxNumFeatures;
    maxTrackingNum = config.maxTrackingNum;
    initialDetectionNum = config.initialDetectionNum;
    trackSearchWinSize = config.trackSearchWinSize;
    //selecting the correct flag for the
    switch(config.myflag){
      case 0 :
        myflag = CV_HAAR_SCALE_IMAGE;
        break;
      case 1 :
        myflag = CV_HAAR_FIND_BIGGEST_OBJECT;
        break;
      case 2 :
        myflag = CV_HAAR_DO_CANNY_PRUNING;
        break;
      case 3 :
        myflag = CV_HAAR_DO_ROUGH_SEARCH;
        break;
      default:
        myflag = CV_HAAR_SCALE_IMAGE;
        break;
      }

    switch (cascadeValue) {
      case 0:
        face_cascade = face_cascade_0;
        break;
      case 1:
        face_cascade = face_cascade_1;
        break;
      case 2:
        face_cascade = face_cascade_2;
        break;
      case 3:
        face_cascade = face_cascade_3;
        break;
      default:
        face_cascade = face_cascade_0;
        break;
    }

    if (imageInput != config.imageInput || inputSkipp != config.inputSkipp) {
      imageInput = config.imageInput;
      image_sub_ = it_.subscribe(imageInput, inputSkipp, &FaceDetector::newImageCallBack, this);
    }

    if (imageOutput != config.imageOutput || inputSkipp != config.inputSkipp) {
      imageOutput = config.imageOutput;
      image_pub_ = it_.advertise(imageOutput, inputSkipp);
    }

  }


  // check the reconfigure sever
  void callSrv()
  {
    srv.setCallback(f);
  }

};




//########################################################
//##################### Main #############################
//########################################################
int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  printf("\n");
  printf("\n");
  printf("##############################################\n");
  printf("############ ROS Face Detection ##############\n");
  printf("##############################################\n");
  printf("\n");
  if(argc < 5){
    printf("Not Enough arguments, use one of the provided Roslaunch files\n");
    printf("\n");
    printf("Alternatively, arguments are needed as follows:\n");
    printf("01) Detection Cascade file 1\n");
    printf("02) Detection Cascade file 2\n");
    printf("03) Detection Cascade file 3\n");
    printf("04) Detection Cascade file 4\n");
    printf("\n");
    printf("\n");
    exit(0);
  }



  ros::init(argc, argv, "face_tracking");


  ROS_INFO("Starting to spin...");

  FaceDetector faceDet(argv[1],argv[2],argv[3],argv[4]);
  faceDet.callSrv();
  ros::spin();
  return 0;
}
