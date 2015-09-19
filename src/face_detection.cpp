#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include <dynamic_reconfigure/server.h>
#include <face_det_rec/arFacerecConfig.h>

#include <iostream>
#include <stdio.h>

#include <sstream>
#include <string>

#include <time.h>


using namespace std;
using namespace cv;


static const std::string OPENCV_WINDOW = "Image window";
//static const std::string OPENCV_WINDOW_2 = "Image window 2";
static const std::string OPENCV_WINDOW_menu = "Menu";

String face_cascade_name_0 = "include/face_detection/HaarCascades/haarcascade_frontalface_alt.xml";
String face_cascade_name_1 = "include/face_detection/HaarCascades/haarcascade_frontalface_alt2.xml";
String face_cascade_name_2 = "include/face_detection/HaarCascades/haarcascade_frontalface_alt_tree.xml";
String face_cascade_name_3 = "include/face_detection/HaarCascades/haarcascade_frontalface_default.xml";

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;

  dynamic_reconfigure::Server<face_detection::arFacerecConfig> srv;
  dynamic_reconfigure::Server<face_detection::arFacerecConfig>::CallbackType f;

  int counter;
  int neighborsValue;
  float scaleValue;
  int minSize;
  int cascadeValue;
  int imgScaleValue;
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
  string imageInput = "/ardrone/image_raw";
  string imageOutput = "/facerec/image_raw";

  float totalTime;
  int windowOnOff;


  char myflag;

  cv::CascadeClassifier face_cascade;
  cv::CascadeClassifier face_cascade_0;
  cv::CascadeClassifier face_cascade_1;
  cv::CascadeClassifier face_cascade_2;
  cv::CascadeClassifier face_cascade_3;
  std::vector<cv::Rect> faces;
  int gFps;
  int gCounter;
  Mat gray;
  std::vector<cv::Rect>::const_iterator i;


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



public:

  ImageConverter()
    : it_(nh_)
  {
    inputSkipp = 1;
    // Subscrive to input video feed and publish output video feed "/camera/image_raw",
    image_sub_ = it_.subscribe(imageInput, inputSkipp, &ImageConverter::imageCb, this);
    image_pub_ = it_.advertise(imageOutput, inputSkipp);


    if (face_cascade_0.load(face_cascade_name_0) == false) {
      printf("cascade.load_0() failed...\n");
      //std::cout << "face_cascade_0: " << face_cascade_name_0;
      }
    if (face_cascade_1.load(face_cascade_name_1) == false) {
     /printf("cascade.load_1() failed...\n");
      }
    if (face_cascade_2.load(face_cascade_name_2) == false) {
      printf("cascade.load_2() failed...\n");
      }
    if (face_cascade_3.load(face_cascade_name_3) == false) {
      printf("cascade.load_3() failed...\n");
      }








    counter = 0;
    gFps = 2;
    gCounter = gFps -1;
    neighborsValue = 2;
    scaleValue = 1.2;
    minSize = 13;
    maxSize = 250;
    cascadeValue = 2;
    imgScaleValue = 2;
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


    f = boost::bind(&ImageConverter::callback, this, _1, _2);
    srv.setCallback(f);

    //CV_HAAR_DO_CANNY_PRUNING
    //CV_HAAR_SCALE_IMAGE
    //CV_HAAR_FIND_BIGGEST_OBJECT
    //CV_HAAR_DO_ROUGH_SEARCH

  }


  ~ImageConverter()
  {
    if(windowOnOff == 1){
      //cv::destroyWindow(OPENCV_WINDOW);
      cv::destroyWindow(OPENCV_WINDOW);
      //cv::destroyWindow(OPENCV_WINDOW_menu);
    }
  }

  void update(String casc0, String casc1, String casc2, String casc3, String windowOption)
  {
    windowOnOff = atoi(windowOption.c_str());

    printf("################\n" );
    if (face_cascade_0.load(casc0) == false) {
      printf("cascade.load_0() failed...\n");
      }
    if (face_cascade_1.load(casc1) == false) {
      printf("cascade.load_1() failed...\n");

      }
    if (face_cascade_2.load(casc2) == false) {
      printf("cascade.load_2() failed...\n");

      }
    if (face_cascade_3.load(casc3) == false) {
      printf("cascade.load_3() failed...\n");

      }
    std::cout << "face_cascade_0: " << casc0 << "\n";
    std::cout << "face_cascade_1: " << casc1 << "\n";
    std::cout << "face_cascade_2: " << casc2 << "\n";
    std::cout << "face_cascade_3: " << casc3 << "\n";


    if(windowOnOff == 1){
      //generate windows
      cv::namedWindow(OPENCV_WINDOW);
      cv::namedWindow(OPENCV_WINDOW_menu);

    }


  }



  //called every time theres a new image
  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {

    // reset the fps frame counter every once in a while
    if (frameCounter == 0){
        time(&start);
        if (totalFrameCounter == 0) {
            time(&timeZero);
            begin = ros::Time::now();
        }
    }

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

    //ros::Time begin = ros::Time::now();
    imgScale = 1.0/imgScaleValue;
    gCounter += 1;

    //####################################################################
    //image preprocessing
    //####################################################################

    // change contrast: 0.5 = half  ; 2.0 = double
    cv_ptr->image.convertTo(gray, -1, contrastFactor, 0);

    // create B&W image
    cvtColor( gray, gray, CV_BGR2GRAY );

    //equalize the histogram
    if(histOnOff == 1){
      equalizeHist( gray, gray );
    }

    //blue image by blurfactor
    if(blurFactor > 0){
      blur( gray, gray, Size( blurFactor, blurFactor) );
    }

    //scale image
    resize(gray, gray, Size(), imgScale , imgScale);



    //####################################################################
    //####################### detection ##################################
    //####################################################################

    //this part is only excecuted every couple of frames, depending on gFps setting
    if(gCounter > gFps -1){
      gCounter = 0;
      face_cascade.detectMultiScale(
        gray,
        faces,
        scaleValue,          // scale factor
        neighborsValue,            // minimum neighbors
        0|myflag,            // flags
        cv::Size(minSize, minSize), // minimum size
        cv::Size(maxSize, maxSize) // minimum size
      );
      //0|myflag,            // flags
    }

    totalDetections += faces.size();

    //print faces on top of image
    for (i = faces.begin(); i != faces.end(); ++i) {
        cv::rectangle(
          cv_ptr->image,
          cv::Point((i->x)*imgScaleValue, (i->y)*imgScaleValue),
          cv::Point((i->x)*imgScaleValue + (i->width)*imgScaleValue, (i->y)*imgScaleValue + (i->height)*imgScaleValue),
          CV_RGB(50, 255 , 50),
          2);
    }


    //Display Section, images will only displayed if option is selected
    if(debug != 0){
      if(debug == 1){
        cv::imshow(OPENCV_WINDOW, cv_ptr->image);
      }
      else if(debug == 2){
        for (i = faces.begin(); i != faces.end(); ++i) {
          //if(i->width <250){
            cv::rectangle(
              gray,
              cv::Point(i->x, i->y),
              cv::Point(i->x + i->width, i->y + i->height),
              CV_RGB(0, 255, 0),
              2);
          //}
        }
        cv::imshow(OPENCV_WINDOW, gray);
      }
    }


    // Output modified video stream
    image_pub_.publish(cv_ptr->toImageMsg());


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


    totalTime = ( ros::Time::now() - begin).toSec();

    sec = difftime(end, timeZero);
    //printf("FPS : %.2f   #   skipped frames : %i   #   Detected faces : %lu  \n",fps, gFps-1,faces.size());
    //printf("frame: %i, time passed: %.2f, number of detections: %i  \n",totalFrameCounter, sec ,totalDetections );
    printf("time passed: %.2f  #  frames: %i  #  fps: %f  #  total number of detections: %i # FPS in last 100 frames : %.2f \n", totalTime, totalFrameCounter, totalFrameCounter/totalTime, totalDetections, fps);

    totalFrameCounter += 1;
    cv::waitKey(3);
  }

  //############################################
  //########## callback function ###############
  //############################################
  void callback(face_det_rec::arFacerecConfig &config, uint32_t level)
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
    imgScaleValue = config.imgScaleValue;
    histOnOff = config.histOnOff;
    blurFactor = config.blurFactor;
    brightnessFactor = config.brightnessFactor;
    contrastFactor = config.contrastFactor;
    //flag = config.flag;
    debug = config.debug;
    inputSkipp = config.inputSkipp;


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
    }

    if (imageInput != config.imageInput || inputSkipp != config.inputSkipp) {
      imageInput = config.imageInput;
      image_sub_ = it_.subscribe(imageInput, inputSkipp, &ImageConverter::imageCb, this);
    }

    if (imageOutput != config.imageOutput || inputSkipp != config.inputSkipp) {
      imageOutput = config.imageOutput;
      image_pub_ = it_.advertise(imageOutput, inputSkipp);
    }

  }



  void callSrv()
  {
    srv.setCallback(f);
  }

};





int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  if(argc < 6){
    printf("You must provide at least 5 arguments (cascade files) plus OnOff window\n");
    exit(0);
  }

  ros::init(argc, argv, "face_detection");


  ROS_INFO("Starting to spin...");




  ImageConverter ic;
  ic.update(argv[1],argv[2],argv[3],argv[4], argv[5]);
  ic.callSrv();
  ros::spin();
  return 0;
}
