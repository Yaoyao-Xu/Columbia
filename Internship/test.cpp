// #ifndef ADVANCED_SENSING
// #define ADVANCED_SENSING
#include "dji_linux_helpers.hpp"
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include "dji_vehicle.hpp"
#include "dji_perception.hpp"
#include <vector>
#include <iostream>
// #include "opencv2/opencv.hpp"
// #include "opencv2/highgui/highgui.hpp"
// #endif
// using namespace DJI::OSDK;
// using namespace cv;
using namespace std;

// void show_rgb(CameraRGBImage img, void *p)
// {
//   string name = string(reinterpret_cast<char *>(p));
//   cout << "#### Got image from:\t" << name << endl;
//   Mat mat(img.height, img.width, CV_8UC3, img.rawData.data(), img.width*3);
//   cvtColor(mat, mat, COLOR_RGB2BGR);
//   imshow(name,mat);
//   cv::waitKey(1);
// }

// int main3(int argc,char **argv)
// {
//   bool f = false;
//   bool m = false;
//   char c = 0;
//   cout << "Please enter the type of camera stream you want to view\n"
//        << "m: Main Camera\n"
//        << "f: FPV  Camera" << endl;
//   cin >> c;

//   switch(c)
//   {
//     case 'm':
//       m=true; break;
//     case 'f':
//       f=true; break;
//     default:
//       cout << "No camera selected";
//       return 1;
//   }

//   bool enableAdvancedSensing = true;
//   LinuxSetup linuxEnvironment(argc, argv, enableAdvancedSensing);
//   Vehicle*   vehicle = linuxEnvironment.getVehicle();
//   const char *acm_dev = linuxEnvironment.getEnvironment()->getDeviceAcm().c_str();
//   vehicle->advancedSensing->setAcmDevicePath(acm_dev);
//   if (vehicle == NULL)
//   {
//     std::cout << "Vehicle not initialized, exiting.\n";
//     return -1;
//   }

//   char fpvName[] = "FPV_CAM";
//   char mainName[] = "MAIN_CAM";

//   bool camResult = false;
//   if(f)
//   {
//     camResult = vehicle->advancedSensing->startFPVCameraStream(&show_rgb, &fpvName);
//   }
//   else if(m)
//   {
//     camResult = vehicle->advancedSensing->startMainCameraStream(&show_rgb, &mainName);
//   }

//   if(!camResult)
//   {
//     cout << "Failed to open selected camera" << endl;
//     return 1;
//   }

//   //cameraZoomControl(vehicle);     //run camera zoom control test
//   CameraRGBImage camImg;            //get the camera's image

//   // main thread just sleep
//   // callback function will be called whenever a new image is ready
//   sleep(20);

//   if(f)
//   {
//     vehicle->advancedSensing->stopFPVCameraStream();
//   }
//   else if(m)
//   {
//     vehicle->advancedSensing->stopMainCameraStream();
//   }
//   return 0;
// }

// int main(int argc,char **argv)
// {
//     bool enableAdvancedSensing = false;
//     LinuxSetup linuxEnvironment(argc, argv, enableAdvancedSensing);
// }