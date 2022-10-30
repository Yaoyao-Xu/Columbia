# Ultrasonos_RedPitaya_Summer2020

As part of the Ultrasonos open source ultrasound system, the Red Pitaya embedded system samples the analog system, control the mechanical probe, and process the information before sending to PC for 3D image rendering.

This is the C program that enable data acquisition, information integration, and data transmission on Red Pitaya System. It features 15.6Msps ADC sampling speed, combing information of time stamp, encoder readings, adc readings, and crc checksum. In addition, it provides three different way to transmit the data files: save data on local, UDP, and TCP transmission. For more information, please refer to the documentation. 

## Getting Started

These instructions will get you a copy of the project up and running on your Red Pitaya board  for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

The Red Pitaya system with STEMlab environment.

### Installing

Clone the repo

```
git clone https://github.com/ColumbiaOpenSourceUltrasound/Ultrasonos_RedPitaya_Summer2020.git 
```

And repeat

```
cd Ultrasonos_RedPitaya_Summer2020
```

## Running the program
Load the FPGA image on Red Pitaya

```
cat /opt/redpitaya/fpga/fpga_0.94.bit > /dev/xdevcfg
```

Compile
```
make all
```

Run the executable
```
LD_LIBRARY_PATH=/opt/redpitaya/lib ./main
```

## Built With

* [PyLX-16A](https://github.com/ethanlipson/PyLX-16A) - LX-16 Servo Library


## Authors

* **Hanwen Zhao** - *Initial work* - [hanwenzhao](https://github.com/hanwenzhao)

* **Yanwen Jing** - *Red Pitaya Control with Ultrasonos All-in-One Software* - [YourSoulShallBeMine](https://github.com/YourSoulShallBeMine)

* **Ethan Lipson** - *LX-16 Servo Library* - [ethanlipson](https://github.com/ethanlipson)


See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
