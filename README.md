# Human-Data-Analytics
Human Data Analytics exam repository. All the code used for the model and the live demo can be found here.

Together with other physiological activities, sleep has been shown to have a significant impact on different aspects of human health. Monitoring sleep posture can give valuable information not only to improve sleep quality, but also to prevent diseases such as pressure ulcers or sleep apnea. 
In this study, we use pressure maps collected with commercial pressure sensing mats to classify both subjects and 17 different in-bed postures, learning both tasks in parallel.
By comparing the results obtained using standard deep learning architectures, we build our own model making use of a simplified version of the \mbox{inception} architecture, built from scratch. 
With that, we are able to outperform the accuracy of the state-of-the-art model in the classification of postures, going beyond $85\%$ accuracy when validating the model with LOSO using augmented data.
The experimental results show that all proposed models can identify the patients and their in-bed posture with almost no errors. 
We demonstrate the effectiveness of using augmentation and different feature extraction levels to further generalize the network and make it more robust to previously unseen data. Finally we show how our model can  ultimately be deployed in clinical and smart home environments as a complementary tool with other available automated patient monitoring systems. This is done by implementing the LOSO validation scheme that reaches a $88\%$ classification accuracy for the 17 different postures on previously unseen subjects.

![alt text](https://github.com/ZiliottoFilippoDev/Human-Data-Analytics/blob/c4f18a4c2b5016a4b8f72ad8c02732ea8f21f4ac/architecture.jpg)

![alt text](https://github.com/ZiliottoFilippoDev/Human-Data-Analytics/blob/9a437bc847587ca5c3b0e7e759416ad1f66f3c4e/result.jpg)


