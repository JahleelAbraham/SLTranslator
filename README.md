# Using Machine Learning models to recognise Sign Language

_Jahleel A., Connor R., Bobby A., Isabel H_

# Summary

After learning more about AI & Machine Learning technology, how it works and, what can be accomplished using it, we as a group decided to apply a Machine Learning model to recognise sign language to improve the communication between hearing impaired / deaf people and modern computers.

# Table of Contents

[Summary](#_Toc153533771)

[Introduction](#_Toc153533772)

[Legal & Ethical](#_Toc153533773)

[Ethical Aspects of Neural Networking](#_Toc153533774)

[Legal aspects of Neural Networking](#_Toc153533775)

[Training the Model](#_Toc153533776)

[Finding and Recognising a Hand and Predicting the Sign](#_Toc153533777)

[Dealing with Inaccuracies](#_Toc153533778)

[Suggested Improvements to the Model](#_Toc153533779)

[Suggested Improvements to the Reader](#_Toc153533780)

[Results](#_Toc153533781)

[Conclusion](#_Toc153533782)

**By students from**

[REDACTED FOR PRIVACY]

**With help from John B. from The University of Cambridge**

# Introduction

In the modern age, computers and other devices on the market are catered for people without any impairments to their mobility, vision, or hearing. This means that people who struggle in these areas, lots of technology is not available to them. We wanted to bridge the gap between people with and without disabilities and create accessible tech in the process. At the core, our project's goal is to address this issue through a proof-of-concept.

Our approach to tackling this problem is divided into two main parts:

1. **Training a Model for Sign Language Recognition** : We used a pre-captured dataset encompassing 24 out of the 26 letters of the alphabet to train a model for recognizing sign language. 2 of the letters involved movement, (J and Z) so with our current model we are not able to translate them as it requires still images. We hope to enable translation of these letters and potentially signs for words in the future.

![image](https://github.com/JahleelAbraham/SLTranslator/assets/53318509/08f6f89e-facc-4776-b6cb-4baa0f433abc)

1. **Real**** -time Hand Gesture Recognition**: Our system focuses on identifying, isolating, and transforming a live frame of a hand captured by a computer webcam. This information is then fed into the trained model.

We decided to use American Sign Language (ASL) as it only uses a single hand for the alphabet signs whereas British Sign Language (BSL) uses both hands. This would mean it was easier to train the model as we would only need one hand frame. In the future, we hope to be able to translate BSL but at the moment it is too time consuming and expensive. By using experimental data, we can build guidelines for future self-created data sets.

We designed our model based on a previously successful one from the well-known MNIST database. We did this as both involves classifying images in a set of characters.

Identifying, isolating, and transforming hand frames was carried out in a separate program. We used Computer Vision algorithms and a pre-trained model by Google created specifically for hand recognition. The data is process through this pipeline and then forwarded to our model to be recognised as signs.

In conclusion, our project deals with accessibility issues in technology by training an ASL recognition model to recognise live hand gestures and translate them to text. By focusing on ASL for practical reasons, we were able to ensure we could create a high-quality model to meet our deadlines. We hope this project can help those who need ASL translation but also inspire others to create similar projects by building off our strengths and weaknesses.

# Legal & Ethical

## Ethical Aspects of Neural Networking

The key issues of neural networking are ethicality, as neural networks are almost incapable of generating content that can be used in any industry without raising concerns. To train a neural network, an enormous dataset is required. The more data, the more sophisticated the AI model will be. Access to this information however can create issues related to intellectual property rights and confidentiality.

## Legal aspects of Neural Networking

The legal issues of this include; the artificial intelligence community, there are several approaches to modelling human intelligence. One approach applicable to the legal domain is the use of symbolic reasoning systems, which are called expert systems. These systems are called symbolic systems because they transform symbols representing things in the real world into other symbols according to explicit rules.

AI is also requiring companies to confront an evolving host of questions across different areas of law, including how it would not breach the privacy of users, how it will keep the information safe in cybersecurity.

# Training the Model

Our dataset was a CSV file. The first column had an identifier for what letter the sign was. For example, '2' was assigned to 'B'. There were 784 columns containing a grey scale value between 0 – 225 which determined the brightness of the pixel in the image of the sign.

![image](https://github.com/JahleelAbraham/SLTranslator/assets/53318509/3241b5e7-e249-4a92-88bb-7dcd84106403)

We used PyTorch data set class to create our own custom dataset. The spreadsheet was read and converted to a NumPy array. We then extracted the labels into a different array and then deleted the label in the original file, so we only had the array. The image was transformed into tensor and then normalised by 0.5 which maps the numbers to either –1 or 1.

![image](https://github.com/JahleelAbraham/SLTranslator/assets/53318509/b1d4668f-a5a5-40bb-ace5-95e1fb99c2b0)

The model was defined with these layers:

![image](https://github.com/JahleelAbraham/SLTranslator/assets/53318509/cc3a7a54-5ba6-4d1f-9104-066c0c0edcf3)

We trained and tested the model, achieving an accuracy of 87.1%

This version of the model was saved as 'Version 3'.

# Finding and Recognising a Hand and Predicting the Sign

We used CV2 to handle image processing. Media pipe, a Google service, was used to recognise hands. It initialises at 30 fps and relays the images back to user on screen. A copy is taken and sent to Media Pipe.

In order for the model to be able to predict the sign, it first needs to find the hand and the points used to determine the hand position. To make hand identification easier for the model, we changed the image into grey scale so the hand would better stand out from the background. The next step was to find the important points on the hand. These are called landmarks and represent the joints in the hand. Landmarks are used in the dataset as well as live images. We can identify signs by matching landmark configurations to letters.

![image](https://github.com/JahleelAbraham/SLTranslator/assets/53318509/d93f76b0-5f0f-49be-bacc-32e3a200d6ca)

It takes the distance of the 2 landmarks furthest apart and creates a box around the hand based on these. The image is then flipped to be the right way around and 35 pixels of padding are added to the image on all sides which ensures none of the hand is cropped out. The image is then converted to grey scale and resized to 28x28 pixels. This image is sent for processing.

The hand must be in a certain area of the camera to be read properly. If it is too close to the edge, then it is not picked up. This prevents the model from crashing.

![image](https://github.com/JahleelAbraham/SLTranslator/assets/53318509/822a91cf-5c60-4438-913d-3b29e1128273)

In the screen shot above, you can see all 3 stages of recognition. First, in the frame captured by the Webcam, you can see the box surrounding the hand with an overlayed skeleton. The white frame on the outside is the bounding box which is the boundary for reading the predicted letter. In this case, the sign is 'A', and it is being correctly identified.

Secondly, the coloured, cropped image of the hand in the top left is the isolated hand with 35 pixels of padding.

Finally, the greyscale image of the hand in the top left is resized.

# Dealing with Inaccuracies

## Suggested Improvements to the Model

We could improve the model by creating our own dataset. We would record our own images from scratch. This means we can pick the conditions the hand is in like lighting and background to ensure the data is varied enough. We could record it different ways such as images with different backgrounds, isolating the hand from the background or taking the coordinates of the skeleton landmarks.

Another way this could be improved is by separating the hand from the background entirely. This was attempted by using the colour difference between the hand and the background, finding contours around the and taking these two operations and applying them to each other to cut out any unnecessary pixels.

![image](https://github.com/JahleelAbraham/SLTranslator/assets/53318509/29a4bb28-4659-409c-8275-c3199a222319)

##

## Suggested Improvements to the Reader

Fri 24/11/2023 14:42

**Challenges encountered:**

Complexity in bounding box Resizing – One of the significant challenges we faced was correctly resizing the bounding box and normalising the hand landmark coordinates relative to this box. This process is vital in ensuring that the gestures are recognised consistently, regardless of their position or size in the video frame. The goal was to standardise the hand's position and scale across frames, which proved complex in implementation.

The first step involved dynamically determining the dimensions of a bounding box that encases the hand. This task was crucial, requiring accurate calculations of boundaries of the hand landmarks in each frame. We were able to successfully complete this and stored the values in variables.

A problem we encountered when resizing the bounding box to maintain a consistent scale across different frames. This consistency is crucial for accurate gesture recognition, as variations in scale could lead to misinterpretation of gestures.

The problem was worsened by the need to maintain the aspect ratio of the hand within the bounding box. Improper handling of aspect ratios could result in distorted images of hands, leading to inaccurate gesture recognition.

![image](https://github.com/JahleelAbraham/SLTranslator/assets/53318509/4054b780-1bea-4172-84d3-a06b8ad779bf)

Rough sketch of what we planned the resizing and moving of bounding box to look like: -

Normalisation of the hand landmarks – After resizing, the hand landmarks needed to be normalised relative to the bounding box. This normalisation involved translating and scaling the landmarks so that they remain consistent across different hand sizes and positions.

![image](https://github.com/JahleelAbraham/SLTranslator/assets/53318509/bc728a08-880a-49ec-a8ae-b250bd3da2a9)

We used the formula above to normalise each hand landmark point.

- Point xy is the hand landmark
- LL xy is the lower left boundary
- Max (H or W of the box). Numerator is divided by the larger value between height and width.

A difficulty was introduced when deciding the appropriate scale factor for normalisation. Using either height or width of the bounding box as a scale factor required careful consideration to avoid any distortion in relative positioning of the landmarks. We chose to use the maximum value of either the height or width after consulting Mr John X.

At the end of the day, we were not able to fully implement our findings into our code however, progress towards how to make our sign language predictions more accurate was made.

# Results

![image](https://github.com/JahleelAbraham/SLTranslator/assets/53318509/0c071b5f-425e-4828-bfc3-730af2b929d6)

The results in testing had an accuracy of 87%, However in practice it ended up following a normal distribution suggesting that the model is near random in real world conditions.

# Conclusion

The project has made significant strides in developing a real time hand gesture recognition system. Key functionalities, including hand tracking, bounding box calculation were successfully implemented. However, challenges in normalising and repositioning hand landmarks need further refinement. Overcoming this challenge was essential for the success of the hand gesture recognition system, as it directly impacted the accuracy and reliability of gesture detection.

We have a functioning AI module however it cannot accurately predict what sign language letters are shown. However, future research in this field could focus on developing more advanced algorithms that can dynamically adjust to the different hand sizes and orientations. Machine learning techniques may also be used to improve accuracy and efficiency.
