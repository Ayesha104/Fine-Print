# Fine-Print
Final Year Project

Checking Compliance of privacy policies with Data Protection laws using Natural Language Processing and Deep Learning techniques.

The example below shows how the compliance score is affected when the privacy policy meets the 'Data Retention' criteria of GDPR and when it doesn't.

![Alt Text](https://github.com/Ayesha104/Fine-Print/blob/master/Documentation/FinePrintDemo.gif)

The user enters a privacy policy and selects a law. Our software runs a compliance check on the privacy policy according to the law selected and generates a compliance report: 

![alt text](https://github.com/Ayesha104/Fine-Print/blob/master/Documentation/FinePrintProcess.png?raw=true)


The project is implemented in Python and to run the application, the following main installations should be made:
Numpy
Tensorflow
Bert

To run:
Open terminal and navigate to folder containing app.py file and type
python app.py

It starts running on port number 5001.

To view the compliance score, simply enter the privacy policy and select a law. For now, only GDPR and PDPA are available.
