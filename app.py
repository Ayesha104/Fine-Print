from flask import Flask, render_template, flash, redirect, request, session, url_for
from wtforms import Form, StringField, TextAreaField, validators
from SimilarityFinal import getCompliance
from BERTClassifier import Categorize
import sys
#from celery import Celery
import numpy as np
app = Flask(__name__)

'''CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'


client = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
client.conf.update(app.config)

@client.task
def get_results(policyText, law):
    
    #print(policyText, file=sys.stderr)
    PolicyFile = CategorizePolicy(policyText, law)
    #flash(PolicyFile, "success")
    #flash(policyText, "success")
    #flash(filee, "success")
    complianceResults = FindCompliance(PolicyFile)
    imageNames, complianceResults = getImageName(complianceResults)
    #print(imageNames)

    # print(imageNames['CompScore'])

    return render_template('compliance.html', imageNames=imageNames, complianceResults=complianceResults)
'''

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/processing')
def processing():
    return render_template('processing.html')

@app.route('/compliance')
def compliance():
    return render_template('compliance.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/about')
def about():
    return render_template('about.html')


class PolicyForm(Form):
    policyText = StringField("", [validators.Length(min=50, max=5000)])


def getImageName(complianceResults):
    imgs = [0, 5, 10, 15, 25, 30, 35, 40, 50, 60, 65, 70, 75,  
    80, 90, 95, 100]
    arrayImg = np.asarray(imgs)
    imgName = []
    for i in range(1,5):
        score = complianceResults["GDPR"+str(i)]
        idx = (np.abs(arrayImg - score)).argmin()
        imgName.append(arrayImg[idx])
    idx = (np.abs(arrayImg - complianceResults["compliance"])).argmin()
    score = (arrayImg[idx])
    complianceResults["compliance"] = str(complianceResults["compliance"])+"%"
    complianceResults["GDPR1"] = str(complianceResults["GDPR1"])+"%"
    complianceResults["GDPR2"] = str(complianceResults["GDPR2"])+"%"
    complianceResults["GDPR3"] = str(complianceResults["GDPR3"])+"%"
    complianceResults["GDPR4"] = str(complianceResults["GDPR4"])+"%"
    return {
        "CompScore": "../static/img/"+str(score)+".png",
        "GDPR1" : "../static/img/"+str(imgName[0])+".png",
        "GDPR2" : "../static/img/"+str(imgName[1])+".png",
        "GDPR3" : "../static/img/"+str(imgName[2])+".png",
        "GDPR4" : "../static/img/"+str(imgName[3])+".png",
    }, complianceResults



@app.route('/policy', methods = ['GET', 'POST'])
def policy():
    #print("hooPolicy")
    lawFileName = "GDPRSegments.csv"
    form =PolicyForm(request.form)
    if request.method == "POST":# and form.validate():#if button pressed
        policyText = request.form['policyText']#form.policyText.data
        law = request.form['lawSelected']
        filee = request.form['policyFile']
        if(filee):
            PolicyFile = Categorize(filee)
        #flash(law, "success")
        ##get_results.apply_async(args=[data])
        #print(policyText, file=sys.stderr)
        #policyText=policyText.encode('utf-8').strip()
        else:
            fileNamepolicy = open("policytxt.txt", "w",encoding='utf-8')
            fileNamepolicy.write(policyText)
            fileNamepolicy.close()
            PolicyFile = Categorize("policytxt.txt")
        #flash(PolicyFile, "success")
        #flash(policyText, "success")
        #flash(filee, "success")
        if(law=="GDPR"):
            lawFileName = "GDPRSegments.csv"
        complianceResults = getCompliance(PolicyFile, lawFileName)#FindCompliance(PolicyFile)
        imageNames, complianceResults = getImageName(complianceResults)
        #print(imageNames)

       # print(imageNames['CompScore'])



        return render_template('compliance.html', imageNames=imageNames, complianceResults=complianceResults)
        #redirect(url_for('compliance'), imageNames=imageNames, complianceResults=complianceResults) #takes method mame as input and returns the route
    return render_template('policy.html', form=form)


        ##return redirect(url_for('processing')) #takes method mame as input and returns the route
    ##return render_template('policy.html', form=form)


if __name__ == '__main__':
    app.secret_key = b'_5#y2L"R104z\n\xec]/'
    app.run(debug=True, port=5001)