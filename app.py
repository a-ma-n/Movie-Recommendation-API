from flask import Flask, render_template, request
import pickle
import numpy as np

from movierecfinal import *

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json
        # response = keyword_s(data['keyword'])
        # normalized=cv.transform([text])
        # result=lr.predict(normalized)
        print(data["text"])
        print(get_recommendations("The Dark Knight Rises"))
        result=get_recommendations(data['text'])
        print("\n\n\nresult: \t\t\t",result)
        return result
        # if(result==1):
        #     return render_template('index.html',prediction_text="Positive Review")
        # else:
        #     return render_template('index.html',prediction_text="Negative Review")
    #else:
     #   return render_template('index.html',prediction_texts="Positive Review")

if __name__=="__main__":
    app.run(debug=True)