import easyocr
import os
from rich.progress import track
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import spacy

nlp = spacy.load("en_core_web_md")

allowed_text = ["GEOGUESSR", "playing", "MAP", "ROUND", "SCORE", "World", "1/5", "2/5", "3/5", "4/5", "5/5", "0", "NORTH", "AMERICA", "EUROPE", "ASIA", "AFRICA", "SOUTH", "OCEANIA", "GOOGLE",
                "{", "}", "(", ")", "[", "]", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "Keyboard shortcuts"]


img_folder = "Roy/Test_Images"
reader = easyocr.Reader(['en', 'fr', 'de', 'es', 'it', 'pt'])
reader2 = easyocr.Reader(["ar","fa","ur","ug","en"])
reader3 = easyocr.Reader(["en","ja"])
reader4 = easyocr.Reader(["en","ko"])
reader5 = easyocr.Reader(["en","ch_sim"])
reader6 = easyocr.Reader(["hi","mr","ne","en"])
reader7 = easyocr.Reader(["ru","rs_cyrillic","be","bg","uk","mn","en"])

for image in track(os.listdir(img_folder)):
    if image.endswith(".jpg"):
        img_path = os.path.join(img_folder, image)
        result = reader.readtext(img_path)
        result2 = reader2.readtext(img_path)
        result3 = reader3.readtext(img_path)
        result4 = reader4.readtext(img_path)
        result5 = reader5.readtext(img_path)
        result6 = reader6.readtext(img_path)
        result7 = reader7.readtext(img_path)

        result.extend(result2)
        result.extend(result3)
        result.extend(result4)
        result.extend(result5)
        result.extend(result6)
        result.extend(result7)
        print(f"Results for {image}:")
        for (bbox, text, prob) in result:
            # check if the recognized text is allowed, we allow for spelling mistakes, so get similarities using spacy for each word in allowed_text
            if any(nlp(text).similarity(nlp(word)) > 0.8 for word in allowed_text):
                continue
            if prob < 0.5:
                continue
            print(f"  {text} (confidence: {prob:.2f})")
        print("------------------------------------------------------------------------------")
        # display the image
        img = Image.open(img_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

