from flask import Flask, request, jsonify, render_template_string
import librosa
import numpy as np
import tensorflow as tf
import tempfile
import os

app = Flask(__name__)
model = tf.keras.models.load_model("C:\\Users\\Manolis\Desktop\\MY_LSTM_MODEL.keras")  # Load your LSTM model
#http://127.0.0.1:5000/

def pad_or_truncate(feature, target_length):
    if feature.shape[1] > target_length:
        return feature[:, :target_length]
    elif feature.shape[1] < target_length:
        pad_width = target_length - feature.shape[1]
        return np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
    else:
        return feature

def extract_features(file_path, target_frames=130):
    y, sr = librosa.load(file_path, sr=22050)
    y = y[:sr * 3]  # Keep only the first 3 seconds

    # Extract MFCC and Mel Spectrogram
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Pad or truncate both features to match the target_frames
    
    mel_db = pad_or_truncate(mel_db, target_frames)

    # Stack features (MFCC + Mel) along the feature dimension
    features = np.vstack((mfccs, mel_db))  # Shape: (13+128, target_frames)

    return features.T.reshape(1, target_frames, -1)  # Final shape: (1, target_frames, 141)
    

@app.route("/")
def upload_page():
    return render_template_string('''
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload Song</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: url("https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgzyOlNIh9jimgvkoZv2v6dziBA27kVZwOqld485xBz3RsIZzK6OrLTP5ZFqKWJ0zWbX3EmEuimm8jsxL41YAx3fhNfPv5-g4_URhR33uYwk-IM4ayGeDjQ-pdcVV0EdHUUaBeWbkY2XUA7IYy4E4xgmE2sep50CaLAsZf-cpn1cg0-2jhYt-G4pirdg67H/s1600/colorful_piano_keys_4k.png") no-repeat center center fixed;
                background-size: cover;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                color: white;
                font-size: 100% ;                 
                
            }
            .container {
                background: rgba(0, 0, 0, 0.7);
                padding: 5.5%;
                border-radius: 50%;
                text-align: start;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
            }
            h2 {
                margin-bottom: 20px;
                                
            }
            .file-input {
                position: relative;
                display: inline-block;
                width: 100%;
                max-width: 300px;
            }
            .file-input input {
                display: none;
            }
            .file-label {
                display: block;
                padding: 5%;
                background: white;
                color: black;
                border-radius: 5px;
                cursor: pointer;
                text-align: center;
                font-size: 18px;                  
            }
            .file-name {
                display: block;
                margin-top: 10px;
                font-size: 14px;
                color: #ddd;
            }
            button {
                background: blue;
                color: white;
                border: none;
                padding: 3.8% ;
                border-radius: 5px;
                cursor: pointer;
                font-size: 17px;
                margin-top:  -6%;
                
            }
            button:hover {
                background: #black;
            }
            form {
                    display: flex;
                    align-items: center; /* Vertically align the buttons */
                    gap: 10%; /* Add a 10px gap between the buttons */
                }
        </style>
    </head>
    <body>
        <div class="container">
            <h2
               style=" text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6); 
              font-size: 25px; 
              font-weight: bold; 
              "
              >
            Upload a Song for Genre Prediction</h2>
                                  
            <form action="/predict" method="post" enctype="multipart/form-data">
                <div class="file-input">
                    <label for="file" class="file-label">Choose a file</label>
                    <input type="file" id="file" name="file" required onchange="updateFileName()">
                    <span class="file-name" id="file-name">No file chosen</span>
                </div>
                <button type="submit">Upload</button>
            </form>
        </div>

        <script>
            function updateFileName() {
                var input = document.getElementById('file');
                var fileName = input.files.length > 0 ? input.files[0].name : "No file chosen";
                document.getElementById('file-name').textContent = fileName;
            }
        </script>
    </body>
    </html>
''')



@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file.save(tmp.name)
        features = extract_features(tmp.name)
        #os.unlink(tmp.name)


    prediction = model.predict(features)[0]
    percentages = (prediction/ prediction.sum()) * 100  # Normalize to 100%
    rounded_percentages = np.round(percentages, 2)    
    #prediction = model.predict(features)
    print("Raw prediction:", prediction)  # See softmax values
    print("Argmax index:", np.argmax(prediction))  # Get softmax output
    genre_index = np.argmax(prediction)  # Find most probable class
    genres = ["blues", "classical" , "unknown"]  # Ensure matches model output


    unknown_index = genres.index("unknown")
    if rounded_percentages[unknown_index] >= 0.014:
        genre = "unknown"
    else:
        genre_index = np.argmax(prediction)
        genre = genres[genre_index]


    #genre = genres[genre_index]



    return render_template_string('''
    <!doctype html>
    <html>
    <head>
        <title>Genre Prediction</title>
        <style>
            body { font-family: Arial, sans-serif;
                background: url("https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgzyOlNIh9jimgvkoZv2v6dziBA27kVZwOqld485xBz3RsIZzK6OrLTP5ZFqKWJ0zWbX3EmEuimm8jsxL41YAx3fhNfPv5-g4_URhR33uYwk-IM4ayGeDjQ-pdcVV0EdHUUaBeWbkY2XUA7IYy4E4xgmE2sep50CaLAsZf-cpn1cg0-2jhYt-G4pirdg67H/s1600/colorful_piano_keys_4k.png") no-repeat center center fixed;
                text-align: center; padding: 50px; 
                background-size: cover;
                display: flex; }
            .container { max-width: 60%;
                    margin-top: -4%; 
                    margin-left: auto;
                    margin-right: auto;
                    padding: 3%; 
                    background: white;
                    border-radius: 50%; 
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); }
            h2 { color: #333; }
            .back-btn {
                 display: inline-block; margin-top: 20px; padding: 10px 15px; 
                background: #007BFF; color: white; text-decoration: none;
                border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Prediction Percentages:<br>
                Blues: {{ prediction[0] }}% &nbsp;&nbsp;
                Classical: {{ prediction[1] }}% &nbsp;&nbsp;
                Unknown: {{ prediction[2] }}%
            </h2>                                                 
    <h2 style="color: black; font-weight: bold; text-shadow: 0 0 8px #007BFF, 0 0 10px #007BFF, 0 0 15px #00bfff; animation: bounce 1s infinite; font-size: 2em; background-color: rgba(255, 255, 255, 0.7); padding: 5px 10px; border-radius: 5px;">Predicted Genre: {{ genre }}</h2>

         <style>
                @keyframes bounce {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-10px); }
                }
            </style>  <!–  This style is only for the prediction line to be bouncing */ –>

            <a href="/" class="back-btn">Upload Another Song</a>
        </div>
    </body>
    </html>
''', genre=genre, prediction= rounded_percentages )
    
    #return jsonify({"genre": genre})

if __name__ == "__main__":
    app.run(debug=True)
