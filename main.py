# # from flask import Flask, request, jsonify, render_template
# # from transformers import RobertaTokenizer, RobertaForSequenceClassification
# # from bs4 import BeautifulSoup
# # from langdetect import detect
# # import torch
# # import json
# # import os

# # app = Flask(__name__)
# # device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# # # Load model and tokenizer
# # MODEL_PATH = "pankaj100567/Intent-classification"
# # tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
# # model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
# # model.eval().to(device)

# # # Load label mappings from a JSON file
# # solution_file_path = os.path.join('surprise.solution')
# # with open(solution_file_path, 'r') as solutions_file:
# #     labels = [json.loads(line)['intent'] for line in solutions_file]

# # label2id = {label: i for i, label in enumerate(set(labels))}
# # id2label = {i: label for label, i in label2id.items()}

# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# # @app.route('/classify', methods=['POST'])
# # def classify():
# #     try:
# #         sentence = request.form['sentence']
# #         soup = BeautifulSoup(sentence, "html.parser")
# #         cleaned_sentence = soup.get_text().strip()

# #         if detect(cleaned_sentence) != 'en':
# #             return jsonify({"error": "Please enter the sentence in English."})

# #         encodings = tokenizer(cleaned_sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
# #         input_ids = encodings['input_ids'].to(device)
# #         attention_mask = encodings['attention_mask'].to(device)

# #         with torch.no_grad():
# #             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
# #             logits = outputs.logits
# #             probabilities = torch.softmax(logits, dim=1)
# #             predicted_class_index = probabilities.argmax().item()

# #         predicted_intent = id2label[predicted_class_index]
# #         return jsonify({"intent": predicted_intent, "sentence": cleaned_sentence})

# #     except Exception as e:
# #         return jsonify({"error": str(e)})

# # if __name__ == '__main__':
# #     app.run(debug=True)

# from flask import Flask, request, jsonify, render_template
# from transformers import RobertaTokenizer, RobertaForSequenceClassification
# from bs4 import BeautifulSoup
# from langdetect import detect
# from torch.utils.data import DataLoader, TensorDataset
# import json
# import torch
# import os

# app = Flask(__name__)
# device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# cache_dir = "/code/cache/huggingface"
# if not os.path.exists(cache_dir):
#     try:
#         os.makedirs(cache_dir)
#         os.chmod(cache_dir, 0o777)  # Set directory permissions to read, write, and execute by all users
#     except Exception as e:
#         print(f"Failed to create or set permissions for directory {cache_dir}: {e}")

# # cache_dir = "/code/cache/huggingface"
# # if not os.path.exists(cache_dir):
# #     os.makedirs(cache_dir)
# # Load model and tokenizer
# MODEL_PATH = "pankaj100567/Intent-classification"
# tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH, cache_dir=cache_dir)
# model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH, cache_dir=cache_dir, num_labels=150)
# # model.eval().to(device)


# # Load label mappings
# solution_file_path=os.path.join('surprise.solution')
# # test_data_path=os.path.join(data_path,'massive_test.data')
# # loading surprise.solution file for getting id2label and label2id mapping
# with open(solution_file_path,'r') as solutions_file:
#     solutions=[json.loads(line) for line in solutions_file] # reading json data from data_path and parse it into a test_data list

# labels_list=[]
# for label in solutions:
#     labels_list.append(label['intent'])
# unique_labels_list=[]
# for x in labels_list:
#     if x not in unique_labels_list:
#         unique_labels_list.append(x)
# # unique_labels_list, len(unique_labels_list)

# label2id={}
# id2label={}
# for i, label in enumerate(unique_labels_list):
#     label2id[label]=i
#     id2label[i]=label
# # # Load label mappings from a JSON file
# # solution_file_path = os.path.join('surprise.solution')
# # with open(solution_file_path, 'r') as solutions_file:
# #     labels = [json.loads(line)['intent'] for line in solutions_file]

# # label2id = {label: i for i, label in enumerate(set(labels))}
# # id2label = {i: label for label, i in label2id.items()}
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/classify', methods=['POST'])
# def classify():
#     try:
#         sentence = request.form['sentence']
#         soup = BeautifulSoup(sentence, "html.parser")
#         cleaned_sentence = soup.get_text().strip()

#         if detect(cleaned_sentence) != 'en':
#             return jsonify({"error": "Please enter the sentence in English."})

#         encodings = tokenizer(cleaned_sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
#         input_ids = encodings['input_ids'].to(device)
#         attention_mask = encodings['attention_mask'].to(device)
#         # Create a TensorDataset
#         test_dataset = TensorDataset(input_ids, attention_mask,)

#         # Define batch size
#         batch_size = 32

#         # Create a DataLoader
#         test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
#         # Set the model in evaluation mode
#         model.eval()

#         # Iterate through the batches in the DataLoader
#         for batch in test_dataloader:
#             # Unpack the batch
#             input_ids, attention_mask = batch

#             # Move tensors to the device (e.g., GPU if available)
#             input_ids = input_ids.to(device)
#             attention_mask = attention_mask.to(device)


#             # Forward pass to get logits
#             with torch.no_grad():
#                 outputs = model(input_ids=input_ids, attention_mask=attention_mask)

#             # Extract the logits tensor from the outputs
#             logits = outputs.logits

#             # Apply softmax to get class probabilities
#             probabilities = torch.softmax(logits, dim=1)

#             # Get the predicted class (index with the highest probability)
#             predicted_class = torch.argmax(probabilities, dim=1)

            
#             # Append the predicted class to the list of predictions
#             # predictions.extend(predicted_class.tolist())

#         # with torch.no_grad():
#         #     outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         #     logits = outputs.logits
#         #     probabilities = torch.softmax(logits, dim=1)
#         #     predicted_class_index = probabilities.argmax().item()

#         predicted_intent = id2label[predicted_class]
#         print(predicted_class, predicted_intent)
#         return jsonify({"intent": predicted_intent, "sentence": cleaned_sentence})

#     except Exception as e:
        # return jsonify({"error": str(e)})
from flask import Flask, request, jsonify, render_template
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from bs4 import BeautifulSoup
from langdetect import detect
from torch.utils.data import DataLoader, TensorDataset
import json
import torch
import os

app = Flask(__name__)
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
cache_dir = "/code/cache/huggingface"
if not os.path.exists(cache_dir):
    try:
        os.makedirs(cache_dir)
        os.chmod(cache_dir, 0o777)  # Set directory permissions to read, write, and execute by all users
    except Exception as e:
        print(f"Failed to create or set permissions for directory {cache_dir}: {e}")

# cache_dir = "/code/cache/huggingface"
# if not os.path.exists(cache_dir):
#     os.makedirs(cache_dir)
# Load model and tokenizer
MODEL_PATH = "pankaj100567/Intent-classification"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH, cache_dir= cache_dir)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH, cache_dir= cache_dir, num_labels=150)
# model.eval().to(device)


# Load label mappings
solution_file_path=os.path.join('surprise.solution')
# test_data_path=os.path.join(data_path,'massive_test.data')
# loading surprise.solution file for getting id2label and label2id mapping
with open(solution_file_path,'r') as solutions_file:
    solutions=[json.loads(line) for line in solutions_file] # reading json data from data_path and parse it into a test_data list

labels_list=[]
for label in solutions:
    labels_list.append(label['intent'])
unique_labels_list=[]
for x in labels_list:
    if x not in unique_labels_list:
        unique_labels_list.append(x)
# unique_labels_list, len(unique_labels_list)

label2id={}
id2label={}
for i, label in enumerate(unique_labels_list):
    label2id[label]=i
    id2label[i]=label
# # Load label mappings from a JSON file
# solution_file_path = os.path.join('surprise.solution')
# with open(solution_file_path, 'r') as solutions_file:
#     labels = [json.loads(line)['intent'] for line in solutions_file]

# label2id = {label: i for i, label in enumerate(set(labels))}
# id2label = {i: label for label, i in label2id.items()}
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        sentence = request.form['sentence']
        soup = BeautifulSoup(sentence, "html.parser")
        cleaned_sentence = soup.get_text().strip()

        if detect(cleaned_sentence) != 'en':
            return render_template('result.html', error="Please enter the sentence in English.")

        encodings = tokenizer(cleaned_sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)

        test_dataset = TensorDataset(input_ids, attention_mask)
        test_dataloader = DataLoader(test_dataset, batch_size=1)  # Assume a batch size of 1 for individual predictions
        model.eval()
        
        for batch in test_dataloader:
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()

        predicted_intent = id2label[predicted_class]
        return render_template('result.html', intent=predicted_intent, sentence=cleaned_sentence)

    except Exception as e:
        return render_template('result.html', error=str(e))


# if __name__ == '__main__':
#     app.run(debug=True)
