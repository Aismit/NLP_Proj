
#importing the libraries
print("\nStep 1: Importing Necessary Libraries\n")
import json, re, string, sklearn, json, os.path, keras, pandas, pickle
from sklearn.model_selection import train_test_split
from sklearn import ensemble, feature_extraction, linear_model, pipeline, metrics, naive_bayes, neural_network, svm
import numpy as np
from datetime import datetime
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from keras.utils import to_categorical
from keras import Sequential
from keras.preprocessing.text import Tokenizer
import keras.layers
from keras.preprocessing.sequence import pad_sequences
import MacOSFile
#preprocessing the data
print("\nStep 2: Preprocessing the Data\n")

#defining general clean string method for cleaning the data 
def clean_string(string, var = False):
	"""
	Tokenization/string cleaning for all datasets 
	"""
	
	"""removes bullet points"""
	string = re.sub("\\u2022", "", string) 
	string = re.sub("\\u00b7", "", string)
	
	"""removes numbers"""
	string = re.sub(r"[0-9]","", string)
	
	"""removes punctuation and other symbols"""
	filters='!"\#$%&()*+-/:;,<=>?@[\\]^`{|}~\t\n'
	translate_dict = dict((c, " ") for c in filters)
	translate_map = str.maketrans(translate_dict)
	filters2='.'
	translate_dict2 = dict((c, "_") for c in filters2)
	translate_map2 = str.maketrans(translate_dict2)
	string = string.translate(translate_map)
	string = string.translate(translate_map2)
	
	"""removes unnecessary spaces"""
	string = re.sub('\s+', ' ', string).strip()
	return string.strip() if var else string.strip().lower()


#setting up the data frame

#setting up the x value
training_X_data = open("train_X_languages.json.txt")
X_train_content= training_X_data.readlines()
X = [json.loads(obj)['text'] for obj in X_train_content]
X_clean = [clean_string(elem) for elem in X] 

#setting up the y value
training_Y_data = open("train_y_languages.json.txt")
Y_train_content= training_Y_data.readlines()
Y = [json.loads(obj)['classification'] for obj in Y_train_content]
Y_value_labels = list(set(Y))
#encoding the y value into a number
lang_encode = dict(zip(set(Y),range(len(Y))))
lang_encode_flipped = {}
for key,value in lang_encode.items():
	lang_encode_flipped[value] = key

#combining the values for the dataframe
trainDF = pandas.DataFrame()
trainDF['text'] = X
trainDF['text_clean'] = X
trainDF['label'] = Y
trainDF['label_encode'] = [lang_encode[i] for i in Y]

training_X_data.close()
training_Y_data.close()

#splitting the data into test and train set 
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.2)
# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', ngram_range=(1,6))
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(test_x) 

#now for training 
print("Step 3: Training the Different Models \n")


def train_model(classifier, feature_vector_train, label, feature_vector_valid, label_valid, is_neural_net=False,ep=1):
	# fit the training dataset on the classifier
	#print(label_valid)

	if is_neural_net:
		classifier.fit(feature_vector_train, label,epochs=ep)
		predictions = classifier.predict(feature_vector_valid)
		predictions = predictions.argmax(axis=-1)
	
	else:
		# fit the training dataset on the classifier
		classifier.fit(feature_vector_train, label)
	
		# predict the labels on validation dataset
		predictions = classifier.predict(feature_vector_valid)
 
	return predictions, metrics.accuracy_score(predictions, label_valid)
	

#feel free to comment out the model that doesn't need to be compiled


new_file = open("performance.txt", "w")

print("Sarting Logistic Regression - it might be a bit slower than the other models so feel free to comment out as it wasn't the model I used for the predictions")
starting = datetime.now()
log_reg_model = linear_model.LogisticRegression()
y_predictions_lr, accuracy_lr = train_model(log_reg_model, xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars, test_y)
logistic_regression_results = metrics.classification_report(test_y, y_predictions_lr,target_names=Y_value_labels)
time_lr = datetime.now() - starting
print("Done with LogisticRegression")
print("Accuracy is: ", accuracy_lr)
print("Time for Logistic Regression is :", time_lr)
print("\n\n")

new_file = open("performance.txt", "w")
new_file.write("First: Report for Naive Bayes\n\n")
new_file.write("Accuracy is: " + str(accuracy_lr))
new_file.write("\n")
new_file.write("Time is:\n" + str(time_lr))
new_file.write("\n")
new_file.write("\n")
new_file.write("Classification Report is: \n")
new_file.write(logistic_regression_results)
new_file.write("\n")
new_file.write("\n")

print("Starting Convolutional LSTM Neural Network\n")
starting = datetime.now()
vocabulary_size = 5394
tokenizer = Tokenizer(num_words= vocabulary_size,char_level=True)
tokenizer.fit_on_texts(trainDF['text'])
sequences = tokenizer.texts_to_sequences(trainDF['text'])
data = pad_sequences(sequences, maxlen=200)
y_binary = to_categorical(trainDF["label_encode"])


def create_conv_model():
    model_conv = Sequential()
    model_conv.add(layers.Embedding(vocabulary_size, 100, input_length=200))
    model_conv.add(layers.Dropout(0.2))
    model_conv.add(layers.Conv1D(64, 5, activation='relu'))
    model_conv.add(layers.MaxPooling1D(pool_size=4))
    model_conv.add(layers.LSTM(100))
    model_conv.add(layers.Dense(56, activation='softmax'))
    model_conv.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_conv

model_conv = create_conv_model()


x_train, x_test, y_train,y_test = train_test_split(data,trainDF["label_encode"],test_size=0.2)
y_train = to_categorical(y_train)
y_predictions_neural_net, accuracy_neural_net = train_model(model_conv,x_train, y_train, x_test, y_test,is_neural_net=True, ep=3)

neural_net_results = metrics.classification_report(y_predictions_neural_net, y_test,target_names=Y_value_labels)
time_nn = datetime.now() - starting
print("Done with LogisticRegression")
print("Accuracy is: ", accuracy_neural_net)
print("Time for Logistic Regression is :", time_nn)
print("\n\n")

new_file = open("performance.txt", "w")
new_file.write("Second: Report for LSTM Neural Net\n\n")
new_file.write("Accuracy is: " + str(accuracy_neural_net))
new_file.write("\n")
new_file.write("Time is:" + str(time_nn))
new_file.write("\n")
new_file.write("\n")
new_file.write("Classification Report is: \n")
new_file.write(neural_net_results)
new_file.write("\n")
new_file.write("\n")



print("Starting Linear Support Vector One vs All Classification")
starting = datetime.now()
svc_model = svm.LinearSVC()
y_predictions_svc, accuracy_svc = train_model(svc_model, xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars,test_y ,is_neural_net=False)
svc_results = metrics.classification_report(test_y, y_predictions_svc,target_names=Y_value_labels)
time_svc = datetime.now() - starting
print("Done with Linear Support Vector")
print("Accuracy is: ", accuracy_svc)
print("Time for Linear Support Vector is :", time_svc)
print("\n\n")

new_file = open("performance.txt", "w")
new_file.write("Third: Report for Linear Support Vector \n")
new_file.write("Accuracy is:" + str(accuracy_svc))
new_file.write("\n")
new_file.write("Time is:" + str(time_svc))
new_file.write("\n")
new_file.write("\n")
new_file.write("Classification Report is: \n")
new_file.write(svc_results)
new_file.write("\n")
new_file.write("\n")


new_file.close()

#storing the model with the corresponding vectorizer for reshaping the input

best_model_filename = "best_model.bin"

MacOSFile.pickle_dump(svc_model,best_model_filename)

vectorizer_file = "vectorizer.bin"

MacOSFile.pickle_dump(tfidf_vect_ngram_chars,vectorizer_file)

print("Complete\n")







