from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QMainWindow,
    QRadioButton,
    QInputDialog,
    QLineEdit,
    QLabel,
    QTextBrowser,
    QFileDialog,
    QComboBox,
    QVBoxLayout,
    QTabWidget
)
import textract
import re
import os
from sklearn import feature_extraction, svm, preprocessing
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import numpy as np
import psycopg2 as pg
import pickle
import time
import sys # Только для доступа к аргументам командной строки

# def depth_search(path, depth = 0):
#     files = os.listdir(path)
#     for i in range(len(files)):
#         file_path = os.path.join(path, files[i])
#         print('|'*depth + files[i])
#         if os.path.isdir(file_path):
#             dir_files = depth_search(file_path, depth+1)
#             files[i] = (files[i], dir_files)
#     return files

def path_split(path):
    fp_split = []
    path = os.path.split(path)  # Splitting path into subdirectories with loop
    while (True):
        path = os.path.split(path[0])
        if (path[1] == ''):
            break
        fp_split.insert(0, path[1])
    return fp_split

def load_files(path): # Returns list of all files in selected directory and all subdirectories, plus list of categories (that is, names of directories)
    path = os.path.normpath(path)
    files_list = []
    categories = {}
    for root, dirs, files in os.walk(path): # Walking through all directories and subdirectories
        for file in files:
            filepath = os.path.join(root, file) # Full path to the file
            fp = filepath.replace(path, '') # Below we're just getting all subdirectories' names excluding the root part
            fp_split = path_split(fp)
            for i, category in enumerate(fp_split): # Finally, we're putting all obtained names in 'categories' dictionary, grouping them by level
                if (len(categories) <= i):
                    categories[i] = []
                if (category != '' and category not in categories[i]):
                    categories[i].append(category)
            files_list.append(filepath) # Then we put full path to the file in 'files_list'
    return files_list, categories

def load_data(files, categories_index=None):
    global files_root, categories
    snowball = SnowballStemmer(language='russian')
    regex = re.compile('([А-Яа-я]{2,100})')
    content = []
    if categories_index is None:
        for i in range(len(files)):
            words = list(map(str.lower, regex.findall(textract.process(files[i]).decode())))
            if(len(words) != 0):
                words = list(filter(lambda word: word not in stopwords.words('russian'), words))
                words = list(map(snowball.stem, words))
                content.append((str.join(' ', words)))
    else:
        for i in range(len(files)):
            words = list(map(str.lower, regex.findall(textract.process(files[i]).decode())))
            if(len(words) != 0):
                category = path_split(files[i].replace(files_root, ''))[categories_index]
                words = list(filter(lambda word: word not in stopwords.words('russian'), words))
                words = list(map(snowball.stem, words))
                content.append((str.join(' ', words), category))
    return content

# def val_split(dataset, frac):
#     # val_split = {}
#     val_split = np.unique([content[1] for content in dataset])
#
#     val_content = []
#     for label in val_split:
#         indices = []
#         for i in range(len(dataset)):
#             if dataset[i][1] == label:
#                 indices.append(i)
#         [val_content.append(dataset.pop(index)) for index in [indices[::frac][j]-j for j in range(len(indices[::frac]))]]
#     return val_content

def model_fit(files, categories_index):
    global model, label_encoder, vectorizer
    output = []
    print('Extracting files data, please wait')
    startTime = time.time()
    train_content = load_data(files, categories_index)
    print(f'Data loaded successfully in {round(time.time() - startTime, 2)} seconds')
    output.append(f'Data loaded successfully in {round(time.time() - startTime, 2)} seconds')
    x_train = vectorizer.fit_transform([data[0] for data in train_content]).toarray()
    y_train = label_encoder.fit_transform([data[1] for data in train_content])
    print('Fitting...')
    startTime = time.time()
    model.fit(x_train, y_train)
    print(f'Fitting completed successfully in {round(time.time() - startTime, 2)} seconds')
    output.append(f'Fitting completed successfully in {round(time.time() - startTime, 2)} seconds')
    return '\n'.join(output)

def model_predict(files):
    global model, vectorizer
    output = []
    for i in range(len(files)):
        if file_content := load_data([files[i]]):
            x_train = vectorizer.transform(file_content).toarray()
            print(f'File: {os.path.basename(files[i])}')
            output.append(f'File: {os.path.basename(files[i])}')
            prediction = model.predict_proba(x_train.reshape(1,-1))
            prediction = [round(proba, 2) for proba in prediction.reshape(-1,)]
            for j in range(len(prediction)):
                output.append(f'Класс {j} - {label_encoder.classes_[j]}: {prediction[j]*100}%')
                print(f'Класс {j} - {label_encoder.classes_[j]}: {prediction[j]*100}%')
    return '\n'.join(output)

def model_save(path, model, vectorizer, label_encoder):
    if (hasattr(model, "classes_")):
        pickle.dump(model, open(path+'/model.ml', 'wb'))
        pickle.dump(vectorizer, open(path+'/vectorizer.vct', 'wb'))
        pickle.dump(label_encoder, open(path+'/encoder.le', 'wb'))
    else:
        print("No fitted model to be saved")

def model_load(path):
    try:
        model = pickle.load(open(path+'/model.ml', 'rb'))
        vectorizer = pickle.load(open(path+'/vectorizer.vct', 'rb'))
        label_encoder = pickle.load(open(path+'/encoder.le', 'rb'))
    except:
        return "An error occured. Please, select a folder where a saved model is located."
    return model, vectorizer, label_encoder

model = svm.SVC(kernel='linear', probability=True, cache_size=2000)
label_encoder = preprocessing.LabelEncoder()  # Encodes class labels to indices
vectorizer = feature_extraction.text.TfidfVectorizer()
working_directory = 'D:/БИБЛИОТЕКА'
files_root = ''
train_files = []
categories = {}

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("App")

        widget = TabsWidget()

        self.setCentralWidget(widget)

class TabsWidget(QWidget):

    def __init__(self):
        super(QWidget, self).__init__()
        self.layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.tab1 = Tab1()
        self.tab2 = Tab2()

        self.tabs.addTab(self.tab1, "Fit")
        self.tabs.addTab(self.tab2, "Predict")

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

class Tab1(QWidget):

    def __init__(self):
        super(QWidget, self).__init__()

        self.layout = QVBoxLayout()
        self.filepath = QLineEdit()
        self.browse = QPushButton("Select train data")
        self.category_label = QLabel("Select categories to train on")
        self.category_choice = QComboBox()
        self.fit = QPushButton("Fit")
        self.save = QPushButton("Save")
        self.load = QPushButton("Load")
        self.output = QTextBrowser()

        self.browse.clicked.connect(self.browseClick)
        self.fit.clicked.connect(self.fitClick)
        self.save.clicked.connect(self.saveClick)
        self.load.clicked.connect(self.loadClick)

        self.layout.addWidget(self.filepath)
        self.layout.addWidget(self.browse)
        self.layout.addWidget(self.category_choice)
        self.layout.addWidget(self.fit)
        self.layout.addWidget(self.output)
        self.layout.addWidget(self.save)
        self.layout.addWidget(self.load)
        self.setLayout(self.layout)

    def browseClick(self):
        # path = QFileDialog.getOpenFileName(self, "Select Document", "", "Document Files (*.pdf)")[0]
        # self.filepath.setText(path)
        global train_files, files_root, categories
        files_root = os.path.normpath(
            QFileDialog.getExistingDirectory(self, "Select Folder", directory=working_directory))
        train_files, categories = load_files(files_root)
        self.output.append(f'Added {len(train_files)} files from {files_root}.\n')
        self.category_choice.clear()
        for key in categories:
            self.category_choice.addItem(f'{key}: ' + str.join(', ', [c for c in categories[key]]))

    def fitClick(self):
        self.output.append(
            model_fit(train_files, self.category_choice.currentIndex()))

    def saveClick(self):
        global model, vectorizer, label_encoder
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.Directory)
        path = QFileDialog.getExistingDirectory(self, "Select Folder", "")
        model_save(path, model, vectorizer, label_encoder)
        self.output.append(f'Successfully saved model to {path}.\n')

    def loadClick(self):
        global model, vectorizer, label_encoder
        dialog = QFileDialog(self)
        dialog.setFileMode(2)
        path = QFileDialog.getExistingDirectory(self, "Select Folder", "")
        model, vectorizer, label_encoder = model_load(path)
        self.output.append(f'Loaded model from {path}.\n')

class Tab2(QWidget):

    def __init__(self):
        super(QWidget, self).__init__()

        self.layout = QVBoxLayout()
        self.filepath = QLineEdit()
        self.label = QLabel("Select files to make predictions for")
        self.predict = QPushButton("Browse")
        self.output = QTextBrowser()

        self.predict.clicked.connect(self.predictClick)

        self.layout.addWidget(self.filepath)
        self.layout.addWidget(self.predict)
        self.layout.addWidget(self.output)
        self.setLayout(self.layout)

    def predictClick(self):
        # path = QFileDialog.getOpenFileName(self, "Select Document", "", "Document Files (*.pdf)")[0]
        # self.filepath.setText(path)
        predict_files = QFileDialog.getOpenFileNames(self, "Select Files", directory=working_directory, filter="*.pdf")[0]
        print(predict_files)
        model_predict(predict_files)

app = QApplication(sys.argv)

# Создаём виджет Qt — окно.
window = MainWindow()
window.show()  # Важно: окно по умолчанию скрыто.

# Запускаем цикл событий.
app.exec()


# Приложение не доберётся сюда, пока вы не выйдете и цикл
# событий не остановится.