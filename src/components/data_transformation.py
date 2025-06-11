import sys
import os
import re
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import USvisaException
from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact

nltk.download('stopwords')

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def clean_text(text):
        try:
            if isinstance(text, str):
                text = text.lower()
                text = re.sub('\[.*?\]', '', text)
                text = re.sub('https?://\S+|www\.\S+', '', text)
                text = re.sub('<.*?>+', '', text)
                text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
                text = re.sub('\n', '', text)
                text = re.sub('[^a-zA-Z]', ' ', text)
                text = re.sub('\s+[^a-zA-Z]\s+', '', text)
                text = re.sub('\w*\d\w*', '', text)
                pattern = re.compile(r'\b(' + '|'.join(stopwords.words('english')) + r')\b\s*')
                text = re.sub(pattern, '', text)
            return text
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def use_porter(text):
        try:
            ps = PorterStemmer()
            text = str(text)  # Convert to string first
            return ' '.join([ps.stem(word) for word in text.split()])
        except Exception as e:
            raise USvisaException(e, sys)


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
           
            logging.info("Reading train and test files")
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            for df in [train_df, test_df]:
                df["v1"] = df["v1"].apply(self.clean_text)
                df["v1"] = df["v1"].apply(self.use_porter)

            logging.info("Text cleaning and stemming completed")

            # Tokenizer & Padding
            tokenizer = Tokenizer(num_words=15000)
            tokenizer.fit_on_texts(train_df["v1"])

            x_train_seq = tokenizer.texts_to_sequences(train_df["v1"])
            x_test_seq = tokenizer.texts_to_sequences(test_df["v1"])

            x_train = pad_sequences(x_train_seq, maxlen=150)
            x_test = pad_sequences(x_test_seq, maxlen=150)

            y_train = train_df[TARGET_COLUMN]
            y_test = test_df[TARGET_COLUMN]

            label_replace = {"ham":0,'spam':1}
            y_train = y_train.map(label_replace)
            y_train = y_train.values
            y_test = y_test.map(label_replace)
            y_test = y_test.values

            # Save arrays
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, np.c_[x_train, y_train])
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, np.c_[x_test, y_test])

            # Save tokenizer
            save_object(self.data_transformation_config.transformed_object_file_path, tokenizer)

            logging.info("Data transformation completed and objects saved")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise USvisaException(e, sys)