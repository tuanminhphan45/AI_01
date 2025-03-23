import sys 

from dataclasses import dataclass
import numpy as np # type: ignore
import pandas as pd
from sklearn.compose import ColumnTransformer #use to create piplines
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException # errros
from src.logger import logging # recored logs
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_object(self):
        ''''
        This function is used to create a preprocessor object which will be used to transform the data
        '''
 
        try:
            numerical_columns = ['wrinting_score','reading_score','math_score']
            categorical_columns = ['gender','race_ethnicity',
            'parental_level_of_education','lunch','test_preparation_course']
            

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")), # replace missing values with median
                ('std_scaler', StandardScaler()), # standardize the data
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")), # replace missing values with median
                ('onehot', OneHotEncoder()), # one hot encoding
                ('std_scaler', StandardScaler()), # standardize the data
            ])
            logging.info("numerical colums endcoding completed")
            logging.info("Categorical colums endcoding completed")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns), #(pipline_name,what pipline to use,columns_data)
                    ('cat_pipeline', cat_pipeline, categorical_columns),
                ])
            logging.info("preprocessor object created")

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        '''
        This function is used to initiate the data transformation process
        '''
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("read train and test data loaded")

            logging.info("Creating preprocessor object")
            preprocessor=self.get_data_transformer_object()

            target_column_name='math_score'
            numerical_columns = ['wrinting_score','reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"applying preprocessor object on training df and testing df")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ] 
            """"
            np.c_[]:This is shorthand for concatenating arrays along the second axis (columns).
            It's combining the input features and the target feature into one big array — row by row — like this:
                [input1, input2, ..., inputN, target]
                [input1, input2, ..., inputN, target]
            """
            
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info("Data transformation completed")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor)

            return train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path
        
        except Exception as e:
            raise CustomException(e,sys)
