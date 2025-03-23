import os
import sys

import numpy as np # type: ignore
import pandas as pd # type: ignore
from src.exception import CustomException
import dill # type: ignore


def save_object(obj, file_path):
    '''
    This function is used to save the object in the file path
    '''
    try:
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys)