import pandas as pd
import util as utils
import copy
from sklearn.model_selection import train_test_split

def read_raw_data(config: dict) -> pd.DataFrame:
    # Return raw dataset
    return pd.read_csv(config["dataset_path"])

def check_data(input_data: pd.DataFrame, config: dict, api: bool = False):
    input_data = copy.deepcopy(input_data)
    config = copy.deepcopy(config)

    if not api:
        # check column data types
        assert input_data.select_dtypes("int").columns.to_list() == \
            config["int_columns"], "an error occurs in int column(s)."
        assert input_data.select_dtypes("float").column.to_list() == \
            config ["float_columns"], "an error occurs in float column(s)."
    
    
    
    else:
        # In case checking data from api
        int_columns = config["int_columns"]
        del int_columns[-1]

        # all column float is used to predict
        float_columns = config["float_columns"]
        
        # Check column data types
        assert input_data.select_dtypes("int64").columns.to_list() == \
            int_columns, "an error occurs in int column(s)."
        assert input_data.select_dtypes("float64").columns.to_list() == \
            float_columns, "an error occurs in float column(s)."
        
        
    # check range of BMI
    assert input_data[config["float_columns"][0]].between(
        config["range_BMI"][0],
        config["range_BMI"][1]
        ).sum() == len(input_data), "an error occurs in range BMI."
        
    # check range DiabetesPedigreeFunction
    assert input_data[config["float_columns"][1]].between(
        config["range_DiabetesPedigreeFunction"][0],
        config["range_DiabetesPedigreeFunction"][1]
        ).sum() == len(input_data), "an error occurs in range DiabetesPedigreeFunction."
        
    # check range Pregnancies
    assert input_data[config["int_columns"][0]].between(
        config["range_Pregnancies"][0],
        config["range_Pregnancies"][1]
        ).sum() == len(input_data), "an error occurs in range Pregnancies."
        
    # check range Glucose
    assert input_data[config["int_columns"][1]].between(
        config["range_Glucose"][0],
        config["range_Glucose"][1]
        ).sum() == len(input_data), "an error occurs in range Glucose."
        
    # check range BloodPressure
    assert input_data[config["int_columns"][2]].between(
        config["range_BloodPressure"][0],
        config["range_BloodPressure"][1]
        ).sum() == len(input_data), "an error occurs in range BloodPressure."
                
    # check range SkinThickness
    assert input_data[config["int_columns"][3]].between(
        config["range_SkinThickness"][0],
        config["range_SkinThickness"][1]
        ).sum() == len(input_data), "an error occurs in range SkinThickness."
        
    # check range Insulin
    assert input_data[config["int_columns"][4]].between(
        config["range_Insulin"][0],
        config["range_Insulin"][1]
        ).sum() == len(input_data), "an error occurs in range SkinThickness."
        
    # check range Age
    assert input_data[config["int_columns"][5]].between(
        config["range_Age"][0],
        config["range_Age"][1]
        ).sum() == len(input_data), "an error occurs in range Age."

def split_data(input_data: pd.DataFrame, config: dict):
    # Split predictor and label
    x = input_data[config["predictors"]].copy()
    y = input_data[config["label"]].copy()

    # 1st split train and test
    X_train, X_test, \
    y_train, y_test = train_test_split(
        x, y,
        test_size = config["test_size"],
        random_state = 123,
        stratify = y
    )

    # 2nd split test and valid
    X_valid, X_test, \
    y_valid, y_test = train_test_split(
        X_test, y_test,
        test_size = config["valid_size"],
        random_state = 123,
        stratify = y_test
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test


if __name__ == "__main__":
    # 1. Load configuration file
    config = utils.load_config()

    # 2. Read all raw dataset
    raw_dataset = read_raw_data(config)

    # 3. Data defense for non API data
    check_data(raw_dataset, config)

    # 4. Splitting train, valid, and test set
    X_train, X_valid, X_test, \
        y_train, y_valid, y_test = split_data(raw_dataset, config)

    # 5. Save train, valid and test set
    utils.pickle_dump(X_train, config["train_set_path"][0])
    utils.pickle_dump(y_train, config["train_set_path"][1])

    utils.pickle_dump(X_valid, config["valid_set_path"][0])
    utils.pickle_dump(y_valid, config["valid_set_path"][1])

    utils.pickle_dump(X_test, config["test_set_path"][0])
    utils.pickle_dump(y_test, config["test_set_path"][1])

    utils.pickle_dump(raw_dataset, config["dataset_cleaned_path"])         