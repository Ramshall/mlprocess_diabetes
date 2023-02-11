import pandas as pd
import util as utils
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE


def load_dataset(config_data: dict):
    X_train = utils.pickle_load(config_data["train_set_path"][0])
    y_train = utils.pickle_load(config_data["train_set_path"][1])

    X_valid = utils.pickle_load(config_data["valid_set_path"][0])
    y_valid = utils.pickle_load(config_data["valid_set_path"][1])

    X_test = utils.pickle_load(config_data["test_set_path"][0])
    y_test = utils.pickle_load(config_data["test_set_path"][1])

    train_set = pd.concat([X_train, y_train], axis = 1)
    valid_set = pd.concat([X_valid, y_valid], axis = 1)
    test_set = pd.concat([X_test, y_test], axis = 1)
    
    return train_set, valid_set, test_set

def rus_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Create sampling object
    rus = RandomUnderSampler(random_state = 123)

    # Balancing set data
    X_rus, y_rus = rus.fit_resample(set_data.drop(columns = config["label"]),
                                 set_data[config["label"]])

    # Concatenate balanced data
    set_data_rus = pd.concat([X_rus, y_rus], axis = 1)

    # Return balanced data
    return set_data_rus

def ros_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Create sampling object
    ros = RandomOverSampler(random_state = 123)

    # Balancing set data
    X_ros, y_ros = ros.fit_resample(set_data.drop(columns = config["label"]),
                                 set_data[config["label"]])

    # Concatenate balanced data
    set_data_ros = pd.concat([X_ros, y_ros], axis = 1)

    # Return balanced data
    return set_data_ros

def sm_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Create sampling object
    sm = SMOTE(random_state = 123)

    # Balancing set data
    X_sm, y_sm = sm.fit_resample(set_data.drop(columns = config["label"]),
                                 set_data[config["label"]])

    # Concatenate balanced data
    set_data_sm = pd.concat([X_sm, y_sm], axis = 1)

    # Return balanced data
    return set_data_sm

def remove_outliers(set_data):
    set_data = set_data.copy()
    list_of_set_data = list()

    for col_name in set_data.columns[:-1]:
        q1 = set_data[col_name].quantile(0.25)
        q3 = set_data[col_name].quantile(0.75)
        iqr = q3 - q1
        set_data_cleaned = set_data[~((set_data[col_name] < (q1 - 1.5 * iqr)) | (set_data[col_name] > (q3 + 1.5 * iqr)))].copy()
        list_of_set_data.append(set_data_cleaned.copy())
    
    set_data_cleaned = pd.concat(list_of_set_data)
    count_duplicated_index = set_data_cleaned.index.value_counts()
    used_index_data = count_duplicated_index[count_duplicated_index == (set_data.shape[1]-1)].index
    set_data_cleaned = set_data_cleaned.loc[used_index_data].drop_duplicates()

    return set_data_cleaned

if __name__ == "__main__":
    # load configuration file
    config = utils.load_config()
    
    # load dataset
    train_set, valid_set, test_set = load_dataset(config)
    
    # RandomUndersampling dataset
    train_set_rus = rus_fit_resample(train_set)
    
    # RandomOversampling dataset
    train_set_ros = ros_fit_resample(train_set)
    
    # SMOTE dataset
    train_set_sm = sm_fit_resample(train_set)
    
    # remove outlier set rus
    train_set_rus_bal_cleaned = remove_outliers(train_set_rus)
    
    # remove outlier set ros
    train_set_ros_bal_cleaned = remove_outliers(train_set_ros)
    
    # remove outlier set sm
    train_set_sm_bal_cleaned = remove_outliers(train_set_sm)
    
    # Dump set data
    x_train = {
        "Undersampling" : train_set_rus_bal_cleaned.drop(columns = "Outcome"),
        "Oversampling" : train_set_ros_bal_cleaned.drop(columns = "Outcome"),
        "SMOTE" : train_set_sm_bal_cleaned.drop(columns = "Outcome")
        }

    y_train = {
        "Undersampling" : train_set_rus_bal_cleaned.Outcome,
        "Oversampling" : train_set_ros_bal_cleaned.Outcome,
        "SMOTE" : train_set_sm_bal_cleaned.Outcome
        }

    utils.pickle_dump(
        x_train, 
        "data/processed/X_train_feng.pkl"
        )
    
    utils.pickle_dump(
        y_train, 
        "data/processed/y_train_feng.pkl"
        )

    utils.pickle_dump(
        valid_set.drop(columns = "Outcome"), 
        "data/processed/X_valid_feng.pkl"
        )
    
    utils.pickle_dump(
        valid_set.Outcome, 
        "data/processed/y_valid_feng.pkl"
        )

    utils.pickle_dump(
        test_set.drop(columns = "Outcome"), 
        "data/processed/X_test_feng.pkl"
        )

    utils.pickle_dump(
        test_set.Outcome, 
        "data/processed/y_test_feng.pkl"
        )