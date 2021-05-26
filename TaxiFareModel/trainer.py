# imports
import mlflow
import joblib
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from TaxiFareModel.encoders import DistanceTransformer
from TaxiFareModel.encoders import TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data
from TaxiFareModel.data import clean_data


class Trainer():
    
    MLFLOW_URI = "https://mlflow.lewagon.co/"
    EXPERIMENT_NAME = "[UK] [London] [hlars] taxifare v01"
    
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self, estimator):
        """set the pipeline"""
        
        print(f"\n--- Pipeline with {estimator.__class__.__name__} ---")
        
        # create distance pipeline
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        
        # create time pipeline
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # create preprocessing pipeline
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('estimator', estimator)
        ])
        
        self.pipeline = pipe
        return None

    def cv(self):
        """cross-validate the pipeline"""
        
        # CV always maximises the scoring function
        # Thus, the score needs to be negated
        my_scorer = make_scorer(compute_rmse, greater_is_better=False)
        cv_rmse = abs(cross_val_score(self.pipeline, self.X, self.y, scoring=my_scorer, cv=5).mean())
        print("cv rmse:", cv_rmse)
        return cv_rmse
    
    def fit(self):
        """fit the pipeline"""
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        
        y_pred = self.pipeline.predict(X_test)
        test_rmse = compute_rmse(y_pred, y_test)
        print("test rmse:", test_rmse)
        return test_rmse
    
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.EXPERIMENT_NAME)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.EXPERIMENT_NAME).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
    
    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, "model.joblib")


if __name__ == "__main__":
    
    # --- PREPARE THE DATA ---
    # get the data
    df = get_data()
    df = clean_data(df)
    X = df.drop("fare_amount", axis=1)
    y = df["fare_amount"]
    
    # do a hold-out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    
    # --- TRAIN DIFFERENT MODELS ---
    dict_models = {
        "LinReg": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso()
    }
    
    for model in dict_models.values():
        
        # create the trainer
        trainer = Trainer(X_train, y_train)
        
        # include the model into the pipeline
        trainer.set_pipeline(model)
    
        # cross-validate the pipeline
        cv_rmse = trainer.cv()
        
        # fit the pipeline
        trainer.fit()
        
        # evaluate the pipeline
        test_rmse = trainer.evaluate(X_val, y_val)
    
        # store params
        student_name = "hlars"
        model_name = model.__class__.__name__
    
        # log the model
        # trainer.mlflow_log_param("student", student_name)
        # trainer.mlflow_log_param("model", model_name)
        # trainer.mlflow_log_metric("cv_rmse", cv_rmse)
        # trainer.mlflow_log_metric("test_rmse", test_rmse)
    
        # save the model
        trainer.save_model()
