from src.components.data_ingestion import data_ingestion,data_ingestion_config
from src.components.data_transformation import data_transformation,DataTransformationConfig
from src.components.model_trainer import model_trainer,ModelTrainerConfig


if __name__=="__main__":
    obj=data_ingestion()
    train_data,test_data=obj.initiate_data_ingestion()

    transformation_obj=data_transformation()
    train_arr,test_arr,_=transformation_obj.initiate_data_trasformation(train_data,test_data)

    trainer_obj=model_trainer()
    print(trainer_obj.initiate_model_trainer(train_arr,test_arr))