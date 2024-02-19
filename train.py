import mlflow
import mlflow.keras
from datetime import datetime

experiment_name = "uterine_myomas"
model_name = "myoma_classifier"
model_version = 1
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

# Specify tracking server 
mlflow.set_tracking_uri("http://localhost:7777")

# Set the experiment name and create an MLflow run
mlflow.set_experiment(experiment_name)
with mlflow.start_run(run_name = run_name) as mlflow_run:
    
    mlflow.set_experiment_tag("base_model", "VGG16")
    mlflow.set_tag("optimizer", "keras.optimizers.Adam")
    mlflow.set_tag("loss", "categorical_crossentropy")

    mlflow.keras.log_model(model, "model")

    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("input_shape", input_shape)

    mlflow.log_metric("train_loss", history.history["loss"][-1])
    mlflow.log_metric("train_acc", history.history["accuracy"][-1])
    mlflow.log_metric("val_loss", history.history["val_loss"][-1])
    mlflow.log_metric("val_acc", history.history["val_accuracy"][-1])

    mlflow.log_artifact("accuracy.png", "training_accuracy_curves")
    mlflow.log_artifact("loss.png", "training_loss_curves")
    mlflow.log_artifact("confusion_matrix.png", "confusion_matrix")

    mlflow_run_id = mlflow_run.info.run_id
    print("MLFlow Run ID: ", mlflow_run_id)

# Logged model in MLFlow
logged_model_path = f"runs:/{mlflow_run_id}/model"

# Model registration
with mlflow.start_run(run_id=mlflow_run_id) as run:
    mlflow.register_model(
        logged_model_path,
        model_name
    )

# Transition model to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage="Production"
)
