import argparse
import os
from data_load import get_supervised_imu_data_loaders
from data_load import get_tdost_data_loaders
from supervised_imu.utils import *
from supervised_imu.train_and_eval import (
    setup_supervised_imu,
    train_supervised_imu,
    get_supervised_imu_predicted_probabilities,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description='arguments for main application')
    parser.add_argument("--imu_joblib_file", type=str, default="./all_data/MARBLE_IMU.joblib", help="Path to the joblib file to use for the supervised IMU model")
    parser.add_argument("--embeddings_dir", type=str, default="./all_data/", help="Directory where the embeddings for TDOST are located.")
    parser.add_argument("--sentence_encoder", type=str, default="all-MiniLM-L12-v2", help="Name of the sentence encoder to be used for TDOST.")
    parser.add_argument("--batch_size", type=int, default=250, help="Batch size for model training.")
    parser.add_argument("--model_save_path", type=str, default="./models/", help="Directory where the models should be saved.")
    parser.add_argument('--train', action='store_true', help='Train both supervised IMU and TDOST models')
    parser.add_argument('--evaluate', type=str, default="supervised_imu", help="evaluate supervised_imu, tdost, or fusion") # change default to fusion
    args = parser.parse_args()

    return args


def app(args):
    classifier, net, device, optimizer, exp_config = setup_supervised_imu(args)

    if args.train:
        imu_trainloader, imu_valloader, imu_testloader = get_supervised_imu_data_loaders(args)
        train_supervised_imu(imu_trainloader, imu_valloader, net, device, optimizer, exp_config)

        # add tdost training here
    
    if args.evaluate == "supervised_imu":
        _, _, imu_testloader = get_supervised_imu_data_loaders(args)
        imu_pred_probs, imu_gt_labels = get_supervised_imu_predicted_probabilities(net, device, args.model_save_path, imu_testloader)
        evaluate_predictions(np.array(imu_gt_labels), np.array(imu_pred_probs))
    elif args.evaluate == "tdost":
        raise NotImplementedError("Function get_tdost_predicted_probabilities needs to be implemented.")
        # tdost_pred_probs, tdost_gt_labels = get_tdsot_predicted_probabilities(...)
        # evaluate_predictions(np.array(tdost_gt_labels), np.array(tdost_pred_probs))
    elif args.evaluate == "fused":
        raise NotImplementedError("Function get_tdost_predicted_probabilities needs to be implemented.")
        # imu_pred_probs, imu_gt_labels = get_supervised_imu_predicted_probabilities(net, device, model_save_path, imu_testloader)
        # tdost_pred_probs, tdost_gt_labels = get_tdsot_predicted_probabilities(...)
        # fused_probs, fused_gt_labels =  late_fusion(imu_gt_labels, tdost_gt_labels, imu_pred_probs, tdost_pred_probs)
        # evaluate_predictions(fused_gt_labels, fused_probs)


def late_fusion(m1_gt, m2_gt, m1_pred_probs, m2_pred_probs):
    """
    m1_gt: 1D numpy array of model 1's ground truth labels
    m2_gt: 1D numpy array of model 2's ground truth labels
    m1_pred_probs: 2D numpy array of predicted probabilities for each class by model 1
    m2_pred_probs: 2D numpy array of predicted probabilities for each class by model 2
    """

    # sizes of predictions and probabilities for both models should match
    assert np.array_equal(m1_gt, m2_gt), "Arrays are not equal!" # we have serious issues if this ain't equal
    assert m1_pred_probs.shape == m2_pred_probs.shape
    assert m1_gt.shape[0] == m2_pred_probs.shape[0]

    fused_gt_labels = m1_gt.copy()
    fused_probs = np.mean(np.stack([m1_pred_probs, m2_pred_probs]), axis=0)

    return fused_probs, fused_gt_labels


def evaluate_predictions(y_true, y_pred_probs):
    """
    y_true: List or 1D numpy array of ground truth labels
    y_pred_probs: 2D numpy array of predicted probabilities for each class
    """
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    print("\n📊 Evaluation Metrics")
    print("---------------------")
    print(f"Accuracy         : {acc:.4f}")
    print(f"F1 Score (Macro) : {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"Precision        : {precision:.4f}")
    print(f"Recall           : {recall:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred))
    
    print("🧾 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }


if __name__ == '__main__':
    args = parse_arguments()
    print(args)

    if not os.path.exists(args.imu_joblib_file):
        raise FileNotFoundError(f"The MARBLE joblib file '{args.imu_joblib_file}' does not exist. Make sure you have done the preprocessing and processing steps first.")
    if not os.path.exists(args.embeddings_dir):
        raise FileNotFoundError(f"The directory '{args.embeddings_dir}' does not exist.")
    
    app(args)
