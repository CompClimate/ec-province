#!/usr/bin/env python3

import os
import sys
import numpy as np
import tensorflow as tf
import shap
import pickle

def load_data():
    X_train_scaled = np.load("updated_X_train_scaled_V2_no_phyto.npy")  # adjust path if needed
    X_to_explain = np.load("updated_X_total_scaled_fy_V2_no_phyto.npy")
    return X_train_scaled, X_to_explain

def main():
    if len(sys.argv) != 2:
        print("Usage: python compute_shap_single_model.py <model_index>")
        sys.exit(1)

    model_idx = int(sys.argv[1])

    #model_name = 'keras_nn_attempt_jul31'
    #model_name = 'keras_nn_attempt_aug6'
    model_name = 'keras_nn_attempt_oct_14_25_v2'
    feature_names = ["rr_3", "rr_7", "chl"]
    model_dir = f'saved_models/{model_name}'
    shap_save_dir = os.path.join(model_dir, 'shap_info')
    os.makedirs(shap_save_dir, exist_ok=True)

    print(f"Loading data for model {model_idx}...")
    background_data, X_to_explain = load_data()

    model_path = os.path.join(model_dir, f'model_{model_idx}.keras')
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        sys.exit(1)

    print(f"Loading model {model_idx} from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    print(f"Creating explainer for model {model_idx}...")
    explainer = shap.Explainer(model, background_data, algorithm='exact', feature_names=feature_names)

    print(f"Calculating SHAP values for model {model_idx}...")
    shap_values = explainer(X_to_explain)

    shap_vals_path = os.path.join(shap_save_dir, f'shap_values_model_{model_idx}.npy')
    print(f"Saving SHAP values to {shap_vals_path}")
    np.save(shap_vals_path, shap_values.values)

    explainer_path = os.path.join(shap_save_dir, f'explainer_model_{model_idx}.pkl')
    print(f"Saving explainer to {explainer_path}")
    with open(explainer_path, 'wb') as f:
        pickle.dump(explainer, f)

    print(f"Done with model {model_idx}!")

if __name__ == "__main__":
    main()