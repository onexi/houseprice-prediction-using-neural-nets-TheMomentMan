# Load the test.csv file (which lacks SalePrice but includes an 'Id' column)
test_df = pd.read_csv('test.csv')
ids = test_df['Id']

# One-hot encode the test data using the same categorical columns as before
test_encoded = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

# Reindex to match the training features
test_encoded = test_encoded.reindex(columns=train_columns, fill_value=0)
test_encoded = test_encoded[top_features]  # Select the same top features
test_encoded = test_encoded.astype(float)
X_test_new = scaler_X.transform(test_encoded)

X_test_tensor_new = torch.tensor(X_test_new, dtype=torch.float32)
final_deep_model.eval()
with torch.no_grad():
    test_predictions = final_deep_model(X_test_tensor_new.to(device))
    test_predictions_np = test_predictions.cpu().numpy()

# Inverse transform predictions to get SalePrice in the original scale.
test_predictions_unscaled = scaler_y.inverse_transform(test_predictions_np)

submission_df = pd.DataFrame({
    'ID': ids.astype(int),
    'SALEPRICE': test_predictions_unscaled.flatten().astype(float)
})


print("\nSubmission Preview:")
print(submission_df.head())

# Optionally, export to CSV:
submission_df.to_csv('predictions2.csv', index=False)