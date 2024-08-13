## Test on a dataset

Split the features and target from a dataset:
```
# Split features and targets
test_image_paths = 'data/test_images/' + test_data.iloc[:, 0].astype(str) + '.jpeg'

test_ancillary_data = test_data.iloc[:, 1:164]
test_labels = test_data.iloc[:, 164:]

test_image_paths = test_image_paths.to_numpy()

```

Assuming that the previous cells in the notebook has been run, you can generate predictions for a dataset by:

```
# Generate predictions for the test set
test_dataset = ImageDataset(test_image_paths)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

test_image_features = extract_features_dataloader(test_dataloader)  # Define test_image_paths

X_test_combined = np.concatenate([test_ancillary_data, test_image_features], axis=1)

test_ids = test_data['id'].values  # Extract test IDs from the test dataset

submission = pd.DataFrame({'id': test_ids})  # Define test_ids
for trait in models:
    submission[trait] = models[trait].predict(X_test_combined)

submission.to_csv('submission.csv', index=False)
```
