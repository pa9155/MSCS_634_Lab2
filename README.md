# MSCS_634_Lab2
MSCS_634_Lab2
Loading The DataSet
wine = load_wine()

X = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

y = pd.Series(
    wine.target,
    name="target"
)

print("Dataset Shape:", X.shape)

print("\nFeature Names:")
print(X.columns.tolist())

print("\nClass Distribution:")
print(y.value_counts().sort_index())

2. Training and Testing Data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

k_values = [1, 5, 11, 15, 21]

knn_results = []

for k in k_values:

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    knn_results.append({
        "k": k,
        "Accuracy": accuracy,
        "Accuracy (%)": accuracy * 100
    })

knn_results_df = pd.DataFrame(knn_results)

display(knn_results_df)

KNN Accuracy

plt.figure(figsize=(8,5))

plt.plot(
    knn_results_df["k"],
    knn_results_df["Accuracy (%)"],
    marker="o"
)

plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy (%)")
plt.title("KNN Accuracy for Different k Values")

plt.xticks(k_values)
plt.grid(True)

plt.show()

radius_values = [
    350, 400, 450, 500, 550, 600
]

rnn_results = []

for radius in radius_values:

    model = RadiusNeighborsClassifier(
        radius=radius,
        outlier_label="most_frequent"
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    rnn_results.append({
        "Radius": radius,
        "Accuracy": accuracy,
        "Accuracy (%)": accuracy * 100
    })

rnn_results_df = pd.DataFrame(rnn_results)

display(rnn_results_df)

KNN

best_knn_index = knn_results_df["Accuracy"].idxmax()

best_knn_k = knn_results_df.loc[
    best_knn_index,
    "k"
]

best_knn_accuracy = knn_results_df.loc[
    best_knn_index,
    "Accuracy (%)"
]

print(
    f"Best observed KNN configuration: "
    f"k = {best_knn_k}"
)

print(
    f"Accuracy: {best_knn_accuracy:.2f}%"
)
