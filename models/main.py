import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

df = pd.read_csv("data/salary_data.csv")

encoder = LabelEncoder()
df["education"] = encoder.fit_transform(df["education"])

X_single = df[["experience"]]
Y = df["salary"]

X_train_s, X_test_s, Y_train_s, Y_test_s = train_test_split(
    X_single, Y, test_size = 0.2, random_state = 42
)

single_model = LinearRegression()
single_model.fit(X_train_s, Y_train_s)

Y_pred_s = single_model.predict(X_test_s)

rmse_single = mean_squared_error(Y_test_s, Y_pred_s) ** 0.5
r2_single = r2_score(Y_test_s, Y_pred_s)

print("===== Single Feature Model =====")
print("RMSE:", rmse_single)
print("R2 Score:", r2_single)


X_multi = df[["experience", "test_score", "education"]]
X_train_m, X_test_m, Y_train_m, Y_test_m = train_test_split(
    X_multi, Y, test_size = 0.2, random_state = 42
)

multi_model = LinearRegression()
multi_model.fit(X_train_m, Y_train_m)

Y_pred_m = multi_model.predict(X_test_m)

rmse_multi = mean_squared_error(Y_test_m, Y_pred_m) ** 0.5
r2_multi = r2_score(Y_test_m, Y_pred_m)

print("===== Multiple Feature Model =====")
print("RMSE:", rmse_multi)
print("R2 Score:", r2_multi)

if r2_multi > r2_single:
    best_model = multi_model
    print("\nMultiple Feature Model Selected")
else:
    best_model = single_model
    print("\nSingle Feature Model Selected")

joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(encoder, "models/encoder.pkl")

print("Best Model saved successfully")

plt.figure(figsize = (8,5))
plt.scatter(Y_test_m, Y_pred_m)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.show()

print("\nFeature Importance:")

for feature, coef in zip(
    ["experience", "test_score", "education"],
    multi_model.coef_
):
    print(feature, "=", coef)