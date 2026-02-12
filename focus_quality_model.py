import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


df=pd.read_csv("background_noise_focus_dataset.csv")
df_encoded=pd.get_dummies(df,columns=["role","task_type","background_noise_type"],dtype=int,drop_first=True)


x=df_encoded.drop(columns=["task_completion_quality","participant_id"])
y=df_encoded["task_completion_quality"]

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)


model=XGBRegressor()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)



user_input_dict = {
    "age": int(input("Enter Age: ")),
    "role": input("Enter Role: ").strip().title(),
    "task_type": input("Enter Task Type: ").strip().title(),
    "background_noise_type": input("Enter Background Noise Type: ").strip().title(),
    "noise_volume_level": int(input("Enter Noise Volume Level (1-10): ")),
    "focus_duration_minutes": int(input("Enter Focus Duration (minutes): ")),
    "perceived_focus_score": int(input("Enter Perceived Focus Score (1-10): ")),
    "mental_fatigue_after_task": int(input("Enter Mental Fatigue (1-10): "))
}


user_input_dataframe=pd.DataFrame([user_input_dict])
user_input_encoded=pd.get_dummies(user_input_dataframe)
final_user_input=user_input_encoded.reindex(columns=x.columns,fill_value=0)
task_quality=model.predict(final_user_input)


print("Predicted Quality: ",task_quality)
print("Mean Absolute Error: ",mae)


feat_importances = pd.Series(model.feature_importances_, index=x.columns)
plt.figure(figsize=(10, 6))
feat_importances.nlargest(10).plot(kind='barh', color='teal')
plt.title("Key Factors Affecting Task Quality")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()