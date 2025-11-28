import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import pandas as pd
import joblib
import os

def show():
    BASE_DIR = os.path.dirname(__file__)
    st.title("Đánh giá mô hình & đường cong ROC")

    @st.cache_resource
    def load_data():
        X = pd.read_csv(os.path.join(BASE_DIR, 'X_test.csv'))
        Y = pd.read_csv(os.path.join(BASE_DIR, 'Y_test.csv'))
        return X, Y

    @st.cache_resource
    def load_model():
        models_7d = {
            "Ada Boost": joblib.load(os.path.join(BASE_DIR, "AdaBoost_mortality_7d.joblib")),
            "Extra Trees": joblib.load(os.path.join(BASE_DIR, "ExtraTrees_mortality_7d.joblib")),
            "Gradient Boosting": joblib.load(os.path.join(BASE_DIR, "GradientBoosting_mortality_7d.joblib")),
            "Random Forest": joblib.load(os.path.join(BASE_DIR, "RandomForest_mortality_7d.joblib")),
        }

        models_28d = {
            "Ada Boost": joblib.load(os.path.join(BASE_DIR, "AdaBoost_mortality_28d.joblib")),
            "Extra Trees": joblib.load(os.path.join(BASE_DIR, "ExtraTrees_mortality_28d.joblib")),
            "Gradient Boosting": joblib.load(os.path.join(BASE_DIR, "GradientBoosting_mortality_28d.joblib")),
            "Random Forest": joblib.load(os.path.join(BASE_DIR, "RandomForest_mortality_28d.joblib")),
        }

        models_90d = {
            "Ada Boost": joblib.load(os.path.join(BASE_DIR, "AdaBoost_mortality_90d.joblib")),
            "Extra Trees": joblib.load(os.path.join(BASE_DIR, "ExtraTrees_mortality_90d.joblib")),
            "Gradient Boosting": joblib.load(os.path.join(BASE_DIR, "GradientBoosting_mortality_90d.joblib")),
            "Random Forest": joblib.load(os.path.join(BASE_DIR, "RandomForest_mortality_90d.joblib")),
        }

        models_1y = {
            "Ada Boost": joblib.load(os.path.join(BASE_DIR, "AdaBoost_mortality_1y.joblib")),
            "Extra Trees": joblib.load(os.path.join(BASE_DIR, "ExtraTrees_mortality_1y.joblib")),
            "Gradient Boosting": joblib.load(os.path.join(BASE_DIR, "GradientBoosting_mortality_1y.joblib")),
            "Random Forest": joblib.load(os.path.join(BASE_DIR, "RandomForest_mortality_1y.joblib")),
        }

        return models_7d, models_28d, models_90d, models_1y 

    X_test, Y_test = load_data()
    models_7d, models_28d, models_90d, models_1y = load_model()

    mortalities = ['mortality_7d', 'mortality_28d', 'mortality_90d', 'mortality_1y']
    titles = ['7 days', '28 days', '90 days', '1 year']

    st.write("So sánh đường cong ROC của các mô hình học máy qua các mốc thời gian tử vong:")

    # ==== Chọn mô hình hiển thị ====
    all_models = list(models_7d.keys())  # giả sử tất cả các model giống nhau ở 4 mốc
    selected_models = st.multiselect(
        "Chọn mô hình hiển thị:",
        options=all_models,
        default=all_models
    )

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    for i, target in enumerate(mortalities):
        y_test = Y_test[target]

        if target == "mortality_7d":
            current_models = models_7d
        elif target == "mortality_28d":
            current_models = models_28d
        elif target == "mortality_90d":
            current_models = models_90d
        else:  # mortality_1y
            current_models = models_1y

        for name, model in current_models.items():
            if name in selected_models:  # chỉ hiển thị model được chọn
                RocCurveDisplay.from_estimator(
                    estimator=model,
                    X=X_test,
                    y=y_test,
                    ax=axes[i],
                    name=name
                )

        axes[i].set_title(titles[i], fontsize=16)
        axes[i].grid(True)
        axes[i].set_xlabel('False Positive Rate (Positive label: 1)')
        axes[i].set_ylabel('True Positive Rate (Positive label: 1)')
        axes[i].legend(loc='lower right')

    st.pyplot(fig)



    # selected_models = st.multiselect(
    #     "Chọn mô hình hiển thị:",
    #     options=list(models.keys()),
    #     default=list(models.keys())
    # )

    # # ================== PLOT ==================
    # fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    # for i, mortality in enumerate(mortalities):
    #     y_test = Y_test[mortality]
    #     for name in selected_models:
    #         RocCurveDisplay.from_estimator(models[name], X_test, y_test, ax=axes[i], name=name)
    #     axes[i].set_title(titles[i])
    #     axes[i].legend(fontsize=8)
    # st.pyplot(fig)

    st.markdown("---")
    # st.caption("ROC curves for multiple mortality endpoints using different classifiers.")
    












# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import RocCurveDisplay
# import joblib
# import os

# def show():
#     BASE_DIR = os.path.dirname(__file__)
#     @st.cache_resource
#     def load_data():
#         X = pd.read_csv(os.path.join(BASE_DIR, 'X_test.csv'))
#         Y = pd.read_csv(os.path.join(BASE_DIR, 'Y_test.csv'))
#         return X, Y
#     @st.cache_resource
#     def load_models():
#         return {
#             'Ada Boost': joblib.load(os.path.join(BASE_DIR, 'AdaBoost.joblib')),
#             'Extra Trees': joblib.load(os.path.join(BASE_DIR, 'ExtraTrees.joblib')),
#             'Gradient Boosting': joblib.load(os.path.join(BASE_DIR, 'GradientBoosting.joblib')),
#             'Random Forest': joblib.load(os.path.join(BASE_DIR, 'RandomForest.joblib'))
#         }
#     X_test, Y_test = load_data()
#     models = load_models()

#     st.title("📊 Model Evaluation")
#     mortalities = ['mortality_7d', 'mortality_28d', 'mortality_90d', 'mortality_1y']
#     titles = ['7 days', '28 days', '90 days', '1 year']
#     fig, axes = plt.subplots(1, 4, figsize=(20, 6))
#     for i, m in enumerate(mortalities):
#         y_test = Y_test[m]
#         for name, model in models.items():
#             RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[i], name=name)
#         axes[i].set_title(titles[i])
#     st.pyplot(fig)

