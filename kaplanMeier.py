import streamlit as st
import matplotlib.pyplot as plt
import joblib

def show():
    st.title("Phân tích khả năng sống sót (Kaplan–Meier)")

    # Load toàn bộ mô hình
    all_models, all_pvals, groups = joblib.load("kaplanmeier_models.joblib")

    titles = list(all_models.keys())  # 4 thời điểm
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    for i, time_name in enumerate(titles):
        ax = axes[i]
        models = all_models[time_name]
        pval = all_pvals[time_name]

        # Vẽ các đường survival
        for group, kmf in models.items():
            kmf.plot(ax=ax)

        # Tiêu đề và trục
        ax.legend(loc="upper right") 
        ax.set_title(f"{time_name}", fontsize=18)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Survival Probability")

        # Ghi Log-rank lên trên mỗi hình
        p_text = "Log-rank p < 0.001" if pval < 0.001 else f"Log-rank p = {pval:.4f}"
        ax.text(0.05, 0.05, p_text, transform=ax.transAxes, fontsize=14, verticalalignment='bottom')

    plt.tight_layout()
    st.pyplot(fig)




