# =============================================================
# Customer Churn Prediction — Churn_Modelling.csv
# Dataset: Bank Customer Churn (10,000 records)
# Target : Exited (1 = Churned, 0 = Retained)
# Models : Logistic Regression, Random Forest, Gradient Boosting
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)

# ==============================================================
# STEP 1: LOAD DATASET
# ==============================================================
print("=" * 65)
print("STEP 1: Load Dataset — Churn_Modelling.csv")
print("=" * 65)

df = pd.read_csv("Churn_Modelling.csv")

print(f"Dataset shape      : {df.shape}")
print(f"Columns            : {df.columns.tolist()}")
print(f"Churned customers  : {df['Exited'].sum()} ({df['Exited'].mean()*100:.2f}%)")
print(f"Retained customers : {(df['Exited']==0).sum()} ({(1-df['Exited'].mean())*100:.2f}%)")
print(f"Missing values     : {df.isnull().sum().sum()}")
print(f"\nFirst 5 rows:\n{df.head()}")


# ==============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================
print("\n" + "=" * 65)
print("STEP 2: Exploratory Data Analysis")
print("=" * 65)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Bank Customer Churn — EDA Overview", fontsize=16, fontweight='bold')

# 1. Churn Distribution
churn_counts = df['Exited'].value_counts()
axes[0,0].pie(churn_counts, labels=['Retained','Churned'],
              autopct='%1.1f%%', colors=['#2196F3','#F44336'],
              startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
axes[0,0].set_title('Churn Distribution')

# 2. Churn by Geography
churn_geo = df.groupby('Geography')['Exited'].mean() * 100
axes[0,1].bar(churn_geo.index, churn_geo.values,
              color=['#42A5F5','#AB47BC','#FFA726'])
axes[0,1].set_title('Churn Rate by Geography')
axes[0,1].set_ylabel('Churn Rate (%)')
for i, v in enumerate(churn_geo.values):
    axes[0,1].text(i, v+0.3, f'{v:.1f}%', ha='center', fontweight='bold')

# 3. Churn by Gender
churn_gen = df.groupby('Gender')['Exited'].mean() * 100
axes[0,2].bar(churn_gen.index, churn_gen.values,
              color=['#EF5350','#42A5F5'])
axes[0,2].set_title('Churn Rate by Gender')
axes[0,2].set_ylabel('Churn Rate (%)')
for i, v in enumerate(churn_gen.values):
    axes[0,2].text(i, v+0.3, f'{v:.1f}%', ha='center', fontweight='bold')

# 4. Age Distribution by Churn
axes[1,0].hist([df[df['Exited']==0]['Age'],
                df[df['Exited']==1]['Age']],
               bins=25, label=['Retained','Churned'],
               color=['#2196F3','#F44336'], alpha=0.7)
axes[1,0].set_title('Age Distribution by Churn')
axes[1,0].set_xlabel('Age')
axes[1,0].legend()

# 5. Balance Distribution by Churn
axes[1,1].boxplot([df[df['Exited']==0]['Balance'],
                   df[df['Exited']==1]['Balance']],
                  labels=['Retained','Churned'],
                  patch_artist=True,
                  boxprops=dict(facecolor='#E3F2FD'))
axes[1,1].set_title('Account Balance vs Churn')
axes[1,1].set_ylabel('Balance ($)')

# 6. Churn by NumOfProducts
churn_prod = df.groupby('NumOfProducts')['Exited'].mean() * 100
axes[1,2].bar(churn_prod.index, churn_prod.values, color='#7E57C2')
axes[1,2].set_title('Churn Rate by Number of Products')
axes[1,2].set_xlabel('Number of Products')
axes[1,2].set_ylabel('Churn Rate (%)')
for i, (idx, v) in enumerate(churn_prod.items()):
    axes[1,2].text(i, v+0.3, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=150)
plt.show()
print("💾 Saved: eda_overview.png")

# Correlation Heatmap
fig, ax = plt.subplots(figsize=(11, 7))
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()
print("💾 Saved: correlation_heatmap.png")

# Additional EDA: Active Member vs Churn
print("\nChurn rate by IsActiveMember:")
print(df.groupby('IsActiveMember')['Exited'].mean() * 100)
print("\nChurn rate by HasCrCard:")
print(df.groupby('HasCrCard')['Exited'].mean() * 100)
print("\nChurn rate by Geography:")
print(df.groupby('Geography')['Exited'].mean() * 100)


# ==============================================================
# STEP 3: DATA PREPROCESSING
# ==============================================================
print("\n" + "=" * 65)
print("STEP 3: Data Preprocessing")
print("=" * 65)

df_model = df.copy()

# 3a. Drop irrelevant columns
drop_cols = ['RowNumber', 'CustomerId', 'Surname']
df_model.drop(columns=drop_cols, inplace=True)
print(f"Dropped columns  : {drop_cols}")

# 3b. Encode categorical columns
le = LabelEncoder()
cat_cols = ['Geography', 'Gender']
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col])
    print(f"Encoded '{col}'  : {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 3c. Feature / Target split
X = df_model.drop(columns=['Exited'])
y = df_model['Exited']

print(f"\nFeatures used    : {X.columns.tolist()}")
print(f"Feature matrix   : {X.shape}")
print(f"Target shape     : {y.shape}")
print(f"Churn rate       : {y.mean()*100:.2f}%")

# 3d. Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print("Scaling applied  : StandardScaler on all features")


# ==============================================================
# STEP 4: TRAIN-TEST SPLIT (80/20)
# ==============================================================
print("\n" + "=" * 65)
print("STEP 4: Train-Test Split (80 / 20)")
print("=" * 65)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")
print(f"Churn in train   : {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"Churn in test    : {y_test.sum()}  ({y_test.mean()*100:.2f}%)")


# ==============================================================
# STEP 5: MODEL TRAINING
# ==============================================================
print("\n" + "=" * 65)
print("STEP 5: Model Training")
print("=" * 65)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced'
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42,
        class_weight='balanced', n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        random_state=42
    ),
}

trained = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n  ⏳ Training {name}...")
    model.fit(X_train, y_train)
    trained[name] = model
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"  ✅ CV ROC-AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ==============================================================
# STEP 6: MODEL EVALUATION
# ==============================================================
print("\n" + "=" * 65)
print("STEP 6: Model Evaluation")
print("=" * 65)

results = {}
for name, model in trained.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results[name] = {
        "model"    : model,
        "y_pred"   : y_pred,
        "y_prob"   : y_prob,
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall"   : recall_score(y_test, y_pred, zero_division=0),
        "f1"       : f1_score(y_test, y_pred, zero_division=0),
        "roc_auc"  : roc_auc_score(y_test, y_prob),
        "cm"       : confusion_matrix(y_test, y_pred),
    }

# Comparison table
print(f"\n{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7} {'ROC-AUC':>9}")
print("-" * 70)
for name, r in results.items():
    print(f"{name:<22} {r['accuracy']:>9.4f} {r['precision']:>10.4f}"
          f" {r['recall']:>8.4f} {r['f1']:>7.4f} {r['roc_auc']:>9.4f}")

# Detailed reports
for name, r in results.items():
    print(f"\n--- {name} ---")
    print(classification_report(y_test, r['y_pred'],
                                target_names=['Retained','Churned'],
                                zero_division=0))

# Plot 1: Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight='bold')
for ax, (name, r) in zip(axes, results.items()):
    sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Retained','Churned'],
                yticklabels=['Retained','Churned'])
    ax.set_title(f"{name}\nF1: {r['f1']:.4f}  AUC: {r['roc_auc']:.4f}")
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()
print("💾 Saved: confusion_matrices.png")

# Plot 2: ROC Curves
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#1565C0', '#E65100', '#2E7D32']
for (name, r), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"{name} (AUC = {r['roc_auc']:.4f})")
ax.plot([0,1], [0,1], 'k--', linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.show()
print("💾 Saved: roc_curves.png")

# Plot 3: Metric Comparison
metric_keys   = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
x      = np.arange(len(metric_labels))
width  = 0.25
colors = ['#1565C0', '#E65100', '#2E7D32']

fig, ax = plt.subplots(figsize=(13, 5))
for i, (name, color) in enumerate(zip(results.keys(), colors)):
    vals = [results[name][m] for m in metric_keys]
    bars = ax.bar(x + i*width, vals, width, label=name,
                  color=color, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=7.5, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()
print("💾 Saved: model_comparison.png")


# ==============================================================
# STEP 7: FEATURE IMPORTANCE
# ==============================================================
print("\n" + "=" * 65)
print("STEP 7: Feature Importance")
print("=" * 65)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Feature Importance", fontsize=14, fontweight='bold')

rf_imp = pd.Series(trained["Random Forest"].feature_importances_,
                   index=X.columns).sort_values(ascending=True)
rf_imp.plot(kind='barh', ax=axes[0], color='#1565C0')
axes[0].set_title('Random Forest — Feature Importance')
axes[0].set_xlabel('Importance Score')

gb_imp = pd.Series(trained["Gradient Boosting"].feature_importances_,
                   index=X.columns).sort_values(ascending=True)
gb_imp.plot(kind='barh', ax=axes[1], color='#2E7D32')
axes[1].set_title('Gradient Boosting — Feature Importance')
axes[1].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("💾 Saved: feature_importance.png")

print("\nTop 5 Features Driving Churn (Random Forest):")
for feat, score in rf_imp.sort_values(ascending=False).head(5).items():
    print(f"  {feat:<22} {score:.4f}")


# ==============================================================
# STEP 8: BEST MODEL SELECTION
# ==============================================================
print("\n" + "=" * 65)
print("STEP 8: Best Performing Model (by ROC-AUC)")
print("=" * 65)

best_name  = max(results, key=lambda n: results[n]['roc_auc'])
best_model = results[best_name]['model']

ranked = sorted(results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
medals = ['🥇','🥈','🥉']
for rank, (name, r) in enumerate(ranked, 1):
    print(f"  {medals[rank-1]} #{rank}  {name:<22}  "
          f"Acc={r['accuracy']:.4f}  F1={r['f1']:.4f}  AUC={r['roc_auc']:.4f}")

print(f"\n✅ Best Model Selected: {best_name}")


# ==============================================================
# STEP 9: CHURN PREDICTION ON NEW CUSTOMERS
# ==============================================================
print("\n" + "=" * 65)
print("STEP 9: Churn Prediction (Final Output)")
print("=" * 65)

# Encoding maps (matching LabelEncoder alphabetical order)
GEO_MAP    = {'France': 0, 'Germany': 1, 'Spain': 2}
GENDER_MAP = {'Female': 0, 'Male': 1}

def predict_churn(model, scaler, customer: dict) -> dict:
    """
    Predict churn for a single new customer.

    Args:
        model    : Trained sklearn classifier
        scaler   : Fitted StandardScaler
        customer : Dict with customer feature values

    Returns:
        Dict with prediction, churn probability, and risk level
    """
    row = pd.DataFrame([{
        'CreditScore'     : customer['CreditScore'],
        'Geography'       : GEO_MAP[customer['Geography']],
        'Gender'          : GENDER_MAP[customer['Gender']],
        'Age'             : customer['Age'],
        'Tenure'          : customer['Tenure'],
        'Balance'         : customer['Balance'],
        'NumOfProducts'   : customer['NumOfProducts'],
        'HasCrCard'       : customer['HasCrCard'],
        'IsActiveMember'  : customer['IsActiveMember'],
        'EstimatedSalary' : customer['EstimatedSalary'],
    }])

    row_scaled  = scaler.transform(row)
    prediction  = model.predict(row_scaled)[0]
    probability = model.predict_proba(row_scaled)[0][1]
    risk        = ("🔴 HIGH RISK"   if probability > 0.7 else
                   "🟡 MEDIUM RISK" if probability > 0.4 else
                   "🟢 LOW RISK")
    return {
        "prediction" : "🚨 WILL CHURN" if prediction == 1 else "✅ WILL STAY",
        "churn_prob" : f"{probability*100:.1f}%",
        "risk_level" : risk,
    }


new_customers = [
    # High risk: old, German, inactive, many products (2+), zero balance
    {"CreditScore":450,"Geography":"Germany","Gender":"Female","Age":55,
     "Tenure":2,"Balance":0.0,"NumOfProducts":4,"HasCrCard":1,
     "IsActiveMember":0,"EstimatedSalary":50000},

    # Low risk: young, French, active member, good credit
    {"CreditScore":750,"Geography":"France","Gender":"Male","Age":28,
     "Tenure":7,"Balance":85000,"NumOfProducts":2,"HasCrCard":1,
     "IsActiveMember":1,"EstimatedSalary":120000},

    # Medium risk: middle-aged, Spain, moderate activity
    {"CreditScore":600,"Geography":"Spain","Gender":"Female","Age":40,
     "Tenure":4,"Balance":45000,"NumOfProducts":1,"HasCrCard":0,
     "IsActiveMember":0,"EstimatedSalary":75000},
]

print(f"\nUsing best model: 🏆 {best_name}\n")
print("-" * 65)
for i, customer in enumerate(new_customers, 1):
    result = predict_churn(best_model, scaler, customer)
    print(f"Customer #{i}")
    print(f"  Profile     : Age={customer['Age']}, "
          f"Geography={customer['Geography']}, "
          f"Gender={customer['Gender']}")
    print(f"  Credit Score: {customer['CreditScore']}  |  "
          f"Balance: ${customer['Balance']:,.0f}  |  "
          f"Active: {'Yes' if customer['IsActiveMember'] else 'No'}")
    print(f"  Prediction  : {result['prediction']}")
    print(f"  Churn Prob  : {result['churn_prob']}")
    print(f"  Risk Level  : {result['risk_level']}")
    print("-" * 65)

print("\n✅ Pipeline complete!")
print(f"   '{best_name}' ready for production deployment.")
print("=" * 65)
