import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold, KFold
from sklearn.metrics import (classification_report, confusion_matrix, recall_score, 
                            mean_absolute_error, mean_squared_error, r2_score,
                            roc_auc_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed_data.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """Vẽ ma trận nhầm lẫn với màu sắc"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Bình thường', 'Trễ', 'Cực trễ'],
                yticklabels=['Bình thường', 'Trễ', 'Cực trễ'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Thực tế')
    plt.xlabel('Dự đoán')
    plt.tight_layout()
    return plt

def plot_roc_curves_multiclass(y_true, y_pred_proba, n_classes=3):
    """Vẽ ROC curves cho bài toán đa lớp"""
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'orange', 'red']
    class_names = ['Bình thường', 'Trễ', 'Cực trễ']
    
    for i, color, name in zip(range(n_classes), colors, class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Phân loại đa lớp', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

def plot_residuals(y_true, y_pred, title='Residual Plot'):
    """Vẽ biểu đồ phần dư (residuals) cho hồi quy"""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot of residuals
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title(f'{title} - Residuals vs Predicted', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[1].hist(residuals, bins=50, color='skyblue', edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'{title} - Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def evaluate_system():
    print("Bắt đầu đánh giá hệ thống...")
    
    # Tải dữ liệu
    df = pd.read_csv(DATA_PATH)
    
    # --- Giai đoạn 1: Đánh giá bộ phân loại (CatBoost) ---
    print("\n" + "="*70)
    print("   GIAI ĐOẠN 1: ĐÁNH GIÁ BỘ PHÂN LOẠI (CatBoost Classifier)")
    print("="*70)
    
    # Chuẩn bị dữ liệu cho bộ phân loại
    drop_cols_cls = ['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 
                     'order_approved_at', 'order_delivered_timestamp', 'order_estimated_delivery_date', 
                     'delivery_days', 'delivery_days_capped', 'is_extreme', 'delay_vs_estimated', 'delivery_class']
    drop_cols_cls = [c for c in drop_cols_cls if c in df.columns]
    
    X_cls = df.drop(columns=drop_cols_cls)
    y_cls = df['delivery_class']
    
    # Đảm bảo kiểu chuỗi cho CatBoost
    cat_features = ['product_category_name', 'customer_city', 'customer_state', 'seller_id', 'payment_type']
    for col in cat_features:
        if col in X_cls.columns:
            X_cls[col] = X_cls[col].astype(str)
            
    # Chia dữ liệu (Cùng random_state)
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    
    # Tải bộ phân loại
    try:
        clf = load_model(os.path.join(MODELS_DIR, 'catboost_classifier.pkl'))
        
        # Dự đoán
        y_pred_cls = clf.predict(X_test_cls)
        y_pred_proba_cls = clf.predict_proba(X_test_cls)
        
        print("\n--- Kết quả trên tập Test ---")
        print("Ma trận nhầm lẫn:")
        print(confusion_matrix(y_test_cls, y_pred_cls))
        
        print("\nBáo cáo phân loại chi tiết:")
        print(classification_report(y_test_cls, y_pred_cls, 
                                   target_names=['Bình thường', 'Trễ', 'Cực trễ']))
        print(f"Recall trung bình (Macro): {recall_score(y_test_cls, y_pred_cls, average='macro'):.4f}")
        
        # Tính ROC-AUC cho từng lớp
        print("\n--- ROC-AUC Score ---")
        try:
            roc_auc_ovr = roc_auc_score(y_test_cls, y_pred_proba_cls, 
                                        multi_class='ovr', average='weighted')
            print(f"ROC-AUC (One-vs-Rest, Weighted): {roc_auc_ovr:.4f}")
            
            # ROC-AUC cho từng lớp
            y_test_bin = label_binarize(y_test_cls, classes=[0, 1, 2])
            for i, class_name in enumerate(['Bình thường', 'Trễ', 'Cực trễ']):
                roc_auc_class = roc_auc_score(y_test_bin[:, i], y_pred_proba_cls[:, i])
                print(f"  - {class_name}: {roc_auc_class:.4f}")
        except Exception as e:
            print(f"Không thể tính ROC-AUC: {e}")
        
        # ===== CROSS-VALIDATION =====
        print("\n--- Cross-Validation (5-Fold Stratified) ---")
        print("Đang thực hiện cross-validation...")
        
        # Tạo mô hình mới cho CV (không có early_stopping để tránh conflict)
        from catboost import CatBoostClassifier
        
        # Tính class weights
        class_counts = y_cls.value_counts().sort_index()
        total_samples = len(y_cls)
        n_classes = len(class_counts)
        class_weights = {
            0: 1.0,
            1: (total_samples / (n_classes * class_counts[1])) * 1.5,
            2: (total_samples / (n_classes * class_counts[2])) * 2.0
        }
        
        clf_cv = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=10,
            l2_leaf_reg=5,
            loss_function='MultiClass',
            cat_features=[col for col in cat_features if col in X_cls.columns],
            class_weights=list(class_weights.values()),
            verbose=0,
            random_seed=42,
            task_type="CPU"
        )
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Manual cross-validation
        cv_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        
        print("Đang chạy 5 folds...")
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_cls, y_cls), 1):
            X_train_fold = X_cls.iloc[train_idx]
            X_val_fold = X_cls.iloc[val_idx]
            y_train_fold = y_cls.iloc[train_idx]
            y_val_fold = y_cls.iloc[val_idx]
            
            clf_cv.fit(X_train_fold, y_train_fold, verbose=False)
            y_pred_fold = clf_cv.predict(X_val_fold)
            
            from sklearn.metrics import accuracy_score, precision_score, f1_score
            cv_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred_fold))
            cv_scores['precision'].append(precision_score(y_val_fold, y_pred_fold, average='macro', zero_division=0))
            cv_scores['recall'].append(recall_score(y_val_fold, y_pred_fold, average='macro'))
            cv_scores['f1'].append(f1_score(y_val_fold, y_pred_fold, average='macro'))
            
            print(f"  Fold {fold}: Acc={cv_scores['accuracy'][-1]:.4f}, F1={cv_scores['f1'][-1]:.4f}")
        
        print(f"\nKết quả Cross-Validation (5-Fold):")
        print(f"Accuracy:  {np.mean(cv_scores['accuracy']):.4f} (+/- {np.std(cv_scores['accuracy']):.4f})")
        print(f"Precision: {np.mean(cv_scores['precision']):.4f} (+/- {np.std(cv_scores['precision']):.4f})")
        print(f"Recall:    {np.mean(cv_scores['recall']):.4f} (+/- {np.std(cv_scores['recall']):.4f})")
        print(f"F1-Score:  {np.mean(cv_scores['f1']):.4f} (+/- {np.std(cv_scores['f1']):.4f})")
        
        # Vẽ biểu đồ
        print("\n--- Tạo biểu đồ ---")
        
        # Confusion Matrix
        cm_plot = plot_confusion_matrix(y_test_cls, y_pred_cls, 
                                       'Confusion Matrix - CatBoost Classifier')
        cm_plot.savefig('eval_confusion_matrix_classifier.png', dpi=300, bbox_inches='tight')
        print("Đã lưu: eval_confusion_matrix_classifier.png")
        plt.close()
        
        # ROC Curves
        roc_plot = plot_roc_curves_multiclass(y_test_cls, y_pred_proba_cls)
        roc_plot.savefig('eval_roc_curves_classifier.png', dpi=300, bbox_inches='tight')
        print("Đã lưu: eval_roc_curves_classifier.png")
        plt.close()
        
    except Exception as e:
        print(f"Lỗi khi tải bộ phân loại: {e}")

    # --- Giai đoạn 2: Đánh giá bộ hồi quy ---
    print("\n" + "="*70)
    print("   GIAI ĐOẠN 2: ĐÁNH GIÁ BỘ HỒI QUY")
    print("="*70)
    
    # Chuẩn bị dữ liệu cho bộ hồi quy
    drop_cols_reg = ['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 
                     'order_approved_at', 'order_delivered_timestamp', 'order_estimated_delivery_date', 
                     'delivery_days', 'delivery_days_capped', 'is_extreme', 'delay_vs_estimated', 'delivery_class']
    drop_cols_reg = [c for c in drop_cols_reg if c in df.columns]
    
    X_reg = df.drop(columns=drop_cols_reg)
    y_reg = df['delivery_days_capped']
    
    # Chia dữ liệu (Cùng random_state)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42)
    
    # Dictionary để lưu kết quả
    results = {}
    
    # 1. CatBoost Regressor
    print("\n--- CatBoost Regressor ---")
    try:
        # Chuẩn bị X_test cho CatBoost (đảm bảo chuỗi)
        X_test_cb = X_test_reg.copy()
        X_reg_cb = X_reg.copy()
        for col in cat_features:
            if col in X_test_cb.columns:
                X_test_cb[col] = X_test_cb[col].astype(str)
            if col in X_reg_cb.columns:
                X_reg_cb[col] = X_reg_cb[col].astype(str)
                
        reg_cb = load_model(os.path.join(MODELS_DIR, 'catboost_regressor.pkl'))
        y_pred_cb = reg_cb.predict(X_test_cb)
        
        mae_cb = mean_absolute_error(y_test_reg, y_pred_cb)
        mse_cb = mean_squared_error(y_test_reg, y_pred_cb)
        rmse_cb = np.sqrt(mse_cb)
        r2_cb = r2_score(y_test_reg, y_pred_cb)
        
        print(f"MAE:  {mae_cb:.4f}")
        print(f"MSE:  {mse_cb:.4f}")
        print(f"RMSE: {rmse_cb:.4f}")
        print(f"R²:   {r2_cb:.4f}")
        
        results['CatBoost'] = {
            'mae': mae_cb, 'rmse': rmse_cb, 'r2': r2_cb,
            'y_pred': y_pred_cb
        }
        
        # Cross-Validation
        print("\nCross-Validation (5-Fold)...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Tạo mô hình mới cho CV (không có early_stopping)
        reg_cb_cv = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.03,
            depth=10,
            l2_leaf_reg=3,
            loss_function='MAE',
            cat_features=[col for col in cat_features if col in X_reg_cb.columns],
            verbose=0,
            random_seed=42,
            task_type="CPU"
        )
        
        cv_mae_scores = []
        cv_r2_scores = []
        
        print("Đang chạy 5 folds...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_reg_cb), 1):
            X_train_fold = X_reg_cb.iloc[train_idx]
            X_val_fold = X_reg_cb.iloc[val_idx]
            y_train_fold = y_reg.iloc[train_idx]
            y_val_fold = y_reg.iloc[val_idx]
            
            reg_cb_cv.fit(X_train_fold, y_train_fold, verbose=False)
            y_pred_fold = reg_cb_cv.predict(X_val_fold)
            
            mae_fold = mean_absolute_error(y_val_fold, y_pred_fold)
            r2_fold = r2_score(y_val_fold, y_pred_fold)
            
            cv_mae_scores.append(mae_fold)
            cv_r2_scores.append(r2_fold)
            
            print(f"  Fold {fold}: MAE={mae_fold:.4f}, R²={r2_fold:.4f}")
        
        print(f"\nKết quả CV: MAE={np.mean(cv_mae_scores):.4f} (+/- {np.std(cv_mae_scores):.4f}), R²={np.mean(cv_r2_scores):.4f} (+/- {np.std(cv_r2_scores):.4f})")
        
        # Residual plot
        fig_res_cb = plot_residuals(y_test_reg, y_pred_cb, 'CatBoost Regressor')
        fig_res_cb.savefig('eval_residuals_catboost.png', dpi=300, bbox_inches='tight')
        print("Đã lưu: eval_residuals_catboost.png")
        plt.close()
        
    except Exception as e:
        print(f"Lỗi khi đánh giá CatBoost Regressor: {e}")
        results['CatBoost'] = {'mae': float('inf'), 'rmse': float('inf'), 'r2': -float('inf')}

    # 2. XGBoost Regressor
    print("\n--- XGBoost Regressor ---")
    try:
        # Chuẩn bị X_test cho XGBoost (đảm bảo category)
        X_test_xgb = X_test_reg.copy()
        X_reg_xgb = X_reg.copy()
        for col in cat_features:
            if col in X_test_xgb.columns:
                X_test_xgb[col] = X_test_xgb[col].astype('category')
            if col in X_reg_xgb.columns:
                X_reg_xgb[col] = X_reg_xgb[col].astype('category')
                
        reg_xgb = load_model(os.path.join(MODELS_DIR, 'xgboost_regressor.pkl'))
        y_pred_xgb = reg_xgb.predict(X_test_xgb)
        
        mae_xgb = mean_absolute_error(y_test_reg, y_pred_xgb)
        mse_xgb = mean_squared_error(y_test_reg, y_pred_xgb)
        rmse_xgb = np.sqrt(mse_xgb)
        r2_xgb = r2_score(y_test_reg, y_pred_xgb)
        
        print(f"MAE:  {mae_xgb:.4f}")
        print(f"MSE:  {mse_xgb:.4f}")
        print(f"RMSE: {rmse_xgb:.4f}")
        print(f"R²:   {r2_xgb:.4f}")
        
        results['XGBoost'] = {
            'mae': mae_xgb, 'rmse': rmse_xgb, 'r2': r2_xgb,
            'y_pred': y_pred_xgb
        }
        
        # Cross-Validation
        print("\nCross-Validation (5-Fold)...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Tạo mô hình mới cho CV (không có early_stopping)
        reg_xgb_cv = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=10,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='reg:absoluteerror',
            tree_method='hist',
            enable_categorical=True,
            n_jobs=-1,
            random_state=42
        )
        
        cv_mae_scores = []
        cv_r2_scores = []
        
        print("Đang chạy 5 folds...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_reg_xgb), 1):
            X_train_fold = X_reg_xgb.iloc[train_idx]
            X_val_fold = X_reg_xgb.iloc[val_idx]
            y_train_fold = y_reg.iloc[train_idx]
            y_val_fold = y_reg.iloc[val_idx]
            
            reg_xgb_cv.fit(X_train_fold, y_train_fold, verbose=False)
            y_pred_fold = reg_xgb_cv.predict(X_val_fold)
            
            mae_fold = mean_absolute_error(y_val_fold, y_pred_fold)
            r2_fold = r2_score(y_val_fold, y_pred_fold)
            
            cv_mae_scores.append(mae_fold)
            cv_r2_scores.append(r2_fold)
            
            print(f"  Fold {fold}: MAE={mae_fold:.4f}, R²={r2_fold:.4f}")
        
        print(f"\nKết quả CV: MAE={np.mean(cv_mae_scores):.4f} (+/- {np.std(cv_mae_scores):.4f}), R²={np.mean(cv_r2_scores):.4f} (+/- {np.std(cv_r2_scores):.4f})")
        
        # Residual plot
        fig_res_xgb = plot_residuals(y_test_reg, y_pred_xgb, 'XGBoost Regressor')
        fig_res_xgb.savefig('eval_residuals_xgboost.png', dpi=300, bbox_inches='tight')
        print("Đã lưu: eval_residuals_xgboost.png")
        plt.close()
        
    except Exception as e:
        print(f"Lỗi khi đánh giá XGBoost Regressor: {e}")
        results['XGBoost'] = {'mae': float('inf'), 'rmse': float('inf'), 'r2': -float('inf')}
        
    # So sánh mô hình
    print("\n" + "="*70)
    print("   SO SÁNH MÔ HÌNH HỒI QUY")
    print("="*70)
    
    # Tạo bảng so sánh
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df[['mae', 'rmse', 'r2']]
    comparison_df.columns = ['MAE', 'RMSE', 'R²']
    
    print("\n", comparison_df.to_string())
    
    # Biểu đồ so sánh
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = list(results.keys())
    mae_values = [results[m]['mae'] for m in models]
    rmse_values = [results[m]['rmse'] for m in models]
    r2_values = [results[m]['r2'] for m in models]
    
    # MAE
    axes[0].bar(models, mae_values, color=['steelblue', 'coral'])
    axes[0].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('MAE (days)')
    for i, v in enumerate(mae_values):
        axes[0].text(i, v + 0.05, f'{v:.3f}', ha='center', fontweight='bold')
    
    # RMSE
    axes[1].bar(models, rmse_values, color=['steelblue', 'coral'])
    axes[1].set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('RMSE (days)')
    for i, v in enumerate(rmse_values):
        axes[1].text(i, v + 0.1, f'{v:.3f}', ha='center', fontweight='bold')
    
    # R²
    axes[2].bar(models, r2_values, color=['steelblue', 'coral'])
    axes[2].set_title('R² Score', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('R²')
    axes[2].set_ylim([0, 1])
    for i, v in enumerate(r2_values):
        axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('eval_model_comparison.png', dpi=300, bbox_inches='tight')
    print("\nĐã lưu: eval_model_comparison.png")
    plt.close()
    
    # Xác định mô hình tốt nhất
    best_model = min(results.items(), key=lambda x: x[1]['mae'])[0]
    print(f"\nMô hình hồi quy được đề xuất: {best_model}")
    print(f"   MAE: {results[best_model]['mae']:.4f} days")
    print(f"   R²:  {results[best_model]['r2']:.4f}")
    
    print("\n" + "="*70)
    print("   HOÀN THÀNH ĐÁNH GIÁ HỆ THỐNG")
    print("="*70)
    print("\nĐã tạo các biểu đồ:")
    print("  1. eval_confusion_matrix_classifier.png")
    print("  2. eval_roc_curves_classifier.png")
    print("  3. eval_residuals_catboost.png")
    print("  4. eval_residuals_xgboost.png")
    print("  5. eval_model_comparison.png")

if __name__ == "__main__":
    evaluate_system()
