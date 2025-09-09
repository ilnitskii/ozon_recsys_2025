from src.metrics import calculate_metrics_for_all_users
import matplotlib.pyplot as plt

def validate_recommendations(candidates_df, val_n_orders, k_values=[10, 20, 50, 100]):
   
    # Объединяем данные
    validation_df = candidates_df.merge(
        val_n_orders, 
        on='user_id', 
        how='right'
    )

    validation_df = validation_df.rename(columns={
        'item_id': 'predicted_items',
        'item_ids': 'true_items'
    })
    
    # Вычисляем метрики
    aggregated_metrics, user_metrics = calculate_metrics_for_all_users(
        validation_df, k_values
    )

    
    return aggregated_metrics, user_metrics

def plot_metrics_by_k(aggregated_metrics, k_values):
    """Визуализация метрик по разным K"""
    
    metrics = ['P', 'R', 'HR', 'MRR', 'nDCG']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [aggregated_metrics.get(f'{metric}@{k}', 0) for k in k_values]
        axes[i].plot(k_values, values, 'o-', linewidth=2, markersize=8)
        axes[i].set_title(f'{metric}@K')
        axes[i].set_xlabel('K')
        axes[i].set_ylabel(metric)
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()