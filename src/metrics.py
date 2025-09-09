from collections import defaultdict
import numpy as np

def precision_at_k(y_true, y_pred, k=100):
    """Precision@K - доля релевантных среди топ-K рекомендаций"""
    y_pred = y_pred[:k]
    if len(y_pred) == 0:
        return 0.0
    relevant = set(y_true)
    hits = sum(1 for item in y_pred if item in relevant)
    return hits / len(y_pred)

def recall_at_k(y_true, y_pred, k=100):
    """Recall@K - долю найденных релевантных среди всех релевантных"""
    y_pred = y_pred[:k]
    relevant = set(y_true)
    if len(relevant) == 0:
        return 1.0  # Если нет релевантных, считаем recall максимальным
    hits = sum(1 for item in y_pred if item in relevant)
    return hits / len(relevant)

def hit_rate_at_k(y_true, y_pred, k=100):
    """Hit Rate@K - есть ли хотя бы 1 релевантный в топ-K"""
    y_pred = y_pred[:k]
    relevant = set(y_true)
    return 1.0 if any(item in relevant for item in y_pred) else 0.0

def mrr_at_k(y_true, y_pred, k=100):
    """Mean Reciprocal Rank - обратный ранг первого релевантного"""
    y_pred = y_pred[:k]
    relevant = set(y_true)
    for i, item in enumerate(y_pred, 1):
        if item in relevant:
            return 1.0 / i
    return 0.0

def ndcg_at_k(y_true, y_pred, k=100, item_relevance=None):
    """Normalized Discounted Cumulative Gain"""
    y_pred = y_pred[:k]
    if item_relevance is None:
        item_relevance = defaultdict(int, {item: 1 for item in y_true})
    
    # DCG
    dcg = 0.0
    for i, item in enumerate(y_pred, 1):
        rel = item_relevance.get(item, 0)
        dcg += (2 ** rel - 1) / np.log2(i + 1)
    
    # IDCG
    top_relevant = sorted([item_relevance[item] for item in y_true], reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(top_relevant, 1):
        idcg += (2 ** rel - 1) / np.log2(i + 1)
    
    return dcg / idcg if idcg > 0 else 0.0

def calculate_metrics_for_all_users(validation_df, k_values=[10, 20, 50, 100]):
    """Вычисляет метрики для всех пользователей по нескольким K"""
    
    # Инициализация результатов
    metric_names = ['P', 'R', 'HR', 'MRR', 'nDCG']
    results = {}
    for metric in metric_names:
        for k in k_values:
            results[f'{metric}@{k}'] = []
    
    # Вычисляем метрики для каждого пользователя
    for index, row in validation_df.iterrows():
        y_true = row['true_items']
        y_pred = row['predicted_items'] if isinstance(row['predicted_items'], (list, np.ndarray)) else []
        
        for k in k_values:
            # Вычисление всех метрик для одного K
            metrics = {
                f'P@{k}': precision_at_k(y_true, y_pred, k),
                f'R@{k}': recall_at_k(y_true, y_pred, k),
                f'HR@{k}': hit_rate_at_k(y_true, y_pred, k),
                f'MRR@{k}': mrr_at_k(y_true, y_pred, k),
                f'nDCG@{k}': ndcg_at_k(y_true, y_pred, k)
            }
            
            for metric_name, value in metrics.items():
                results[metric_name].append(value)
    
    # Агрегируем результаты (среднее по пользователям)
    aggregated_results = {}
    for metric_name, values in results.items():
        aggregated_results[metric_name] = np.mean(values)
    
    return aggregated_results, results