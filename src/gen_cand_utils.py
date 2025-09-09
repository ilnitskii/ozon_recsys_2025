import polars as pl
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import glob

tracker_files = glob.glob('data/final_apparel_tracker_data_08_action_widget/*/*.parquet')
order_files = glob.glob('data/final_apparel_orders_data_07/*/*.parquet')
test_user_ids = pd.read_parquet('data/ml_ozon_recsys_test.snappy.parquet')
test_user_ids_set = set(test_user_ids['user_id'].to_list())

#################
def get_last_favorite_items(mode, n=50, user_cutoff_time=None, min_date='2025-05-21'):
    """
    mode == 'train' - для отладки, берем данные до cutoff_time
    mode == 'submit' - для сабмита, берем все данные и в соответствии с test_user_ids_list
    n - количество последних добавленных в favorite items
    """
    user_last_purchases = defaultdict(list)
    year, month, day = map(int, min_date.split('-'))
    min_date = pl.datetime(year, month, day)
    for i in tqdm(range(len(tracker_files)-1, -1, -1), desc='Gen last favorite items'):
        file_path = tracker_files[i]
        
        if mode == 'train':
            df = pl.read_parquet(
                file_path, 
                columns=['user_id', 'item_id', 'action_type', 'timestamp']
            ).join(user_cutoff_time, on='user_id'
            ).filter(
                pl.col('action_type').is_in(['to_cart', 'favorite']) & 
                (pl.col('timestamp') > min_date) &
                (pl.col('timestamp') < pl.col('cutoff_time')) # только из обучающей части
            )
        elif mode == 'submit':
            df = pl.read_parquet(
                file_path, 
                columns=['user_id', 'item_id', 'action_type', 'timestamp']
            ).filter(
                pl.col('action_type').is_in(['to_cart', 'favorite']) & 
                pl.col('user_id').is_in(test_user_ids_set) &
                (pl.col('timestamp') > min_date)
            )

        grouped = (
            df.group_by('user_id')
            .agg(pl.col('timestamp'), pl.col('item_id'))
        )
        
        for row in grouped.iter_rows(named=True):
            user_id = row['user_id']
            timestamps = row['timestamp']
            items = row['item_id']

            existing = user_last_purchases[user_id]
            combined = existing + list(zip(timestamps, items))

            combined.sort(key=lambda x: x[0], reverse=True)
            user_last_purchases[user_id] = combined[:n]
        
        if len(user_last_purchases) >= len(test_user_ids_set) and all(
            len(views) >= n for views in user_last_purchases.values()
        ):
            break        

    result_data = []
    for user_id, purchases in user_last_purchases.items():

        purchase_items = [item for _, item in purchases]
        result_data.append({
            'user_id': user_id,
            'item_id': purchase_items
        })

    result_df = pl.DataFrame(result_data)
    test_user_ids_pl = pl.from_pandas(test_user_ids)

    if mode == 'train':
        return result_df
    
    elif mode == 'submit':
        missing_users = test_user_ids_pl.join(
        result_df.select('user_id'),
        on='user_id',
        how='anti'
        )

        missing_users = missing_users.with_columns(
            pl.col('user_id').cast(result_df.schema['user_id'])
        )
        missing_df = missing_users.with_columns(
            pl.Series('item_id', [[]] * missing_users.height)
        )
        result_df_favotites = pl.concat([result_df, missing_df])

        return result_df_favotites
    
#################
def get_last_viewed_def_items(mode, n=50, user_cutoff_time=None, min_date='2025-05-21'):
    """
    mode == 'train' - для отладки, берем данные до cutoff_time
    mode == 'submit' - для сабмита, берем все данные и в соответствии с test_user_ids_list
    n - количество последних добавленных в favorite items
    """
    user_last_purchases = defaultdict(list)
    year, month, day = map(int, min_date.split('-'))
    min_date = pl.datetime(year, month, day)
    for i in tqdm(range(len(tracker_files)-1, -1, -1), desc='Gen last favorite items'):
        file_path = tracker_files[i]
        
        if mode == 'train':
            df = pl.read_parquet(
                file_path, 
                columns=['user_id', 'item_id', 'action_type', 'timestamp']
            ).join(user_cutoff_time, on='user_id'
            ).filter(
                pl.col('action_type').is_in(['review_view', 'view_description']) & 
                (pl.col('timestamp') > min_date) &
                (pl.col('timestamp') < pl.col('cutoff_time')) # только из обучающей части
            )
        elif mode == 'submit':
            df = pl.read_parquet(
                file_path, 
                columns=['user_id', 'item_id', 'action_type', 'timestamp']
            ).filter(
                pl.col('action_type').is_in(['to_cart', 'favorite']) & 
                pl.col('user_id').is_in(test_user_ids_set) &
                (pl.col('timestamp') > min_date)
            )

        grouped = (
            df.group_by('user_id')
            .agg(pl.col('timestamp'), pl.col('item_id'))
        )
        
        for row in grouped.iter_rows(named=True):
            user_id = row['user_id']
            timestamps = row['timestamp']
            items = row['item_id']

            existing = user_last_purchases[user_id]
            combined = existing + list(zip(timestamps, items))

            combined.sort(key=lambda x: x[0], reverse=True)
            user_last_purchases[user_id] = combined[:n]
        
        if len(user_last_purchases) >= len(test_user_ids_set) and all(
            len(views) >= n for views in user_last_purchases.values()
        ):
            break        

    result_data = []
    for user_id, purchases in user_last_purchases.items():

        purchase_items = [item for _, item in purchases]
        result_data.append({
            'user_id': user_id,
            'item_id': purchase_items
        })

    result_df = pl.DataFrame(result_data)
    test_user_ids_pl = pl.from_pandas(test_user_ids)

    if mode == 'train':
        return result_df
    
    elif mode == 'submit':
        missing_users = test_user_ids_pl.join(
        result_df.select('user_id'),
        on='user_id',
        how='anti'
        )

        missing_users = missing_users.with_columns(
            pl.col('user_id').cast(result_df.schema['user_id'])
        )
        missing_df = missing_users.with_columns(
            pl.Series('item_id', [[]] * missing_users.height)
        )
        result_df_favotites = pl.concat([result_df, missing_df])

        return result_df_favotites
###############
def get_last_viewed_items(mode, n=3, user_cutoff_time=None, min_date='2025-05-21'):
    """
    mode == 'train' - для отладки, берем данные до cutoff_time
    mode == 'submit' - для сабмита, берем все данные и в соответствии с test_user_ids_list
    n - количество последни просматриваемых items
    """
    user_last_views = defaultdict(list)
    year, month, day = map(int, min_date.split('-')) 
    min_date = pl.datetime(year, month, day)
    valid_actions = ['review_view', 'view_description']

    for i in tqdm(range(len(tracker_files)-1, -1, -1), desc='Gen last viewed items'):
        file_path = tracker_files[i]
        
        if mode == 'train':

            df = pl.read_parquet(
                file_path, 
                columns=['user_id', 'item_id', 'action_type', 'timestamp']
            ).join(user_cutoff_time, on='user_id'
            ).filter(
                pl.col('action_type').is_in(valid_actions) & 
                (pl.col('timestamp') < pl.col('cutoff_time')) &
                (pl.col('timestamp') > min_date)
            )
        elif mode == 'submit':

            df = pl.read_parquet(
                file_path, 
                columns=['user_id', 'item_id', 'action_type', 'timestamp']
            ).filter(
                pl.col('action_type').is_in(valid_actions) & 
                pl.col('user_id').is_in(test_user_ids_set) &
                (pl.col('timestamp') > min_date)
            )
        grouped = (
            df.sort('timestamp', descending=True)
            .group_by('user_id')
            .agg([
                pl.col('timestamp').head(n).alias('timestamps'),
                pl.col('item_id').head(n).alias('items')
            ])
        )

        for row in grouped.iter_rows(named=True):
            user_id = row['user_id']
            timestamps = row['timestamps']
            items = row['items']
            
            # Объединяем с существующими данными
            new_views = list(zip(timestamps, items))
            combined = user_last_views[user_id] + new_views
            
            # Сортируем и берем n самых новых
            combined.sort(key=lambda x: x[0], reverse=True)
            user_last_views[user_id] = combined[:n]


        # latest_per_user = df.sort('timestamp').group_by('user_id').tail(n)

        # for row in latest_per_user.iter_rows(named=True):
        #     user_id = row['user_id']
        #     user_last_views[user_id].append((row['timestamp'], row['item_id']))
        
        # for user_id in list(user_last_views.keys()):
        #     user_last_views[user_id].sort(key=lambda x: x[0])
        #     user_last_views[user_id] = user_last_views[user_id][-n:]
        
    
        if len(user_last_views) >= len(test_user_ids_set) and all(
            len(views) >= n for views in user_last_views.values()
        ):
            break

    user_last_views = {
        user_id: [item for _, item in items] 
        for user_id, items in user_last_views.items()
    }

    data = []
    for user_id, items in user_last_views.items():
        row = {"user_id": user_id}
        for i in range(n):
            row[f"item_{i+1}"] = items[i] if i < len(items) else None
        data.append(row)

    result_df = pl.DataFrame(data).fill_null(0)

    return result_df

#################
def get_neighbors_of_viewed_items(last_viewed_items, nn_df, k_values, mode='submit'):
    
    max_k = max(k_values.values())

    all_neighbors_df = (
        nn_df.filter(pl.col('rank') <= max_k)
        .group_by('item_id')
        .agg(pl.col('neighbor_item_id').alias('all_neighbors'))
    )

    user_last_views = last_viewed_items
    
    for column, k in k_values.items():
        temp_df = (
            all_neighbors_df
            .with_columns(pl.col('all_neighbors').list.head(k).alias('temp_neighbors'))
            .select(['item_id', 'temp_neighbors'])
        )

        user_last_views = user_last_views.join(
            temp_df,
            left_on=column,
            right_on='item_id',
            how='left'
        ).rename({'temp_neighbors': f'{column}_neighbors'})

    neighbors_columns = [f'{col}_neighbors' for col in k_values.keys()]

    user_last_views = user_last_views.with_columns(
        pl.concat_list([pl.col(col).fill_null([]) for col in neighbors_columns])
        .list.unique()
        .alias('item_id')
    ).select(['user_id', 'item_id'])

    if mode == 'train':
        return user_last_views
    
    elif mode == 'submit':
        
        test_user_ids_pl = pl.from_pandas(test_user_ids)
        user_last_views = user_last_views.filter(
            pl.col('user_id').is_in(test_user_ids_set)
        )
        missing_users = test_user_ids_pl.join(
        user_last_views.select('user_id'),
        on='user_id',
        how='anti'
        )

        missing_users = missing_users.with_columns(
            pl.col('user_id').cast(user_last_views.schema['user_id'])
        )
        missing_df = missing_users.with_columns(
            pl.Series('item_id', [[]] * missing_users.height)
        )
        user_last_views = pl.concat([user_last_views, missing_df])

        return user_last_views

#################
def get_cooccur_neighbors_of_last_delivered_items(cooccurrence_neighbors_of_all_items, mode, n_last=1, user_cutoff_time=None, min_date='2025-05-21'):
    user_last_purchases = defaultdict(list)

    year, month, day = map(int, min_date.split('-')) 
    min_date = pl.datetime(year, month, day)
    for i in tqdm(range(len(order_files)-1, -1, -1), desc='Gen cooccur neighbors of delivered items'):
        file_path = order_files[i]

        if mode == 'train':
            df = pl.read_parquet(
                file_path, 
                columns=['user_id', 'item_id', 'last_status', 'created_timestamp']
            ).join(user_cutoff_time, on='user_id'
            ).filter(
                pl.col('last_status').is_in(['delivered_orders']) & 
                (pl.col('created_timestamp') < pl.col('cutoff_time')) &
                (pl.col('created_timestamp') > min_date)
            )

        elif mode == 'submit':
            df = pl.read_parquet(
                file_path, 
                columns=['user_id', 'item_id', 'last_status', 'created_timestamp']
            ).filter(
                pl.col('last_status').is_in(['delivered_orders']) & 
                pl.col('user_id').is_in(test_user_ids_set) &
                (pl.col('created_timestamp') > min_date)
            )
        grouped = (
            df.sort('created_timestamp', descending=True)
            .group_by('user_id')
            .agg([
                pl.col('created_timestamp').head(n_last).alias('timestamps'),
                pl.col('item_id').head(n_last).alias('items')
            ])
        )

        for row in grouped.iter_rows(named=True):
            user_id = row['user_id']
            timestamps = row['timestamps']
            items = row['items']
            
            # Объединяем с существующими данными
            new_purchases = list(zip(timestamps, items))
            combined = user_last_purchases[user_id] + new_purchases
            
            # Сортируем и берем n_last самых новых
            combined.sort(key=lambda x: x[0], reverse=True)
            user_last_purchases[user_id] = combined[:n_last]

        # for row in df.iter_rows(named=True):
        #     user_id = row['user_id']
        #     user_last_purchases[user_id].append((row['created_timestamp'], row['item_id']))
        
        # for user_id in user_last_purchases:
        #     user_last_purchases[user_id].sort(key=lambda x: x[0], reverse=True)
        #     user_last_purchases[user_id] = user_last_purchases[user_id][:n_last]
        
        # Проверяем, есть ли у всех пользователей достаточно покупок
        if len(user_last_purchases) >= len(test_user_ids_set) and all(
            len(views) >= n_last for views in user_last_purchases.values()
        ):
            break
    result_data = []
    for user_id, purchases in user_last_purchases.items():

        purchase_items = [item for _, item in purchases]

        all_neighbors = set()
        for item in purchase_items:

            neighbors = cooccurrence_neighbors_of_all_items.filter(
                pl.col('item_id') == item
            )['neighbors'].to_list()
            
            if neighbors:
                all_neighbors.update(neighbors[0][:3])

        result_data.append({
            'user_id': user_id,
            'item_id': list(all_neighbors)
        })

    result_df = pl.DataFrame(result_data)
    if mode == 'train':
        return result_df
    
    elif mode == 'submit':

        test_user_ids_pl = pl.from_pandas(test_user_ids)
        missing_users = test_user_ids_pl.join(
        result_df.select('user_id'),
        on='user_id',
        how='anti'
        )

        missing_users = missing_users.with_columns(
            pl.col('user_id').cast(result_df.schema['user_id'])
        )
        missing_df = missing_users.with_columns(
            pl.Series('item_id', [[]] * missing_users.height)
        )
        result_df = pl.concat([result_df, missing_df])

        return result_df
    
#################
def get_popular_items(n=500, min_date='2025-05-21'):

    year, month, day = map(int, min_date.split('-')) 
    min_date = pl.datetime(year, month, day)

    orders_df = (
        pl.scan_parquet(order_files, extra_columns='ignore')
        .filter(pl.col('created_timestamp') > min_date)
        .select(['item_id', 'last_status'])
        .collect()
        )
    popular_items = (
        orders_df.filter(pl.col('last_status') == 'delivered_orders')
        .get_column('item_id')
        .value_counts(sort=True)
        .head(n)
        .get_column('item_id')
        .to_list()
        )
    return popular_items

#################
def get_processed_items(mode, n_last=30, user_cutoff_time=None, min_date='2025-05-21'):
    user_last_purchases = defaultdict(list)
    
    year, month, day = map(int, min_date.split('-')) 
    min_date = pl.datetime(year, month, day)

    for i in tqdm(range(len(order_files)-1, -1, -1), desc='Gen processed items'):
        file_path = order_files[i]

        if mode == 'train':
            df = pl.read_parquet(
                file_path, 
                columns=['user_id', 'item_id', 'last_status', 'created_timestamp']
            ).join(user_cutoff_time, on='user_id'
            ).filter(
                pl.col('last_status').is_in(['proccesed_orders']) & 
                (pl.col('created_timestamp') < pl.col('cutoff_time')) &
                (pl.col('created_timestamp') > min_date)
            )
        elif mode == 'submit':
            df = pl.read_parquet(
                file_path, 
                columns=['user_id', 'item_id', 'last_status', 'created_timestamp']
            ).filter(
                pl.col('last_status').is_in(['proccesed_orders']) & 
                pl.col('user_id').is_in(test_user_ids_set) &
                (pl.col('created_timestamp') > min_date)
            )
        
        grouped = (
            df.sort('created_timestamp', descending=True)
            .group_by('user_id')
            .agg([
                pl.col('created_timestamp').head(n_last).alias('timestamps'),
                pl.col('item_id').head(n_last).alias('items')
            ])
        )

        for row in grouped.iter_rows(named=True):
            user_id = row['user_id']
            timestamps = row['timestamps']
            items = row['items']
            
            # Объединяем с существующими данными
            new_purchases = list(zip(timestamps, items))
            combined = user_last_purchases[user_id] + new_purchases
            
            # Сортируем и берем n_last самых новых
            combined.sort(key=lambda x: x[0], reverse=True)
            user_last_purchases[user_id] = combined[:n_last]

        # for row in df.iter_rows(named=True):
        #     user_id = row['user_id']
        #     user_last_purchases[user_id].append((row['created_timestamp'], row['item_id']))
        
        # for user_id in user_last_purchases:
        #     user_last_purchases[user_id].sort(key=lambda x: x[0], reverse=True)
        #     user_last_purchases[user_id] = user_last_purchases[user_id][:n_last]

        if len(user_last_purchases) >= len(test_user_ids_set) and all(
            len(views) >= n_last for views in user_last_purchases.values()
        ):
            break

    result_data = []
    for user_id, purchases in user_last_purchases.items():
        # Получаем item_id покупок (без timestamp)
        purchase_items = [item for _, item in purchases]

        result_data.append({
            'user_id': user_id,
            'item_id': purchase_items
        })

    result_df = pl.DataFrame(result_data)

    if mode == 'train':
        return result_df
    
    elif mode == 'submit':

        test_user_ids_pl = pl.from_pandas(test_user_ids)
        missing_users = test_user_ids_pl.join(
        result_df.select('user_id'),
        on='user_id',
        how='anti'
        )

        missing_users = missing_users.with_columns(
            pl.col('user_id').cast(result_df.schema['user_id'])
        )
        missing_df = missing_users.with_columns(
            pl.Series('item_id', [[]] * missing_users.height)
        )
        result_df = pl.concat([result_df, missing_df])

        return result_df
    
#################    
def unite_candidates(dfs_to_merge, popular_items, n=300):

    all_user_ids = pl.concat([
        df.select('user_id') for df in dfs_to_merge
    ]).unique().sort('user_id')

    base_df = all_user_ids

    for i, df in enumerate(dfs_to_merge):
        if i == 0:
            merged_df = base_df.join(
                df, 
                on='user_id', 
                how='left'
            )
        else:
            merged_df = merged_df.join(
                df, 
                on='user_id', 
                how='left',
                suffix=f"_{i}"
            )

    recommendation_columns = [
        col for col in merged_df.columns 
        if col != 'user_id' and ('item' in col.lower() or col.startswith('item'))
    ]

    merged_df = merged_df.with_columns(
        *[pl.col(col).fill_null([]) for col in recommendation_columns]
    ).with_columns(
        combined=pl.concat_list(recommendation_columns)
    )

    merged_df = merged_df.with_columns(
        unique_combined=pl.col('combined').list.unique(maintain_order=True)
    )
    
    final_df = merged_df.with_columns(
        final_items=pl.col('unique_combined').list.concat(
            pl.lit(popular_items)
        ).list.eval(
            pl.element().unique(maintain_order=True)
        ).list.head(n)
    ).select(['user_id', 'final_items'])

    final_df = final_df.rename({'final_items': 'item_id'})
    user_pd = final_df.to_pandas()

    return user_pd

def unite_candidates_exploded(dfs_to_merge, popular_items, n=300):

    all_user_ids = pl.concat([
        df.select('user_id') for df in dfs_to_merge
    ]).unique().sort('user_id')

    base_df = all_user_ids

    for i, df in enumerate(dfs_to_merge):
        if i == 0:
            merged_df = base_df.join(
                df, 
                on='user_id', 
                how='left'
            )
        else:
            merged_df = merged_df.join(
                df, 
                on='user_id', 
                how='left',
                suffix=f"_{i}"
            )

    recommendation_columns = [
        col for col in merged_df.columns 
        if col != 'user_id' and ('item' in col.lower() or col.startswith('item'))
    ]

    merged_df = merged_df.with_columns(
        *[pl.col(col).fill_null([]) for col in recommendation_columns]
    ).with_columns(
        combined=pl.concat_list(recommendation_columns)
    )

    merged_df = merged_df.with_columns(
        unique_combined=pl.col('combined').list.unique(maintain_order=True)
    )
    
    final_df = merged_df.with_columns(
        final_items=pl.col('unique_combined').list.concat(
            pl.lit(popular_items)
        ).list.eval(
            pl.element().unique(maintain_order=True)
        ).list.head(n)
    ).select(['user_id', 'final_items'])


    exploded = final_df.explode('final_items')\
                        .rename({'final_items': 'item_id'})
    return exploded.to_pandas()

#################
def gen_submit(res_pd, name='submission'):
    submission_data = []
    for _, row in tqdm(res_pd.iterrows(), total=len(res_pd), desc="Формирование submission"):
        submission_data.append({
            'user_id': row['user_id'],
            'item_id_1 item_id_2 ... item_id_100': ' '.join(map(str, row['item_id']))
        })
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(f'submits/{name}.csv', index=False)