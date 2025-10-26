from data.src.data_processing import load_and_clean_data, prepare_features
from data.src.model import EmissionsPredictor

print('Loading and sampling data...')
df = load_and_clean_data()
X, y = prepare_features(df)
# small sample for quick smoke test
X_sample = X.sample(n=min(500, len(X)), random_state=42).reset_index(drop=True)
y_sample = y.loc[X_sample.index].reset_index(drop=True)

print(f'Sample size: {len(X_sample)}')

pred = EmissionsPredictor()
print('Training (quick mode, no grid search)...')
results = pred.train(X_sample, y_sample, grid_search=False)
print('Results:')
print('R2:', results['r2'])
print('MAE:', results['mae'])

pred.save_model('data/models/trained_model_smoke.pkl')
print('Saved smoke model to data/models/trained_model_smoke.pkl')
