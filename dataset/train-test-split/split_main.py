import sys, os
from source import train_test_split

if __name__ == '__main__':
    data_dir = 'data'
    dataset_dir = os.path.join(data_dir, 'datasets')
    splitting_dir = os.path.join(data_dir, 'splitting')
    files = {
        'create_single_input': True,
        'n': -1,
        'json_dir': {
            'phishes': os.path.join(dataset_dir, 'clean/phishes'),
            'benigns': os.path.join(dataset_dir, 'clean/benigns'),
        },
        'single_input': {
            'phishes': os.path.join(dataset_dir, 'single/phishes.json'),
            'benigns': os.path.join(dataset_dir, 'single/benigns.json')
        },

        'create_fv_files': True,
        'url_fv': {
            'phishes': os.path.join(splitting_dir, 'fv/url_phishes.pkl'),
            'benigns': os.path.join(splitting_dir, 'fv/url_benigns.pkl')
        },
        'html_fv': {
            'phishes': os.path.join(splitting_dir, 'fv/html_phishes.pkl'),
            'benigns': os.path.join(splitting_dir, 'fv/html_benigns.pkl')
        },

        'create_lsh_models': True,
        'n_vectors': {
            'url': 8,
            'html': 16
        },
        'url_lsh': {
            'phishes': os.path.join(splitting_dir, 'lsh/url_phishes.pkl'),
            'benigns': os.path.join(splitting_dir, 'lsh/url_benigns.pkl')
        },
        'html_lsh': {
            'phishes': os.path.join(splitting_dir, 'lsh/html_phishes.pkl'),
            'benigns': os.path.join(splitting_dir, 'lsh/html_benigns.pkl')
        },

        'create_nearest_neighbors': True,
        'max_search_radius': 2,
        'url_nn': {
            'phishes': os.path.join(splitting_dir, 'nn/url_phishes.pkl'),
            'benigns': os.path.join(splitting_dir, 'nn/url_benigns.pkl')
        },
        'html_nn': {
            'phishes': os.path.join(splitting_dir, 'nn/html_phishes.pkl'),
            'benigns': os.path.join(splitting_dir, 'nn/html_benigns.pkl')
        },
        
        'split_train_test': True,
        'dist_thresholds': {
            'url': {'phishes': 0.1, 'benigns': 0.1},
            'html': {'phishes': 0.2, 'benigns': 0.08},
        },
        'splits':{
            'train': {
                'phishes': os.path.join(splitting_dir, 'splits/train/phishes'),
                'benigns': os.path.join(splitting_dir, 'splits/train/benigns')
            },
            'test': {
                'phishes': os.path.join(splitting_dir, 'splits/test/phishes'),
                'benigns': os.path.join(splitting_dir, 'splits/test/benigns')
            }
        },
    }

    seed = 42
    train_ratio = 0.8

    train_test_split.main(files, seed, train_ratio)