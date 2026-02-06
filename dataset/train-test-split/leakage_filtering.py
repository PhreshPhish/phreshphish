import os
import logging
from utils import (
    load_df, save_tfidf_model, save_vectors, save_lsh, save_nn
)


def main(params):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename='train_test_split.log', 
        encoding='utf-8', level=logging.DEBUG
    )
    
    if params['create_tfidf_files']:
        print(f"Creating tfidf model files...")
        for class_type in params['tfidf_class_types']:
            print(f"  for {class_type}...")
            dir = params['train_dir'][class_type]
            assert os.path.exists(dir), f"Train directory {dir} does not exist."
            train_df = load_df(dir)
            print(f"    Loaded {len(train_df)} training samples from {dir}")

            for feat_type in params['tfidf_feat_types']:
                print(f"    for {feat_type}...")
                tfidf_path = params[f'{feat_type}_tfidf'][class_type]
                os.makedirs(os.path.dirname(tfidf_path), exist_ok=True)
                save_tfidf_model(train_df, class_type, feat_type, tfidf_path)
                print(f"    Saved tfidf model at {tfidf_path}")
    
    if params['create_fv_files']:
        print(f"Creating feature vector files...")
        for class_type in params['fv_class_types']:
            for split_type in params['fv_split_types']:
                dir = params[f'{split_type}_dir'][class_type]
                assert os.path.exists(dir), f"{split_type.capitalize()} directory {dir} does not exist."
                df = load_df(dir)
                print(f"    Loaded {len(df)} samples from {dir}")

                for feat_type in params['fv_feat_types']:
                    print(f"    for {feat_type}...")
                    tfidf_path = params[f'{feat_type}_tfidf'][class_type]
                    assert os.path.exists(tfidf_path), f"TFIDF file {tfidf_path} does not exist."

                    fv_path = params[f'{feat_type}_fv'][split_type][class_type]
                    os.makedirs(os.path.dirname(fv_path), exist_ok=True)
                    save_vectors(df, class_type, feat_type, tfidf_path, fv_path)
                    print(f"    Saved feature vectors at {fv_path}")

    if params['create_lsh_models']:
        print(f"Creating LSH models...")
        for class_type in params['lsh_class_types']:
            for feat_type in params['lsh_feat_types']:
                print(f"  for {class_type} - {feat_type}...")
                fv_path = params[f'{feat_type}_fv']['train'][class_type]
                assert os.path.exists(fv_path), f"Feature vector file {fv_path} does not exist."
                n_vectors = params['n_vectors'][feat_type]
                lsh_path = params[f'{feat_type}_lsh'][class_type]
                os.makedirs(os.path.dirname(lsh_path), exist_ok=True)
                save_lsh(class_type, feat_type, fv_path, n_vectors, lsh_path)
                print(f"    Saved LSH model at {lsh_path}")

    if params['create_nearest_neighbors']:
        print(f"Creating nearest neighbors of test...")
        for class_type in params['nn_class_types']:
            for feat_type in params['nn_feat_types']:
                print(f"  for {class_type} - {feat_type}...")
                train_fv_path = params[f'{feat_type}_fv']['train'][class_type]
                test_fv_path = params[f'{feat_type}_fv']['test'][class_type]
                assert os.path.exists(train_fv_path), f"Train feature vector file {train_fv_path} does not exist."
                assert os.path.exists(test_fv_path), f"Test feature vector file {test_fv_path} does not exist."
                lsh_path = params[f'{feat_type}_lsh'][class_type]
                assert os.path.exists(lsh_path), f"LSH file {lsh_path} does not exist."
                max_search_radius = params['max_search_radius']
                nn_path = params[f'{feat_type}_nn'][class_type]
                os.makedirs(os.path.dirname(nn_path), exist_ok=True)
                num_workers = int(os.cpu_count() * 0.9)
                save_nn(class_type, feat_type, train_fv_path, test_fv_path, lsh_path, max_search_radius, nn_path, num_workers=num_workers)
                print(f"    Saved nearest neighbors at {nn_path}")



if __name__ == '__main__':
    data_dir = './data'
    splitting_dir = os.path.join(data_dir, 'splitting')
    os.makedirs(splitting_dir, exist_ok=True)
    params = {
        'create_tfidf_files': False,
        'tfidf_class_types': ['phishes', 'benigns'],
        'tfidf_feat_types': ['url', 'html'],
        'train_dir': {
            'phishes': os.path.join(data_dir, 'splits/train/phishing'),
            'benigns': os.path.join(data_dir, 'splits/train/benign')
        },
        'url_tfidf': {
            'phishes': os.path.join(splitting_dir, 'tfidf/url_phishes.pkl'),
            'benigns': os.path.join(splitting_dir, 'tfidf/url_benigns.pkl')
        },
        'html_tfidf': {
            'phishes': os.path.join(splitting_dir, 'tfidf/html_phishes.pkl'),
            'benigns': os.path.join(splitting_dir, 'tfidf/html_benigns.pkl')
        },

        'create_fv_files': False,
        'fv_class_types': ['phishes', 'benigns'],
        'fv_split_types': ['train', 'test'],
        'fv_feat_types': ['url', 'html'],
        'test_dir': {
            'phishes': os.path.join(data_dir, 'splits/test-before-filtering-lang-balanced/phishing'),
            'benigns': os.path.join(data_dir, 'splits/test-before-filtering-lang-balanced/benign')
        },
        'url_fv': {
            'train': {
                'phishes': os.path.join(splitting_dir, 'fv/train/url_phishes.pkl'),
                'benigns': os.path.join(splitting_dir, 'fv/train/url_benigns.pkl')
            },
            'test': {
                'phishes': os.path.join(splitting_dir, 'fv/test-before-filtering-lang-balanced/url_phishes.pkl'),
                'benigns': os.path.join(splitting_dir, 'fv/test-before-filtering-lang-balanced/url_benigns.pkl')
            }
        },
        'html_fv': {
            'train': {
                'phishes': os.path.join(splitting_dir, 'fv/train/html_phishes.pkl'),
                'benigns': os.path.join(splitting_dir, 'fv/train/html_benigns.pkl')
            },
            'test': {
                'phishes': os.path.join(splitting_dir, 'fv/test-before-filtering-lang-balanced/html_phishes.pkl'),
                'benigns': os.path.join(splitting_dir, 'fv/test-before-filtering-lang-balanced/html_benigns.pkl')
            }
        },

        'create_lsh_models': False,
        'lsh_class_types': ['phishes', 'benigns'],
        'lsh_feat_types': ['url', 'html'],
        'n_vectors': {
            'url': 8,
            'html': 16
        },
        'url_lsh': {
            'phishes': os.path.join(splitting_dir, 'lsh/url_phishes_8.pkl'),
            'benigns': os.path.join(splitting_dir, 'lsh/url_benigns_8.pkl')
        },
        'html_lsh': {
            'phishes': os.path.join(splitting_dir, 'lsh/html_phishes_16.pkl'),
            'benigns': os.path.join(splitting_dir, 'lsh/html_benigns_16.pkl')
        },

        'create_nearest_neighbors': False,
        'nn_class_types': ['phishes', 'benigns'],
        'nn_feat_types': ['url', 'html'],
        'max_search_radius': 2,
        'url_nn': {
            'phishes': os.path.join(splitting_dir, 'nn/url_phishes_8_2.pkl'),
            'benigns': os.path.join(splitting_dir, 'nn/url_benigns_8_2.pkl')
        },
        'html_nn': {
            'phishes': os.path.join(splitting_dir, 'nn/html_phishes_16_2.pkl'),
            'benigns': os.path.join(splitting_dir, 'nn/html_benigns_16_2.pkl')
        },
    }

    main(params)